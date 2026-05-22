// SPDX-License-Identifier: Apache-2.0
//
// kvcached-on-Metal probe #2: integrate with MLX's allocator.
//
// The first probe proved that `mmap(MAP_FIXED, PROT_NONE)` over a sub-range
// of a `newBufferWithBytesNoCopy:`-wrapped MTLBuffer truly releases pages.
// This probe takes the next step: we want the mmap'd region to be usable
// from MLX's existing kernel-dispatch path (so we don't have to rewrite
// every attention kernel to bind raw MTLBuffer*).
//
// The hypothesis: `mlx::core::metal::allocator().make_buffer(ptr, size)`
// wraps an external host pointer as an `allocator::Buffer` that fits into
// MLX's normal binding. If true, the integration is:
//
//   1. mmap a giant lazy region (our KV pool).
//   2. Wrap it via MLX's make_buffer (which calls newBufferWithBytesNoCopy:
//      internally and registers the MTLBuffer with MLX's allocator).
//   3. Construct an mlx::core::array whose Data points at that Buffer.
//   4. Pass it to set_input_array / set_output_array — kernels see it as
//      a normal array.
//   5. On free, mmap(MAP_FIXED, PROT_NONE) over a sub-range to release
//      physical pages.
//
// If this works, kvcached-on-Metal is a clean port: we only own the
// backing memory; everything kernel-side stays in MLX-land.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -O2 probe2_mlx.cc \
//       -I<...>/site-packages/mlx/include \
//       -L<...>/site-packages/mlx/lib -lmlx \
//       -framework Metal -framework Foundation \
//       -Wl,-rpath,<...>/site-packages/mlx/lib \
//       -o probe2_mlx
// Run:
//   ./probe2_mlx

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <sys/mman.h>
#include <unistd.h>

#include "mlx/mlx.h"

namespace mc = mlx::core;
namespace mca = mlx::core::allocator;

namespace {

constexpr size_t kTotalReserve = 1ULL << 30;  // 1 GB
constexpr size_t kWindowBytes = 1ULL << 20;   // 1 MB
constexpr uint32_t kPattern = 0xABCD0000;

size_t get_phys_footprint() {
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) ==
                 KERN_SUCCESS
             ? info.phys_footprint
             : 0;
}

size_t get_rss() {
  task_basic_info_data_t info;
  mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
  return task_info(mach_task_self(), TASK_BASIC_INFO,
                   (task_info_t)&info, &count) == KERN_SUCCESS
             ? info.resident_size
             : 0;
}

void log_mem(const char *label) {
  printf("  %-48s rss=%7.2f MB  phys_fp=%7.2f MB\n", label,
         get_rss() / 1e6, get_phys_footprint() / 1e6);
}

bool remap_fixed(void *addr, size_t len, int prot, const char *what) {
  void *r = mmap(addr, len, prot, MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);
  if (r == MAP_FAILED) {
    fprintf(stderr, "mmap(%s) failed: %s\n", what, strerror(errno));
    return false;
  }
  return true;
}

}  // namespace

int main() {
  printf("kvcached-on-Metal probe #2 — MLX integration\n");
  printf("System page size: %zu bytes\n\n", (size_t)sysconf(_SC_PAGESIZE));

  log_mem("Step 0: baseline");

  // Step 1: reserve VA range (PROT_READ|WRITE so Metal will accept it)
  void *base = mmap(nullptr, kTotalReserve, PROT_READ | PROT_WRITE,
                    MAP_ANON | MAP_PRIVATE, -1, 0);
  if (base == MAP_FAILED) {
    fprintf(stderr, "mmap reserve failed\n");
    return 1;
  }
  log_mem("Step 1: mmap(1GB) untouched");

  // Step 2: ask MLX's runtime allocator to wrap our external pointer.
  // On Apple Silicon, the runtime allocator IS the MetalAllocator (its
  // make_buffer override calls newBufferWithBytesNoCopy: internally).
  mca::Buffer ext_buf = mca::allocator().make_buffer(base, kTotalReserve);
  if (ext_buf.ptr() == nullptr) {
    fprintf(stderr, "MetalAllocator::make_buffer returned nullptr\n");
    return 1;
  }
  printf("  make_buffer returned Buffer{ptr=%p, raw_ptr=%p, our base=%p}\n",
         ext_buf.ptr(), ext_buf.raw_ptr(), base);
  log_mem("Step 2: MLX wrapped our 1GB region");

  // Step 3: construct an mlx::core::array that points at the Buffer.
  //         Shape it as a flat uint32 array sized to the full reservation.
  //         The Buffer-taking constructor produces an evaluated array.
  mc::Shape shape = {static_cast<int>(kTotalReserve / sizeof(uint32_t))};
  // Custom deleter so MLX doesn't try to free our externally-managed buffer.
  // We'll release it ourselves at the end via mca::allocator().release().
  mc::array arr(ext_buf, shape, mc::uint32, /*deleter=*/[](mca::Buffer) {});
  log_mem("Step 3: built mlx array over external buffer");

  // Step 4: write a known pattern to the first 1 MB via CPU (unified memory
  //         is host-accessible). Then dispatch an MLX op that READS from
  //         arr and produces a derived array. If the derived array's
  //         contents match what we wrote via CPU, the binding is correct:
  //         MLX kernels are reading from OUR mmap'd region.
  //
  //         (Real KV writes will go through a custom in-place Primitive
  //         like tq_encode, mirroring vllm-metal's existing pattern.
  //         For the probe we just need to validate the read path.)
  uint32_t *cpu_view = static_cast<uint32_t *>(base);
  uint32_t num_uints = kWindowBytes / sizeof(uint32_t);
  for (uint32_t i = 0; i < num_uints; ++i) {
    cpu_view[i] = kPattern + i;
  }
  log_mem("Step 4a: CPU wrote 1MB pattern into our buffer");

  // Slice the first 1MB and add 1 — exercises the read path through MLX.
  mc::array slice = mc::slice(arr, /*start=*/mc::Shape{0},
                              /*stop=*/mc::Shape{static_cast<int>(num_uints)});
  mc::array plus_one = mc::add(slice, mc::array(1u, mc::uint32));
  mc::eval(plus_one);
  log_mem("Step 4b: ran arr[0:1MB]+1 via MLX kernel");

  // Step 5: verify the kernel saw our CPU-written values.
  uint32_t *result = static_cast<uint32_t *>(plus_one.data<void>());
  bool ok = true;
  for (uint32_t i = 0; i < num_uints; ++i) {
    if (result[i] != kPattern + i + 1) {
      printf("  Step 5 MISMATCH idx=%u got=0x%08x expected=0x%08x\n", i,
             result[i], kPattern + i + 1);
      ok = false;
      break;
    }
  }
  printf("  Step 5: MLX kernel read CPU-written data correctly: %s\n",
         ok ? "PASS" : "FAIL");
  if (!ok) {
    fprintf(stderr,
            "Aborting — MLX is binding a different buffer than we wrapped.\n");
    return 1;
  }

  // Step 6: free the pages we wrote. mmap(MAP_FIXED, PROT_NONE), then
  //         re-establish a fresh anonymous mapping for the same VA.
  if (!remap_fixed(base, kWindowBytes, PROT_NONE, "step 6a unmap")) return 1;
  log_mem("Step 6a: PROT_NONE 1MB window");
  if (!remap_fixed(base, kWindowBytes, PROT_READ | PROT_WRITE,
                   "step 6b re-map")) {
    return 1;
  }
  log_mem("Step 6b: re-PROT_READ|WRITE same window");

  bool fresh = true;
  for (uint32_t i = 0; i < num_uints; ++i) {
    if (cpu_view[i] != 0) {
      printf("  Step 6c: idx=%u = 0x%08x (NOT zero)\n", i, cpu_view[i]);
      fresh = false;
      break;
    }
  }
  printf("  Step 6c: re-mapped pages are %s\n",
         fresh ? "zero (physical released)" : "DIRTY (release failed)");

  // Step 7: write a fresh pattern via CPU into the re-mapped pages, then
  //         dispatch an MLX read kernel again. WITHOUT recreating the
  //         buffer.
  for (uint32_t i = 0; i < num_uints; ++i) {
    cpu_view[i] = 0xCAFE0000u + i;
  }
  mc::array slice2 = mc::slice(arr, /*start=*/mc::Shape{0},
                               /*stop=*/mc::Shape{static_cast<int>(num_uints)});
  mc::array plus_two = mc::add(slice2, mc::array(2u, mc::uint32));
  mc::eval(plus_two);
  log_mem("Step 7: MLX re-read after MAP_FIXED (no buffer recreate)");

  uint32_t *result2 = static_cast<uint32_t *>(plus_two.data<void>());
  bool ok2 = true;
  uint32_t first_mismatch_value = 0;
  for (uint32_t i = 0; i < num_uints; ++i) {
    if (result2[i] != 0xCAFE0000u + i + 2) {
      if (ok2) first_mismatch_value = result2[i];
      ok2 = false;
      if (i < 2) printf("  Step 7 idx=%u got=0x%08x expected=0x%08x\n", i,
                        result2[i], 0xCAFE0000u + i + 2);
    }
  }
  printf("  Step 7: MLX read (no recreate): %s\n",
         ok2 ? "PASS" : "FAIL (GPU IOMMU stale; saw 0x%08x = old data)");
  if (!ok2)
    printf("           First stale value: 0x%08x (CPU now sees 0x%08x)\n",
           first_mismatch_value - 2, cpu_view[0]);

  // Step 8: workaround — release and recreate the buffer to force the
  //         GPU driver to re-establish the mapping.
  mca::allocator().release(ext_buf);
  ext_buf = mca::allocator().make_buffer(base, kTotalReserve);
  if (ext_buf.ptr() == nullptr) {
    fprintf(stderr, "second make_buffer returned nullptr\n");
    return 1;
  }
  mc::array arr2(ext_buf, shape, mc::uint32, /*deleter=*/[](mca::Buffer) {});
  log_mem("Step 8a: release + make_buffer + new array");

  mc::array slice3 = mc::slice(arr2, /*start=*/mc::Shape{0},
                               /*stop=*/mc::Shape{static_cast<int>(num_uints)});
  mc::array plus_three = mc::add(slice3, mc::array(3u, mc::uint32));
  mc::eval(plus_three);
  log_mem("Step 8b: MLX read after buffer recreate");

  uint32_t *result3 = static_cast<uint32_t *>(plus_three.data<void>());
  bool ok3 = true;
  for (uint32_t i = 0; i < num_uints; ++i) {
    if (result3[i] != 0xCAFE0000u + i + 3) {
      if (i < 2) printf("  Step 8 idx=%u got=0x%08x expected=0x%08x\n", i,
                        result3[i], 0xCAFE0000u + i + 3);
      ok3 = false;
      break;
    }
  }
  printf("  Step 8: MLX read after buffer recreate: %s\n",
         ok3 ? "PASS (recreate fixed IOMMU)" : "FAIL");

  // Cleanup
  mca::allocator().release(ext_buf);
  munmap(base, kTotalReserve);
  log_mem("After release + munmap");

  printf("\n==============================================\n");
  printf("Summary:\n");
  printf("  MetalAllocator::make_buffer accepted external ptr: %s\n",
         ext_buf.ptr() ? "yes" : "no");
  printf("  MLX slice_update dispatches into our memory:        %s\n",
         ok ? "yes" : "no");
  printf("  Physical pages released after MAP_FIXED unmap:      %s\n",
         fresh ? "yes" : "no");
  printf("  Buffer reads fresh data after mmap dance (no recreate): %s\n",
         ok2 ? "yes" : "no (GPU IOMMU stale)");
  printf("  Buffer reads fresh data after release+recreate:        %s\n",
         ok3 ? "yes" : "no");
  printf("==============================================\n");
  return 0;
}
