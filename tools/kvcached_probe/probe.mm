// SPDX-License-Identifier: Apache-2.0
//
// kvcached-on-Metal viability probe.
//
// Tests whether `mmap(MAP_FIXED) + MTLBuffer.newBufferWithBytesNoCopy:` can
// be used to implement kvcached-style elastic GPU memory on Apple Silicon.
//
// The CUDA kvcached design relies on `cuMemAddressReserve/cuMemMap/cuMemUnmap`
// to decouple virtual addresses from physical pages and reclaim physical
// memory back to the device on free. Apple has no documented equivalent for
// MTLBuffer, but Apple Silicon's unified memory means the GPU and CPU share
// VM mappings — so in theory we can manage the VA range ourselves with mmap
// and let Metal see the result via newBufferWithBytesNoCopy:.
//
// This harness tests 6 things in order:
//   1. Reserve a 1 GB VA range with PROT_NONE (no physical backing).
//   2. Wrap the whole range as a single MTLBuffer.
//   3. Map a 1 MB window into the range with MAP_FIXED (PROT_READ|WRITE).
//   4. Dispatch a Metal kernel that writes a known pattern to that window.
//   5. Verify on the CPU side that the kernel's writes are visible.
//   6. Unmap the window (PROT_NONE again) and re-map fresh — verify the
//      page contents are zero (i.e. the original physical pages were
//      released back to the OS).
//
// At every step we print rss + phys_footprint so we can see whether IOKit
// accounting tracks our VM mappings.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -O2 probe.mm \
//       -framework Metal -framework Foundation -o probe
// Run:
//   ./probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr size_t kTotalReserve = 1ULL << 30;   // 1 GB
constexpr size_t kMapWindow = 1ULL << 20;      // 1 MB

const char *kShaderSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void fill_with_pattern(device uint* buf [[buffer(0)]],\n"
    "                              constant uint& base [[buffer(1)]],\n"
    "                              uint tid [[thread_position_in_grid]]) {\n"
    "  buf[tid] = base + tid;\n"
    "}\n";

size_t get_phys_footprint() {
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  kern_return_t kr =
      task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
  return (kr == KERN_SUCCESS) ? info.phys_footprint : 0;
}

size_t get_rss() {
  task_basic_info_data_t info;
  mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
  kern_return_t kr =
      task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count);
  return (kr == KERN_SUCCESS) ? info.resident_size : 0;
}

void log_mem(const char *label) {
  printf("  %-44s rss=%7.2f MB  phys_fp=%7.2f MB\n", label,
         get_rss() / 1e6, get_phys_footprint() / 1e6);
}

bool remap_fixed(void *addr, size_t len, int prot, const char *what) {
  void *result =
      mmap(addr, len, prot, MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);
  if (result == MAP_FAILED) {
    fprintf(stderr, "mmap(%s) failed: %s (errno=%d)\n", what, strerror(errno),
            errno);
    return false;
  }
  if (result != addr) {
    fprintf(stderr, "mmap(%s) returned %p, expected %p\n", what, result, addr);
    return false;
  }
  return true;
}

}  // namespace

int main() {
  @autoreleasepool {
    const size_t kPage = sysconf(_SC_PAGESIZE);
    printf("System page size: %zu bytes\n", kPage);
    printf("Total reservation: %.2f GB\n", kTotalReserve / 1e9);
    printf("Map window:        %.2f MB\n\n", kMapWindow / 1e6);

    log_mem("Step 0: baseline");

    // ----- Step 1: reserve VA range -------------------------------------
    // First attempt was PROT_NONE; Metal rejects that. Use PROT_READ|WRITE
    // — macOS still lazy-pages the region (only touched pages become
    // resident), so this isn't worse than PROT_NONE for memory accounting,
    // and Metal accepts the wrap. The relevant question becomes whether we
    // can later MAP_FIXED over sub-regions to release pages on free.
    void *base = mmap(nullptr, kTotalReserve, PROT_READ | PROT_WRITE,
                      MAP_ANON | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) {
      fprintf(stderr, "mmap reserve failed: %s\n", strerror(errno));
      return 1;
    }
    log_mem("Step 1: mmap(1GB, PROT_READ|WRITE) untouched");

    // ----- Step 2: wrap as MTLBuffer ------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "No Metal device available\n");
      return 1;
    }
    printf("\nMetal device: %s\n", [[device name] UTF8String]);

    id<MTLBuffer> buf =
        [device newBufferWithBytesNoCopy:base
                                  length:kTotalReserve
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];
    if (!buf) {
      fprintf(stderr,
              "newBufferWithBytesNoCopy failed (Metal rejected the mapping)\n");
      return 1;
    }
    log_mem("Step 2: newBufferWithBytesNoCopy(1GB)");

    // ----- Step 3: touch some pages via CPU so we have a concrete         ----
    //              non-lazy baseline to compare against ----------------
    // (The first GPU dispatch in step 4 will also touch pages; doing it
    // here makes the per-step phys_footprint deltas easier to read.)
    uint8_t *cpu_bytes = (uint8_t *)base;
    for (size_t off = 0; off < kMapWindow; off += kPage) {
      cpu_bytes[off] = 0xAB;
    }
    log_mem("Step 3: CPU touched first 1MB (one byte per page)");

    // ----- Step 4: build Metal kernel + dispatch -------------------------
    NSError *err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc]
                             options:nil
                               error:&err];
    if (!lib) {
      fprintf(stderr, "shader compile failed: %s\n",
              [[err description] UTF8String]);
      return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"fill_with_pattern"];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      fprintf(stderr, "pipeline failed: %s\n", [[err description] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    const uint32_t kPattern = 0xABCD0000;
    const uint32_t num_uints = (uint32_t)(kMapWindow / sizeof(uint32_t));

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&kPattern length:sizeof(kPattern) atIndex:1];
    NSUInteger tgw = MIN((NSUInteger)num_uints,
                         (NSUInteger)[pso maxTotalThreadsPerThreadgroup]);
    [enc dispatchThreads:MTLSizeMake(num_uints, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if ([cb status] != MTLCommandBufferStatusCompleted) {
      fprintf(stderr,
              "Step 4 dispatch failed: status=%ld error=%s\n",
              (long)[cb status],
              [cb error] ? [[[cb error] description] UTF8String] : "(none)");
      return 1;
    }
    log_mem("Step 4: GPU wrote 1MB into mapped window");

    // ----- Step 5: verify on CPU side -----------------------------------
    uint32_t *cpu_view = (uint32_t *)base;
    bool ok = true;
    for (uint32_t i = 0; i < num_uints; ++i) {
      if (cpu_view[i] != kPattern + i) {
        fprintf(stderr,
                "  MISMATCH at index %u: got 0x%08x expected 0x%08x\n", i,
                cpu_view[i], kPattern + i);
        ok = false;
        break;
      }
    }
    printf("  Step 5: CPU verify of GPU writes: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    // ----- Step 6: unmap the window, re-map fresh, check page contents -
    if (!remap_fixed(base, kMapWindow, PROT_NONE, "step 6 unmap")) return 1;
    log_mem("Step 6a: re-PROT_NONE the window");

    if (!remap_fixed(base, kMapWindow, PROT_READ | PROT_WRITE,
                     "step 6 re-map")) {
      return 1;
    }
    log_mem("Step 6b: re-map window PROT_READ|WRITE");

    bool fresh = true;
    for (uint32_t i = 0; i < num_uints; ++i) {
      if (cpu_view[i] != 0) {
        printf("  Step 6c: page contents at index %u = 0x%08x (NOT zero)\n", i,
               cpu_view[i]);
        fresh = false;
        break;
      }
    }
    printf("  Step 6c: re-mapped pages are %s\n",
           fresh ? "zero (physical released and refreshed)"
                 : "DIRTY (mmap didn't release the pages)");

    // ----- Cleanup ------------------------------------------------------
    munmap(base, kTotalReserve);
    log_mem("After munmap(1GB)");

    printf("\n==============================================\n");
    printf("Summary:\n");
    printf("  newBufferWithBytesNoCopy on PROT_NONE region: %s\n",
           buf ? "accepted" : "rejected");
    printf("  GPU kernel write into MAP_FIXED'd window:     %s\n",
           ok ? "succeeded" : "failed");
    printf("  Pages released after MAP_FIXED'd unmap:       %s\n",
           fresh ? "yes" : "no");
    printf("==============================================\n");
  }
  return 0;
}
