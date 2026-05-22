// SPDX-License-Identifier: Apache-2.0
// See elastic_kv.h for the design.

#include "elastic_kv.h"

#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"
#include "mlx/backend/metal/device.h"

namespace vllm_metal::elastic {

namespace {

size_t system_page_size() {
  static const size_t ps = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return ps;
}

void *reserve_lazy(size_t total_bytes) {
  void *p = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                 MAP_ANON | MAP_PRIVATE, -1, 0);
  if (p == MAP_FAILED) {
    throw std::runtime_error(std::string("ElasticKVPool: mmap reserve(") +
                             std::to_string(total_bytes) +
                             ") failed: " + std::strerror(errno));
  }
  return p;
}

bool remap_fixed(void *addr, size_t len, int prot) {
  void *r = mmap(addr, len, prot, MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);
  return r == addr;
}

bool page_align_inward(size_t offset, size_t length, size_t *out_offset,
                       size_t *out_length) {
  const size_t ps = system_page_size();
  const size_t end = offset + length;
  const size_t aligned_start = (offset + ps - 1) & ~(ps - 1);
  const size_t aligned_end = end & ~(ps - 1);
  if (aligned_end <= aligned_start) {
    return false;
  }
  *out_offset = aligned_start;
  *out_length = aligned_end - aligned_start;
  return true;
}

// Wrap an external host pointer as a real MTL::Buffer via
// newBufferWithBytesNoCopy:. The empty deallocator means the caller owns
// the memory lifetime — Apple won't try to free our mmap'd region when
// the MTL::Buffer is released.
//
// Returns an MTL::Buffer* that MLX's encoder binding code handles
// natively: mlx::core::allocator::Buffer.raw_ptr() does
// `[ptr_ contents]` (objc_msgSend), and set_input_array binds the
// MTL::Buffer to the encoder slot.
//
// We do not go through MLX's MetalAllocator at all. Its
// make_buffer(void*, size_t) overload doesn't wrap the input pointer —
// it allocates a fresh buffer and snapshot-copies the bytes in (verified
// empirically during the kvcached probes — writes wouldn't propagate
// back to the source). Building the MTL::Buffer ourselves keeps the GPU
// writing directly into our mmap region, which is what makes per-page
// release observable on the GPU side after the mmap dance.
MTL::Buffer *wrap_external_as_metal_buffer(void *ptr, size_t size,
                                           MTL::Device *device) {
  MTL::Buffer *buf = device->newBuffer(
      ptr, static_cast<NS::UInteger>(size),
      MTL::ResourceStorageModeShared,
      // Empty deallocator: don't try to free our pointer.
      ^(void *, NS::UInteger){});
  if (buf == nullptr) {
    throw std::runtime_error(
        "ElasticKVPool: newBufferWithBytesNoCopy returned nullptr");
  }
  // Sanity check: verify the newBufferWithBytesNoCopy: overload was
  // actually selected (vs. the copying newBufferWithBytes: overload).
  // contents() must equal our pointer; if it doesn't, the elastic
  // mechanism is broken because GPU writes won't be visible at our base.
  if (buf->contents() != ptr) {
    buf->release();
    throw std::runtime_error(
        "ElasticKVPool: newBufferWithBytesNoCopy: returned a buffer whose "
        "contents() differs from our base pointer — Metal selected a "
        "copying overload. Check metal_cpp version / block syntax.");
  }
  return buf;
}

std::unique_ptr<mlx::core::array> build_array(
    mlx::core::allocator::Buffer buf, const mlx::core::Shape &shape,
    mlx::core::Dtype dtype) {
  // Custom no-op deleter: the pool owns the MTL::Buffer's lifetime and
  // releases it explicitly in the destructor / reclaim path. MLX's
  // default `allocator::free` is wrong for buffers we created outside
  // MLX's allocator.
  auto deleter = [](mlx::core::allocator::Buffer) {};
  return std::make_unique<mlx::core::array>(buf, shape, dtype, deleter);
}

MTL::Device *get_mlx_metal_device() {
  // Use the same MTL::Device MLX uses so command buffers / queues stay
  // consistent. mlx::core::metal::device() is exported from libmlx.
  auto &d = mlx::core::metal::device(mlx::core::Device::gpu);
  return d.mtl_device();
}

}  // namespace

ElasticKVPool::ElasticKVPool(size_t total_bytes, mlx::core::Shape shape,
                             mlx::core::Dtype dtype)
    : total_bytes_(total_bytes),
      shape_(std::move(shape)),
      dtype_(dtype),
      base_(nullptr),
      buf_(nullptr) {
  const size_t ps = system_page_size();
  total_bytes_ = (total_bytes_ + ps - 1) & ~(ps - 1);

  base_ = reserve_lazy(total_bytes_);
  MTL::Buffer *mtlbuf = nullptr;
  try {
    MTL::Device *device = get_mlx_metal_device();
    mtlbuf = wrap_external_as_metal_buffer(base_, total_bytes_, device);
    // Buffer stores the MTL::Buffer pointer — MLX's set_input_array and
    // raw_ptr() (which calls objc_msgSend with "contents") work on this.
    buf_ = mlx::core::allocator::Buffer(static_cast<void *>(mtlbuf));
    arr_ = build_array(buf_, shape_, dtype_);
  } catch (...) {
    if (mtlbuf != nullptr) mtlbuf->release();
    munmap(base_, total_bytes_);
    base_ = nullptr;
    throw;
  }
}

ElasticKVPool::~ElasticKVPool() {
  arr_.reset();
  if (buf_.ptr() != nullptr) {
    static_cast<MTL::Buffer *>(buf_.ptr())->release();
    buf_ = mlx::core::allocator::Buffer(nullptr);
  }
  if (base_ != nullptr) {
    munmap(base_, total_bytes_);
    base_ = nullptr;
  }
}

void ElasticKVPool::mark_freed(size_t offset, size_t length) {
  if (length == 0) return;
  if (offset > total_bytes_ || offset + length > total_bytes_) {
    throw std::out_of_range(
        std::string("ElasticKVPool::mark_freed: range [") +
        std::to_string(offset) + ", " + std::to_string(offset + length) +
        ") exceeds pool size " + std::to_string(total_bytes_));
  }
  pending_.push_back({offset, length});
}

size_t ElasticKVPool::freed_pending_bytes() const {
  size_t total = 0;
  for (const auto &r : pending_) total += r.length;
  return total;
}

size_t ElasticKVPool::reclaim() {
  if (pending_.empty()) return 0;

  // 1. mmap dance — release physical pages for each pending range. With
  // our own newBufferWithBytesNoCopy: buffer, the MTL::Buffer wraps the
  // same VA range, so the GPU sees the fresh mapping after the dance
  // (verified by probe1).
  size_t released = 0;
  for (const auto &r : pending_) {
    size_t aoff = 0, alen = 0;
    if (!page_align_inward(r.offset, r.length, &aoff, &alen)) continue;
    void *target = static_cast<uint8_t *>(base_) + aoff;
    if (!remap_fixed(target, alen, PROT_NONE)) continue;
    if (!remap_fixed(target, alen, PROT_READ | PROT_WRITE)) continue;
    released += alen;
  }
  pending_.clear();

  // 2. Refresh the MTL::Buffer wrap. We tear down and re-create so the
  // GPU IOMMU mapping captures the fresh pages (probe2 found that without
  // this step the GPU may still read stale data from before the unmap).
  // Note this is much cheaper than allocating fresh storage — the
  // newBufferWithBytesNoCopy: path doesn't allocate physical memory,
  // it just registers a VA mapping with the driver.
  arr_.reset();
  static_cast<MTL::Buffer *>(buf_.ptr())->release();
  MTL::Device *device = get_mlx_metal_device();
  MTL::Buffer *mtlbuf = wrap_external_as_metal_buffer(base_, total_bytes_, device);
  buf_ = mlx::core::allocator::Buffer(static_cast<void *>(mtlbuf));
  arr_ = build_array(buf_, shape_, dtype_);

  total_released_bytes_ += released;
  total_reclaim_count_ += 1;
  return released;
}

}  // namespace vllm_metal::elastic
