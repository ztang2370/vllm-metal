// SPDX-License-Identifier: Apache-2.0
//
// ElasticKVPool — kvcached-style elastic memory backing for one MLX array.
//
// Phase 1 probes confirmed that on Apple Silicon we can:
//   1. mmap() a large VA range with PROT_READ|WRITE (lazily backed).
//   2. Wrap it as an MTL::Buffer via MTL::Device::newBuffer(ptr, size, ...)
//      (the no-copy overload, with an empty deallocator). MLX's own
//      MetalAllocator path was rejected — its make_buffer(void*, size_t)
//      snapshot-copies the input pointer into a fresh allocation, so GPU
//      writes wouldn't reach our mmap region.
//   3. Stash that MTL::Buffer in an mlx::core::allocator::Buffer slot and
//      build an mlx::core::array view over it (with a no-op deleter, since
//      MLX's default deleter routes back through its allocator).
//   4. Release physical pages with mmap(MAP_FIXED, PROT_NONE) — phys_fp drops.
//   5. Refresh the GPU's IOMMU mapping by releasing the old MTL::Buffer and
//      calling newBuffer() again over the same base pointer, then rebuilding
//      the mlx::core::array.
//
// One pool owns one mmap'd region + one Buffer + one array (per "slot" — in
// practice one per K-cache layer and one per V-cache layer). Free events
// accumulate as byte ranges; reclaim() does the mmap dance + buffer
// re-creation in one pass and returns a fresh array the caller must rebind.
//
// Reclaim is sync-blocking: all in-flight ops on the old array must complete
// (mx.synchronize) before release(). Caller is responsible for sequencing.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mlx/array.h"
#include "mlx/allocator.h"

namespace vllm_metal::elastic {

struct FreeRange {
  size_t offset;
  size_t length;
};

class ElasticKVPool {
 public:
  // Allocate a pool of `total_bytes`. The bytes are reservation only —
  // physical pages aren't committed until written. `shape` and `dtype`
  // describe the mlx::core::array view returned by array().
  //
  // Throws std::runtime_error on mmap or MTL::Device::newBuffer failure.
  ElasticKVPool(size_t total_bytes,
                mlx::core::Shape shape,
                mlx::core::Dtype dtype);

  ~ElasticKVPool();

  ElasticKVPool(const ElasticKVPool&) = delete;
  ElasticKVPool& operator=(const ElasticKVPool&) = delete;

  // The current mlx::core::array view of the pool. Identity is stable until
  // reclaim() is called.
  const mlx::core::array& array() const { return *arr_; }

  // Mark a byte range as freed. The range must be entirely within the pool.
  // Does NOT release pages yet — call reclaim() to actually do the
  // mmap-dance + buffer refresh.
  //
  // Accepts unaligned offset/length; the underlying mmap will inward-align
  // to a multiple of the system page size, so partial pages remain in use.
  void mark_freed(size_t offset, size_t length);

  // Apply all accumulated mark_freed() calls:
  //   1. For each pending range: mmap(MAP_FIXED, PROT_NONE) then
  //      mmap(MAP_FIXED, PROT_READ|WRITE). Releases physical pages.
  //   2. release(old MTL::Buffer) + MTL::Device::newBuffer(base, total_bytes,
  //      ...). Refreshes the GPU's IOMMU mapping.
  //   3. Reconstruct the mlx::core::array over the new Buffer.
  //
  // The caller MUST ensure all GPU work on the old array has completed
  // before calling this (typically via mx.synchronize()). After return,
  // any held references to the previous array() result are stale; callers
  // must rebind to the new array() result.
  //
  // Returns the number of bytes freed in this pass (sum of inward-aligned
  // range sizes). Returns 0 if there were no pending frees (no-op).
  size_t reclaim();

  // Stats.
  size_t total_bytes() const { return total_bytes_; }
  size_t freed_pending_bytes() const;
  void* base_ptr() const { return base_; }

  // Cumulative bytes released across all reclaim() calls (page-aligned).
  // Useful for observability: divided by total_bytes() it gives the
  // total work the elastic mechanism has done since process start.
  size_t total_released_bytes() const { return total_released_bytes_; }

  // Number of reclaim() calls that did real work (had pending ranges).
  size_t total_reclaim_count() const { return total_reclaim_count_; }

 private:
  size_t total_bytes_;
  mlx::core::Shape shape_;
  mlx::core::Dtype dtype_;

  void* base_;
  mlx::core::allocator::Buffer buf_;
  // arr_ is heap-allocated so we can swap it during reclaim().
  std::unique_ptr<mlx::core::array> arr_;

  std::vector<FreeRange> pending_;

  // Cumulative metrics.
  size_t total_released_bytes_{0};
  size_t total_reclaim_count_{0};
};

}  // namespace vllm_metal::elastic
