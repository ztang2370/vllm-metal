# SPDX-License-Identifier: Apache-2.0
"""Tests for the elastic KV cache path.

Covers:
- `MetalPagedKVCache._merge_runs` collapses block-id lists correctly.
- `ElasticKVPool` end-to-end: construct, publish, CPU-write/MLX-read,
  reclaim releases pages and post-reclaim reads return zero.
- `MetalPagedKVCache(elastic=True)` constructs per-layer pools and
  `reclaim()` actually releases bytes; `get_stats()` exposes the
  cumulative counters.
"""

from __future__ import annotations

import platform
import sys

import mlx.core as mx
import pytest

from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Elastic KV requires Apple Silicon (mmap + unified memory).",
)


class TestMergeRuns:
    def test_empty(self) -> None:
        assert MetalPagedKVCache._merge_runs([]) == []

    def test_single_block(self) -> None:
        assert MetalPagedKVCache._merge_runs([7]) == [(7, 1)]

    def test_contiguous(self) -> None:
        assert MetalPagedKVCache._merge_runs([3, 4, 5, 6]) == [(3, 4)]

    def test_disjoint(self) -> None:
        assert MetalPagedKVCache._merge_runs([10, 1, 2, 20, 11]) == [
            (1, 2),
            (10, 2),
            (20, 1),
        ]

    def test_dedup(self) -> None:
        assert MetalPagedKVCache._merge_runs([5, 5, 6, 6, 7]) == [(5, 3)]


class TestElasticKVPool:
    """End-to-end tests for the ElasticKVPool (mmap + newBuffer + reclaim)."""

    def _pool(self, total_bytes: int = 4 * 1024 * 1024):
        from vllm_metal.metal import get_ops

        n_elems = total_bytes // 4  # uint32
        return get_ops().ElasticKVPool(total_bytes, [n_elems], "uint32")

    def test_construct_and_publish(self) -> None:
        pool = self._pool()
        assert pool.total_bytes >= 4 * 1024 * 1024
        assert pool.base_address != 0
        arr = mx.array(0)
        pool.publish_array(arr)
        # The placeholder is now sized for the full pool.
        assert arr.nbytes == pool.total_bytes
        assert arr.dtype == mx.uint32

    def test_read_path_through_mlx(self) -> None:
        import ctypes
        import numpy as np

        pool = self._pool()
        arr = mx.array(0)
        pool.publish_array(arr)
        # CPU writes via the base pointer should be visible to MLX kernels.
        n = 1024
        ptr = ctypes.cast(pool.base_address, ctypes.POINTER(ctypes.c_uint32))
        for i in range(n):
            ptr[i] = 0x10000 + i
        result = arr[:n] + 1
        mx.eval(result)
        assert np.array(result)[0] == 0x10001
        assert np.array(result)[-1] == 0x10000 + n - 1 + 1

    def test_reclaim_releases_and_zeros(self) -> None:
        import ctypes
        import numpy as np

        pool = self._pool()
        arr = mx.array(0)
        pool.publish_array(arr)
        # Touch a region.
        n = 1024
        ptr = ctypes.cast(pool.base_address, ctypes.POINTER(ctypes.c_uint32))
        for i in range(n):
            ptr[i] = 0xCAFE
        # Free + reclaim.
        # 16 KB page on Apple Silicon — free at least one full page so the
        # inward-aligned release isn't empty.
        pool.mark_freed(0, 32 * 1024)
        mx.synchronize()
        released = pool.reclaim(arr)
        assert released > 0
        # After reclaim, the freed region must read zero.
        slice_arr = arr[:n]
        mx.eval(slice_arr)
        assert (np.array(slice_arr) == 0).all()


class TestMetalPagedKVCacheElastic:
    """Tests the elastic branch of MetalPagedKVCache."""

    def test_elastic_construct_and_reclaim(self) -> None:
        cache = MetalPagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            num_blocks=64,
            block_size=16,
            dtype=mx.float16,
            elastic=True,
        )
        assert cache.elastic is True
        assert len(cache._key_pools) == 2
        assert len(cache._value_pools) == 2

        # Reading through the elastic-backed arrays works.
        out = cache.key_caches[0] + 1.0
        mx.eval(out)

        # Free enough blocks to span a full 16 KB page.
        # block_size = 16 * 4 * 64 * 2 bytes = 8192 bytes per block — two blocks per page.
        cache.mark_blocks_freed(list(range(8)))
        released = cache.reclaim()
        assert released > 0

        # The cache still works post-reclaim.
        out2 = cache.value_caches[1] + 2.0
        mx.eval(out2)

    def test_elastic_and_turboquant_mutex(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            MetalPagedKVCache(
                num_layers=1,
                num_kv_heads=4,
                head_dim=64,
                num_blocks=32,
                block_size=16,
                dtype=mx.float16,
                turboquant=True,
                k_quant="q8_0",
                v_quant="q3_0",
                elastic=True,
            )

    def test_non_elastic_reclaim_is_noop(self) -> None:
        """When elastic mode is off, reclaim() returns 0."""
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            num_blocks=16,
            block_size=16,
            dtype=mx.float16,
        )
        assert cache.elastic is False
        assert cache.reclaim() == 0

    def test_get_stats_elastic_mode(self) -> None:
        """Stats report pool size + cumulative reclaim counters."""
        cache = MetalPagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            num_blocks=64,
            block_size=16,
            dtype=mx.float16,
            elastic=True,
        )
        stats = cache.get_stats()
        assert stats["elastic"] is True
        assert stats["num_layers"] == 2
        assert stats["num_blocks"] == 64
        # Pool size = num_layers * 2(K/V) * num_blocks * block_size * heads * dim * dtype
        # Each per-layer pool is rounded up to a system-page multiple, so
        # the aggregate may exceed the naive product slightly.
        per_layer = 64 * 16 * 4 * 64 * 2  # bytes
        assert stats["elastic_total_bytes"] >= 2 * 2 * per_layer
        assert stats["elastic_pending_free_bytes"] == 0
        assert stats["elastic_cumulative_released_bytes"] == 0
        assert stats["elastic_total_reclaim_count"] == 0

        # After a reclaim, counters bump.
        cache.mark_blocks_freed(list(range(8)))
        cache.reclaim()
        stats2 = cache.get_stats()
        assert stats2["elastic_cumulative_released_bytes"] > 0
        # 2 layers * 2 (K+V) = 4 pools, all reclaimed once → count of 4.
        assert stats2["elastic_total_reclaim_count"] == 4

    def test_get_stats_non_elastic(self) -> None:
        """Stats omit elastic-specific fields when elastic mode is off."""
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            num_blocks=16,
            block_size=16,
            dtype=mx.float16,
        )
        stats = cache.get_stats()
        assert stats["elastic"] is False
        assert "elastic_total_bytes" not in stats
        assert "elastic_cumulative_released_bytes" not in stats
