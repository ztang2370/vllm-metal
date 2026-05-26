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


class _CapturingBackend:
    """Minimal ``PagedAttentionBackend`` stand-in for hook tests."""

    def __init__(self) -> None:
        self.freed: list[int] = []

    def mark_blocks_freed(self, ids: list[int]) -> None:
        self.freed.extend(ids)

    def reclaim(self) -> int:
        return 0


class TestBlockPoolElasticHook:
    """Verify the BlockPool free-blocks → elastic-listener bridge.

    Covers the prefix-cache split: only ``ref_cnt == 0`` blocks whose
    ``block_hash is None`` should reach the listener; cached blocks
    (block_hash set) and still-ref'd blocks must be skipped.
    """

    @staticmethod
    def _block_pool(num_blocks: int = 8):
        # Importing here keeps module import side-effects off test
        # collection (the patch is installed in compat.apply_compat_patches).
        import vllm_metal  # noqa: F401  (triggers plugin register)

        vllm_metal._register()
        from vllm.v1.core.block_pool import BlockPool

        return BlockPool(
            num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16
        )

    def test_hook_forwards_only_truly_released(self) -> None:
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool()
        backend = _CapturingBackend()
        h.register_elastic_listener(backend)
        try:
            uncached = bp.blocks[1]
            uncached.ref_cnt = 1
            uncached.block_hash = None

            cached = bp.blocks[2]
            cached.ref_cnt = 1
            cached.block_hash = "stand-in-hash"

            still_ref = bp.blocks[3]
            still_ref.ref_cnt = 2  # decremented to 1 by free_blocks
            still_ref.block_hash = None

            bp.free_blocks([uncached, cached, still_ref])
        finally:
            h.unregister_elastic_listener()

        assert backend.freed == [1], (
            "Only the uncached block whose ref_cnt drops to 0 should reach the "
            "elastic backend; cached blocks (prefix cache) and still-ref'd "
            "blocks must be skipped."
        )

    def test_hook_no_op_without_listener(self) -> None:
        from vllm_metal.v1 import elastic_kv_hooks as h

        h.unregister_elastic_listener()  # ensure clean state
        bp = self._block_pool()
        blk = bp.blocks[1]
        blk.ref_cnt = 1
        blk.block_hash = None
        # Should not raise / not require a listener at all.
        bp.free_blocks([blk])
        assert h._active_listener is None

    def test_install_is_idempotent(self) -> None:
        from vllm.v1.core.block_pool import BlockPool

        from vllm_metal.v1 import elastic_kv_hooks as h

        h.install_block_pool_hook()
        first = BlockPool.free_blocks
        h.install_block_pool_hook()
        assert BlockPool.free_blocks is first

    def test_register_listener_installs_hook_lazily(self) -> None:
        """``register_elastic_listener`` must guarantee the hook is in place
        even if the install attempt at platform-plugin registration was
        skipped — ``vllm.v1.core.block_pool`` transitively imports
        ``vllm.config``, which is mid-initialization at that point, so the
        early install raises ``ImportError`` and bails. The worker-side
        register call is the safety net.
        """
        from vllm.v1.core.block_pool import BlockPool

        from vllm_metal.v1 import elastic_kv_hooks as h

        # Simulate the "early install bailed on ImportError" state by clearing
        # the cached success flag. The class-level _vllm_metal_elastic_hook
        # attribute (if any prior test installed) lets the install short-
        # circuit on the second-best path.
        h._hook_installed = False  # type: ignore[attr-defined]

        backend = _CapturingBackend()
        try:
            h.register_elastic_listener(backend)
            assert getattr(BlockPool, "_vllm_metal_elastic_hook", False), (
                "register_elastic_listener must ensure the hook is installed"
            )
            assert h._hook_installed  # type: ignore[attr-defined]
        finally:
            h.unregister_elastic_listener()


class TestBlockPoolElasticHookLRU:
    """Verify the prefix-cache LRU cap that bounds elastic-mode memory growth.

    Without a cap, cached prefix blocks retain pages until vLLM evicts them on
    allocation — a high-diversity workload can fill the pool and stall reclaim.
    These tests cover insertion, LRU ordering on re-free, eviction at the cap,
    and the lazy-cleanup contract that lets the hook avoid patching ``touch``
    or ``_maybe_evict_cached_block``.
    """

    @staticmethod
    def _block_pool(num_blocks: int = 16):
        import vllm_metal  # noqa: F401

        vllm_metal._register()
        from vllm.v1.core.block_pool import BlockPool

        return BlockPool(
            num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16
        )

    @staticmethod
    def _make_cached(bp, block_id: int, key) -> None:
        """Make ``block_id`` look like a freshly-cached block (ref_cnt=1, hashed,
        present in the pool's hash map). Free will drop ref to 0 and route it
        through the LRU.
        """
        blk = bp.blocks[block_id]
        blk.ref_cnt = 1
        # Bypass the property setter's "block_hash is None" assertion so
        # successive tests can re-use the same KVCacheBlock object.
        blk._block_hash = key  # type: ignore[attr-defined]
        bp.cached_block_hash_to_block.insert(key, blk)

    def test_cap_evicts_oldest_when_exceeded(self) -> None:
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool(num_blocks=16)
        backend = _CapturingBackend()
        h.register_elastic_listener(backend, max_cached_blocks=2)
        try:
            for bid in (1, 2, 3):
                self._make_cached(bp, bid, f"hash-{bid}")
                bp.free_blocks([bp.blocks[bid]])

            # After freeing 3 cached blocks with cap=2, the oldest (block 1)
            # must have been evicted: forwarded to backend AND removed from
            # vLLM's prefix-cache map.
            assert backend.freed == [1], backend.freed
            assert bp.blocks[1].block_hash is None
            # Surviving entries stay cached and in LRU.
            assert bp.blocks[2].block_hash == "hash-2"
            assert bp.blocks[3].block_hash == "hash-3"
            assert list(h._cached_lru.keys()) == [2, 3]
        finally:
            h.unregister_elastic_listener()

    def test_refree_moves_to_lru_tail(self) -> None:
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool(num_blocks=16)
        backend = _CapturingBackend()
        h.register_elastic_listener(backend, max_cached_blocks=2)
        try:
            # Free three cached blocks at cap=2 → 1 evicted, [2, 3] survive.
            for bid in (1, 2, 3):
                self._make_cached(bp, bid, f"hash-{bid}")
                bp.free_blocks([bp.blocks[bid]])
            backend.freed.clear()

            # Simulate a cache hit on block 2: ref-up via touch, then re-free.
            # Real path: scheduler.touch → block.ref_cnt += 1; later free_blocks
            # drops it back to 0. The hook should move block 2 to the LRU tail.
            bp.touch([bp.blocks[2]])
            bp.free_blocks([bp.blocks[2]])

            # Now add a 4th cached block. With cap=2 the oldest should be 3,
            # NOT 2 — block 2 was just re-freed so it's the freshest.
            self._make_cached(bp, 4, "hash-4")
            bp.free_blocks([bp.blocks[4]])

            assert backend.freed == [3], backend.freed
            assert bp.blocks[3].block_hash is None
            assert list(h._cached_lru.keys()) == [2, 4]
        finally:
            h.unregister_elastic_listener()

    def test_unbounded_cap_never_evicts_cached(self) -> None:
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool(num_blocks=16)
        backend = _CapturingBackend()
        h.register_elastic_listener(backend, max_cached_blocks=-1)
        try:
            for bid in (1, 2, 3, 4, 5):
                self._make_cached(bp, bid, f"hash-{bid}")
                bp.free_blocks([bp.blocks[bid]])

            # Default (unbounded): cached blocks stay backed indefinitely.
            assert backend.freed == []
            for bid in (1, 2, 3, 4, 5):
                assert bp.blocks[bid].block_hash == f"hash-{bid}"
        finally:
            h.unregister_elastic_listener()

    def test_cap_zero_evicts_every_cached_block(self) -> None:
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool(num_blocks=16)
        backend = _CapturingBackend()
        h.register_elastic_listener(backend, max_cached_blocks=0)
        try:
            for bid in (1, 2):
                self._make_cached(bp, bid, f"hash-{bid}")
                bp.free_blocks([bp.blocks[bid]])

            # cap=0 means "no retention" — every cached freed block evicts.
            assert backend.freed == [1, 2]
            for bid in (1, 2):
                assert bp.blocks[bid].block_hash is None
            assert len(h._cached_lru) == 0
        finally:
            h.unregister_elastic_listener()

    def test_stale_lru_entry_after_touch_is_skipped(self) -> None:
        """A block re-acquired by ``touch`` (ref_cnt > 0) but still sitting in
        the LRU must be popped *without* eviction when the cap fires. Each
        stale pop shrinks the LRU by one — often the cap is satisfied before
        any real victim is needed."""
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool(num_blocks=16)
        backend = _CapturingBackend()
        h.register_elastic_listener(backend, max_cached_blocks=2)
        try:
            self._make_cached(bp, 1, "hash-1")
            bp.free_blocks([bp.blocks[1]])
            self._make_cached(bp, 2, "hash-2")
            bp.free_blocks([bp.blocks[2]])
            assert list(h._cached_lru.keys()) == [1, 2]
            assert backend.freed == []

            # Touch(1): ref_cnt 0 → 1. Block 1 is now active; the LRU entry is
            # stale but still occupies the head slot.
            bp.touch([bp.blocks[1]])

            # Free a third cached block. LRU grows to [1, 2, 3]; cap=2 fires.
            # Walker pops 1 (stale, ref_cnt=1 → skip), then len=2 == cap → done.
            # No real eviction happens; backend hears nothing.
            self._make_cached(bp, 3, "hash-3")
            bp.free_blocks([bp.blocks[3]])

            assert backend.freed == [], backend.freed
            assert bp.blocks[1].block_hash == "hash-1"  # untouched
            assert list(h._cached_lru.keys()) == [2, 3]
        finally:
            h.unregister_elastic_listener()

    def test_stale_lru_entry_after_external_evict_is_skipped(self) -> None:
        """If a block had its hash cleared by vLLM's own allocation path
        (``get_new_blocks`` → ``_maybe_evict_cached_block``) between our hook
        adding it to the LRU and the next cap check, the cap walker must
        discard the stale entry rather than re-evicting an already-evicted
        block."""
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool(num_blocks=16)
        backend = _CapturingBackend()
        h.register_elastic_listener(backend, max_cached_blocks=2)
        try:
            self._make_cached(bp, 1, "hash-1")
            bp.free_blocks([bp.blocks[1]])
            self._make_cached(bp, 2, "hash-2")
            bp.free_blocks([bp.blocks[2]])
            backend.freed.clear()

            # vLLM externally clears block 1's hash (simulating get_new_blocks
            # popping it for a new request). Our LRU still references it.
            bp._maybe_evict_cached_block(bp.blocks[1])
            assert bp.blocks[1].block_hash is None

            # Free block 3. Walker pops 1 (stale, hash=None → skip); cap met.
            self._make_cached(bp, 3, "hash-3")
            bp.free_blocks([bp.blocks[3]])

            assert backend.freed == [], backend.freed
            assert list(h._cached_lru.keys()) == [2, 3]
        finally:
            h.unregister_elastic_listener()

    def test_stale_then_real_victim(self) -> None:
        """Once a stale entry is sloughed off the LRU head, subsequent cap
        enforcement must still find and evict real victims correctly."""
        from vllm_metal.v1 import elastic_kv_hooks as h

        bp = self._block_pool(num_blocks=16)
        backend = _CapturingBackend()
        h.register_elastic_listener(backend, max_cached_blocks=2)
        try:
            self._make_cached(bp, 1, "hash-1")
            bp.free_blocks([bp.blocks[1]])
            self._make_cached(bp, 2, "hash-2")
            bp.free_blocks([bp.blocks[2]])

            # Make block 1 stale, then add block 3 — stale block 1 gets
            # popped, no eviction; LRU = [2, 3].
            bp.touch([bp.blocks[1]])
            self._make_cached(bp, 3, "hash-3")
            bp.free_blocks([bp.blocks[3]])
            assert backend.freed == []
            assert list(h._cached_lru.keys()) == [2, 3]

            # Add block 4 — no stale entries now, oldest (block 2) is the
            # real victim.
            self._make_cached(bp, 4, "hash-4")
            bp.free_blocks([bp.blocks[4]])

            assert backend.freed == [2], backend.freed
            assert bp.blocks[2].block_hash is None
            assert list(h._cached_lru.keys()) == [3, 4]
        finally:
            h.unregister_elastic_listener()


class TestElasticKvMaxCachedBlocksConfig:
    """Verify the env→cap_blocks parser used by setup_paged_attention."""

    def test_unset_yields_unbounded(self, monkeypatch) -> None:
        monkeypatch.delenv("VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION", raising=False)
        from vllm_metal.v1.cache_policy import WorkerCachePlanner

        assert WorkerCachePlanner._elastic_kv_max_cached_blocks(1000) == -1

    def test_fraction_converts_to_block_count(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION", "0.25")
        from vllm_metal.v1.cache_policy import WorkerCachePlanner

        assert WorkerCachePlanner._elastic_kv_max_cached_blocks(1000) == 250

    def test_out_of_range_falls_back_to_unbounded(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION", "1.5")
        from vllm_metal.v1.cache_policy import WorkerCachePlanner

        assert WorkerCachePlanner._elastic_kv_max_cached_blocks(1000) == -1

    def test_unparseable_falls_back_to_unbounded(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION", "half")
        from vllm_metal.v1.cache_policy import WorkerCachePlanner

        assert WorkerCachePlanner._elastic_kv_max_cached_blocks(1000) == -1
