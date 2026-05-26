# SPDX-License-Identifier: Apache-2.0
"""Bridge from vLLM's stock ``BlockPool`` to the Metal elastic-KV reclaim path.

vLLM's scheduler frees KV blocks by calling ``BlockPool.free_blocks``. With
``VLLM_METAL_ELASTIC_KV=1`` we want those releases to flow into the per-layer
``ElasticKVPool`` queues so ``MetalPagedKVCache.reclaim()`` later drops the
physical pages back to the OS.

In prefix-cache mode the criterion "finished by its owner" is not the same as
"physically reclaimable": a block whose ``block_hash`` is set is being retained
in ``cached_block_hash_to_block`` for cross-request reuse, so its pages must
stay backed. The hook therefore forwards only blocks that ended ``free_blocks``
with ``ref_cnt == 0`` *and* ``block_hash is None`` — i.e. truly released by both
the request and the prefix cache.

A workload with high prefix diversity can fill the cache to the size of the
pool, at which point elastic reclaim never fires. To bound that, the hook
maintains an LRU of cached-and-freed blocks; once it exceeds ``max_cached_blocks``
the oldest entries are evicted (their hashes cleared via
``BlockPool._maybe_evict_cached_block``) and their pages handed to the elastic
backend. The LRU uses lazy cleanup: stale entries (re-acquired by ``touch`` or
already evicted by ``get_new_blocks``) are skipped at eviction time, so the
hook does not need to patch those code paths.

Blocks evicted from the prefix cache during a subsequent ``get_new_blocks``
allocation are intentionally *not* forwarded: they are about to be overwritten
by the new request, so reclaiming would only cost a synchronize + buffer
rebuild before the OS hands back the same physical pages.

vllm-metal uses ``distributed_executor_backend="uni"``, so the scheduler's
``BlockPool`` and the worker's ``MetalPagedKVCache`` live in one process and
this in-process listener registration is sufficient.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_metal.paged_attention_backend.protocol import PagedAttentionBackend

logger = init_logger(__name__)


_active_listener: PagedAttentionBackend | None = None
# Negative = unbounded. Zero = evict every cached-and-freed block immediately.
# Positive = LRU cap in block count.
_max_cached_blocks: int = -1
# block_id → KVCacheBlock. Insertion order = LRU order (oldest first).
_cached_lru: OrderedDict[int, Any] = OrderedDict()
_hook_installed: bool = False


def register_elastic_listener(
    backend: PagedAttentionBackend,
    *,
    max_cached_blocks: int = -1,
) -> None:
    """Route ``BlockPool.free_blocks`` events to ``backend``.

    ``max_cached_blocks`` caps the LRU of cached-and-freed blocks (those
    parked in vLLM's prefix cache awaiting reuse). Negative = unbounded.

    Idempotent; replaces any previously registered listener and resets the
    LRU. Also re-attempts hook installation — at platform-plugin registration
    time ``vllm.v1.core.block_pool`` may not yet be importable (a circular
    import via ``vllm.config``); by the time the worker registers a listener
    that risk is gone, so this is the right place to make sure the hook is
    actually in place.
    """
    global _active_listener, _max_cached_blocks
    install_block_pool_hook()
    _active_listener = backend
    _max_cached_blocks = max_cached_blocks
    _cached_lru.clear()


def unregister_elastic_listener() -> None:
    """Clear the active listener and reset cap state (called on teardown)."""
    global _active_listener, _max_cached_blocks
    _active_listener = None
    _max_cached_blocks = -1
    _cached_lru.clear()


def install_block_pool_hook() -> None:
    """Patch ``vllm.v1.core.block_pool.BlockPool.free_blocks`` once.

    Safe to call before vLLM is imported — returns silently if the module is
    unavailable. Safe to call multiple times — second and later calls are
    no-ops.
    """
    global _hook_installed
    if _hook_installed:
        return

    try:
        from vllm.v1.core.block_pool import BlockPool
    except ImportError:
        # vLLM not importable in this context (e.g. tooling that loads the
        # plugin without the full engine stack). Nothing to patch.
        return

    if getattr(BlockPool, "_vllm_metal_elastic_hook", False):
        _hook_installed = True
        return

    original_free_blocks = BlockPool.free_blocks

    def _patched_free_blocks(self, ordered_blocks):
        # Materialize once so we can both pass through to upstream and inspect.
        blocks_list = list(ordered_blocks)
        original_free_blocks(self, blocks_list)

        listener = _active_listener
        if listener is None:
            return

        # Bucket the just-freed batch.
        #   - uncached_ids → blocks ready to reclaim now (no prefix-cache claim).
        #   - cached_blocks → blocks parked in the prefix cache; enter LRU,
        #     may be evicted below if we're over the cap.
        uncached_ids: list[int] = []
        cached_blocks: list[Any] = []
        for block in blocks_list:
            if block.ref_cnt != 0 or getattr(block, "is_null", False):
                continue
            if block.block_hash is None:
                uncached_ids.append(block.block_id)
            else:
                cached_blocks.append(block)

        # Refresh LRU position for the cached batch. Re-freed blocks (touched
        # then freed again) move to the end so they're the most-recently-used.
        for block in cached_blocks:
            _cached_lru.pop(block.block_id, None)
            _cached_lru[block.block_id] = block

        evicted_ids = _enforce_cap(self)

        freed_ids = uncached_ids + evicted_ids
        if freed_ids:
            listener.mark_blocks_freed(freed_ids)

    BlockPool.free_blocks = _patched_free_blocks
    BlockPool._vllm_metal_elastic_hook = True
    _hook_installed = True
    logger.debug("Installed vllm-metal elastic-KV hook on BlockPool.free_blocks")


def _enforce_cap(block_pool: Any) -> list[int]:
    """Pop oldest LRU entries until ``len(_cached_lru) <= _max_cached_blocks``.

    Each pop clears the block's prefix-cache hash via the BlockPool's own
    ``_maybe_evict_cached_block`` (so future ``get_cached_block`` calls miss)
    and reports the block id back to the caller for forwarding to the elastic
    backend.

    Stale LRU entries — blocks that were re-acquired by ``touch`` (ref_cnt > 0)
    or already un-cached by ``get_new_blocks`` (block_hash is None) — are
    discarded silently; this is the contract that lets us avoid patching
    ``touch`` and ``_maybe_evict_cached_block``.
    """
    if _max_cached_blocks < 0:
        return []

    evicted: list[int] = []
    # Cap loop iterations at the current LRU size so a pathological burst of
    # stale entries can't spin forever. Each iteration either pops a real
    # victim or discards a stale entry — both shrink the LRU monotonically.
    max_iters = len(_cached_lru)
    while len(_cached_lru) > _max_cached_blocks and max_iters > 0:
        max_iters -= 1
        bid, blk = _cached_lru.popitem(last=False)
        # Lazy cleanup: skip blocks that no longer qualify as cached-evictable.
        if blk.ref_cnt != 0:
            continue
        if blk.block_hash is None:
            continue
        # Clears block.block_hash, removes from cached_block_hash_to_block,
        # and (when enabled) emits a BlockRemoved KV-cache event.
        # ``block_pool._maybe_evict_cached_block`` does not touch the
        # free_block_queue, so the block remains allocatable by vLLM.
        block_pool._maybe_evict_cached_block(blk)
        evicted.append(bid)
    return evicted
