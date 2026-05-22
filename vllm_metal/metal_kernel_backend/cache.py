# SPDX-License-Identifier: Apache-2.0
"""MLX-backed paged KV cache for native Metal paged attention.

Stores per-layer key/value caches as MLX arrays.  KV is written via
MLX-native scatter (Python, donation-friendly) and read by the v2
paged-attention kernels:

- key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]
- value_cache: [num_blocks, block_size, num_kv_heads, head_dim]

Both caches use the same token-contiguous layout where each token's
KV vector is stored contiguously.  This simplifies indexing and is
compatible with variable-length / FlashInfer-style paged attention.

Block allocation is managed externally by the scheduler's KV cache manager.
"""

from __future__ import annotations

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.metal_kernel_backend.turboquant import (
    BLOCK_SIZE,
    FWHT_SUPPORTED_HEAD_DIMS,
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    packed_dim,
)

logger = init_logger(__name__)


class MetalPagedKVCache:
    """Per-layer MLX arrays for native Metal paged attention.

    Supports heterogeneous per-layer shapes: when ``kv_heads_per_layer``
    and ``head_dim_per_layer`` are provided, each cache layer is allocated
    with its own ``(num_kv_heads, head_dim)`` pair.  When omitted, all
    layers share the scalar ``num_kv_heads`` / ``head_dim`` (backward
    compat for MLA, Hybrid, and uniform MHA models).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.float16,
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
        *,
        kv_heads_per_layer: list[int] | None = None,
        head_dim_per_layer: list[int] | None = None,
        sliding_window_per_layer: list[int] | None = None,
        elastic: bool = False,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.turboquant = turboquant
        self.k_quant = k_quant
        self.v_quant = v_quant

        if turboquant:
            if k_quant is None or k_quant not in QUANT_PARAMS:
                available = ", ".join(sorted(QUANT_PARAMS.keys()))
                raise ValueError(
                    f"turboquant requires valid k_quant, got {k_quant!r}. "
                    f"Available: {available}"
                )
            # Default v_quant to "q3_0" (3-bit Lloyd-Max)
            if v_quant is None:
                v_quant = "q3_0"
                self.v_quant = v_quant
            if v_quant not in V_QUANT_PARAMS:
                available = ", ".join(sorted(V_QUANT_PARAMS.keys()))
                raise ValueError(
                    f"turboquant requires valid v_quant, got {v_quant!r}. "
                    f"Available: {available}"
                )
            if head_dim % 32 != 0:
                raise ValueError(
                    f"TurboQuant requires head_dim divisible by 32, got {head_dim}"
                )
            if head_dim not in FWHT_SUPPORTED_HEAD_DIMS:
                supported_head_dims = ", ".join(
                    str(dim) for dim in FWHT_SUPPORTED_HEAD_DIMS
                )
                raise ValueError(
                    "TurboQuant V FWHT only supports "
                    f"head_dim in ({supported_head_dims}), got {head_dim}"
                )

        self.k_size = QUANT_PARAMS[k_quant]["dtype"] if turboquant else None

        # Bit-packed dimensions for sub-8-bit quant types
        if turboquant:
            k_bits = QUANT_PARAMS[k_quant]["bits"]
            v_bits = V_QUANT_PARAMS[v_quant]["bits"]
            self.k_bits = k_bits
            self.v_bits = v_bits
            self.k_packed_dim = packed_dim(head_dim, k_bits)
            self.v_packed_dim = packed_dim(head_dim, v_bits)
        else:
            self.k_bits = 0
            self.v_bits = 0
            self.k_packed_dim = head_dim
            self.v_packed_dim = head_dim

        self.kv_heads_per_layer = kv_heads_per_layer or [num_kv_heads] * num_layers
        self.head_dim_per_layer = head_dim_per_layer or [head_dim] * num_layers
        self.sliding_window_per_layer = sliding_window_per_layer or [-1] * num_layers

        if len(self.kv_heads_per_layer) != num_layers:
            raise ValueError(
                f"kv_heads_per_layer length {len(self.kv_heads_per_layer)} "
                f"!= num_layers {num_layers}"
            )
        if len(self.head_dim_per_layer) != num_layers:
            raise ValueError(
                f"head_dim_per_layer length {len(self.head_dim_per_layer)} "
                f"!= num_layers {num_layers}"
            )
        if len(self.sliding_window_per_layer) != num_layers:
            raise ValueError(
                f"sliding_window_per_layer length "
                f"{len(self.sliding_window_per_layer)} != num_layers {num_layers}"
            )

        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for paged KV cache: {dtype}")

        self.elastic = elastic
        if elastic and turboquant:
            raise ValueError(
                "elastic mode and turboquant are mutually exclusive for now. "
                "Elastic support for TurboQuant is a planned follow-up."
            )

        # Per-layer caches — each layer sized by its own (kv_heads, head_dim)
        self.key_caches: list[mx.array] = []
        self.value_caches: list[mx.array] = []
        self.key_scale_caches: list[mx.array] = []
        self.value_scale_caches: list[mx.array] = []
        self.key_zero_caches: list[mx.array] = []  # asymmetric K zero_point
        # Elastic pools (one per layer for K and V). Empty when elastic=False.
        self._key_pools: list[object] = []
        self._value_pools: list[object] = []
        if not turboquant:
            for i in range(num_layers):
                shape = (
                    num_blocks,
                    block_size,
                    self.kv_heads_per_layer[i],
                    self.head_dim_per_layer[i],
                )
                if elastic:
                    self._allocate_elastic_layer(i, shape, dtype)
                else:
                    self.key_caches.append(mx.zeros(shape, dtype=dtype))
                    self.value_caches.append(mx.zeros(shape, dtype=dtype))
            if not elastic:
                mx.eval(*self.key_caches, *self.value_caches)

            # Log KV cache memory usage
            kv_bytes = (
                num_layers
                * num_blocks
                * block_size
                * num_kv_heads
                * head_dim
                * 2
                * self._dtype_size(dtype)
            )
            logger.info(
                f"KV cache: {kv_bytes / 1e6:.1f} MB "
                f"({num_layers} layers, {num_blocks} blocks, {block_size} tokens/block)"
            )
        else:
            for _ in range(num_layers):
                self.key_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, self.k_packed_dim),
                        dtype=self.k_size,
                    )
                )
                self.value_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, self.v_packed_dim),
                        dtype=mx.uint8,
                    )
                )
                self.key_scale_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, head_dim // BLOCK_SIZE),
                        dtype=mx.float16,
                    )
                )
                self.value_scale_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, head_dim // BLOCK_SIZE),
                        dtype=mx.float16,
                    )
                )
                self.key_zero_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, head_dim // BLOCK_SIZE),
                        dtype=mx.float16,
                    )
                )
            mx.eval(
                *self.key_caches,
                *self.value_caches,
                *self.key_scale_caches,
                *self.value_scale_caches,
                *self.key_zero_caches,
            )

            # Log TurboQuant KV cache memory usage with comparison.
            # Single source of truth: _turboquant_page_size_bytes in cache_policy.
            from vllm_metal.v1.cache_policy import _turboquant_page_size_bytes

            per_block_bytes = _turboquant_page_size_bytes(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                k_quant=k_quant,
                v_quant=v_quant,
            )
            tq_total = num_layers * num_blocks * per_block_bytes
            fp16_equivalent = (
                num_layers * num_blocks * block_size * num_kv_heads * head_dim * 2 * 2
            )  # fp16 K+V
            compression = fp16_equivalent / tq_total if tq_total > 0 else float("inf")
            logger.info(
                f"TurboQuant KV cache (packed): {tq_total / 1e6:.1f} MB "
                f"(K: {self.k_bits}b->{self.k_packed_dim}d, V: {self.v_bits}b->{self.v_packed_dim}d, "
                f"vs {fp16_equivalent / 1e6:.1f} MB fp16, {compression:.2f}x compression)"
            )

    _DTYPE_SIZES = {
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.float32: 4,
        mx.int8: 1,
        mx.uint8: 1,
    }

    @staticmethod
    def _dtype_size(dtype: mx.Dtype) -> int:
        """Return size in bytes for an MLX dtype."""
        size = MetalPagedKVCache._DTYPE_SIZES.get(dtype)
        if size is None:
            raise ValueError(
                f"Unknown dtype {dtype} in _dtype_size. "
                f"Supported: {list(MetalPagedKVCache._DTYPE_SIZES.keys())}"
            )
        return size

    _DTYPE_NAMES: dict = {
        mx.float16: "float16",
        mx.bfloat16: "bfloat16",
        mx.float32: "float32",
        mx.uint8: "uint8",
        mx.int8: "int8",
    }

    def _allocate_elastic_layer(
        self, layer_idx: int, shape: tuple[int, ...], dtype: mx.Dtype
    ) -> None:
        """Allocate one layer's K and V via ElasticKVPool.

        Each pool reserves a lazily-paged mmap region of the full layer size
        and wraps it as an MTL::Buffer directly (newBufferWithBytesNoCopy:,
        bypassing MLX's allocator so the GPU sees our pointer). The
        published array is stable until reclaim() rebuilds the buffer; we
        re-publish into the same Python placeholder so caller references
        remain valid.
        """
        from vllm_metal.metal import get_ops

        dtype_name = self._DTYPE_NAMES.get(dtype)
        if dtype_name is None:
            raise ValueError(
                f"elastic mode does not support dtype {dtype} yet "
                f"(supported: {list(self._DTYPE_NAMES.values())})"
            )

        ops = get_ops()
        Pool = ops.ElasticKVPool
        nbytes = (
            shape[0] * shape[1] * shape[2] * shape[3] * self._dtype_size(dtype)
        )
        key_pool = Pool(nbytes, list(shape), dtype_name)
        value_pool = Pool(nbytes, list(shape), dtype_name)

        # Publish into stable Python placeholders.
        key_arr = mx.array(0)
        value_arr = mx.array(0)
        key_pool.publish_array(key_arr)
        value_pool.publish_array(value_arr)

        self._key_pools.append(key_pool)
        self._value_pools.append(value_pool)
        self.key_caches.append(key_arr)
        self.value_caches.append(value_arr)
        if layer_idx == 0:
            logger.info(
                "Elastic KV pools: per-layer K/V backed by mmap'd "
                "%.1f MB regions (lazy-paged; pages physical only when touched)",
                nbytes / 1e6,
            )

    @staticmethod
    def _merge_runs(block_ids: list[int]) -> list[tuple[int, int]]:
        """Collapse a list of block ids into (start, count) contiguous runs."""
        if not block_ids:
            return []
        ordered = sorted(set(block_ids))
        runs: list[tuple[int, int]] = []
        run_start = ordered[0]
        run_len = 1
        for bid in ordered[1:]:
            if bid == run_start + run_len:
                run_len += 1
            else:
                runs.append((run_start, run_len))
                run_start = bid
                run_len = 1
        runs.append((run_start, run_len))
        return runs

    def mark_blocks_freed(self, block_ids: list[int]) -> None:
        """Queue freed block byte-ranges on each per-layer ``ElasticKVPool``.

        The actual page release happens on the next ``reclaim()`` call —
        that's where the mmap dance + ``MTL::Buffer`` recreate refreshes
        the GPU's IOMMU mapping and lets the released pages drop from
        RSS / phys_footprint.

        No-op when elastic mode is off.
        """
        if not self.elastic or not block_ids:
            return
        runs = self._merge_runs(block_ids)
        if not runs:
            return
        num_blocks = self.num_blocks
        if num_blocks <= 0:
            return
        for layer_idx in range(self.num_layers):
            key_pool = self._key_pools[layer_idx]
            value_pool = self._value_pools[layer_idx]
            key_block_bytes = key_pool.total_bytes // num_blocks
            value_block_bytes = value_pool.total_bytes // num_blocks
            for start, count in runs:
                if key_block_bytes > 0:
                    key_pool.mark_freed(
                        int(start * key_block_bytes),
                        int(count * key_block_bytes),
                    )
                if value_block_bytes > 0:
                    value_pool.mark_freed(
                        int(start * value_block_bytes),
                        int(count * value_block_bytes),
                    )

    def get_stats(self) -> dict:
        """Return aggregated KV cache stats. The ``elastic_*`` fields are
        only present when elastic mode is enabled — they're summed across
        all per-layer K and V pools. ``elastic_total_reclaim_count``
        counts pool-level reclaim ops; divide by ``(2 * num_layers)`` to
        get the number of cache-level ``reclaim()`` calls (each one
        reclaims every pool).
        """
        stats: dict = {
            "elastic": self.elastic,
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "dtype": str(self.dtype),
        }
        if self.elastic and self._key_pools:
            total_bytes = sum(p.total_bytes for p in self._key_pools) + sum(
                p.total_bytes for p in self._value_pools
            )
            pending_bytes = sum(
                p.freed_pending_bytes for p in self._key_pools
            ) + sum(p.freed_pending_bytes for p in self._value_pools)
            released_bytes = sum(
                p.total_released_bytes for p in self._key_pools
            ) + sum(p.total_released_bytes for p in self._value_pools)
            reclaim_count = sum(
                p.total_reclaim_count for p in self._key_pools
            ) + sum(p.total_reclaim_count for p in self._value_pools)
            stats.update(
                {
                    "elastic_total_bytes": total_bytes,
                    "elastic_pending_free_bytes": pending_bytes,
                    "elastic_cumulative_released_bytes": released_bytes,
                    "elastic_total_reclaim_count": reclaim_count,
                }
            )
        return stats

    def reclaim(self) -> int:
        """Release physical pages for all queued freed ranges and refresh
        GPU mappings. Returns total bytes released across all per-layer
        pools. No-op when elastic mode is off.

        Caller MUST synchronize GPU work before calling this — the
        underlying Buffer recreate invalidates any in-flight kernel
        bindings to the old MTLBuffer. ``mx.synchronize()`` is sufficient.
        """
        if not self.elastic:
            return 0
        if not self._key_pools:
            return 0

        # Skip the synchronize + reclaim entirely if nothing is pending.
        pending = sum(
            (p.freed_pending_bytes for p in self._key_pools), 0
        ) + sum(
            (p.freed_pending_bytes for p in self._value_pools), 0
        )
        if pending == 0:
            return 0

        mx.synchronize()
        total_released = 0
        for layer_idx in range(self.num_layers):
            key_pool = self._key_pools[layer_idx]
            value_pool = self._value_pools[layer_idx]
            # reclaim() rebuilds the underlying buffer and re-publishes the
            # array into the existing Python placeholder so wrapper modules
            # reading self.key_caches[i] continue to see the live array.
            total_released += key_pool.reclaim(self.key_caches[layer_idx])
            total_released += value_pool.reclaim(self.value_caches[layer_idx])
        return total_released
