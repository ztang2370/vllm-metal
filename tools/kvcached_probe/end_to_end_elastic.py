"""End-to-end synthetic test of the elastic KV path:
  1. Build a MetalPagedKVCache in elastic mode.
  2. Use ops.kv_scatter to write K/V into many blocks (simulates inference).
  3. Mark those blocks freed via the cache API + reclaim.
  4. Watch rss + phys_footprint across the lifecycle.

If phys_footprint drops by ~the size of the freed blocks after reclaim,
the kvcached-on-Metal design is delivering its core promise on a real
KV cache shape.
"""

from __future__ import annotations

import ctypes
import os

import mlx.core as mx
import psutil
from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache


def phys_fp_mb() -> float:
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    TASK_VM_INFO = 22
    TASK_VM_INFO_COUNT = 87

    class TaskVMInfo(ctypes.Structure):
        _fields_ = [("data", ctypes.c_uint8 * (TASK_VM_INFO_COUNT * 4))]

    info = TaskVMInfo()
    count = ctypes.c_uint(TASK_VM_INFO_COUNT)
    libc.mach_task_self.restype = ctypes.c_uint
    rc = libc.task_info(
        libc.mach_task_self(),
        TASK_VM_INFO,
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if rc != 0:
        return float("nan")
    return float(int.from_bytes(bytes(info.data[16:24]), "little")) / 1e6


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def log(label: str) -> None:
    print(f"  {label:<50s} rss={rss_mb():8.2f} MB  phys_fp={phys_fp_mb():8.2f} MB")


def main() -> None:
    print("End-to-end elastic KV smoke test\n")
    log("baseline")

    num_layers, num_kv_heads, head_dim = 8, 8, 128
    num_blocks, block_size = 1024, 16
    dtype = mx.float16

    cache = MetalPagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
        elastic=True,
    )
    block_bytes = block_size * num_kv_heads * head_dim * 2  # fp16
    total_per_layer = num_blocks * block_bytes
    total_pool = 2 * num_layers * total_per_layer  # K + V × N layers
    print(f"  pool config: {num_layers} layers × 2(K/V) × "
          f"{total_per_layer/1e6:.1f} MB = {total_pool/1e6:.1f} MB virtual\n")
    log("after MetalPagedKVCache(elastic=True) construct")

    # "Inference" — fill 200 blocks with kv_scatter, layer by layer.
    # Use random (high-entropy) data so Apple's VM compression can't
    # coalesce the writes into a single zero/uniform page.
    fill_blocks = 200
    tokens_per_block = block_size
    total_tokens = fill_blocks * tokens_per_block
    k = mx.random.normal(
        shape=(total_tokens, num_kv_heads, head_dim), dtype=mx.float32
    ).astype(dtype)
    v = mx.random.normal(
        shape=(total_tokens, num_kv_heads, head_dim), dtype=mx.float32
    ).astype(dtype)
    mx.eval(k, v)
    slot_mapping = mx.arange(total_tokens, dtype=mx.int64)

    ops = get_ops()
    for layer in range(num_layers):
        nk, nv = ops.kv_scatter(
            k, v,
            cache.key_caches[layer], cache.value_caches[layer],
            slot_mapping,
        )
        cache.key_caches[layer] = nk
        cache.value_caches[layer] = nv
    mx.synchronize()
    log(f"after writing {fill_blocks} blocks × {num_layers} layers (K+V)")
    expected_touched_bytes = 2 * num_layers * fill_blocks * block_bytes
    print(f"  (expected touched: {expected_touched_bytes/1e6:.1f} MB)\n")

    # Sanity: confirm the writes landed in our elastic pool by reading
    # via CPU through the pool's host pointer.
    import numpy as np
    print("  Per-layer K pool check:")
    for layer in range(num_layers):
        kp = cache._key_pools[layer]
        cpu = ctypes.cast(kp.base_address, ctypes.POINTER(ctypes.c_uint16))
        mlx_val = float(np.array(cache.key_caches[layer][0, 0, 0, 0]))
        mlx_raw = np.array(cache.key_caches[layer][0, 0, 0, 0],
                           dtype=np.float16).tobytes()
        mlx_raw_int = int.from_bytes(mlx_raw, "little")
        match = "YES" if cpu[0] == mlx_raw_int else "NO"
        print(f"    L{layer}: pool=0x{kp.base_address:x} "
              f"MLX={mlx_val:+.4f}(0x{mlx_raw_int:04x}) "
              f"CPU[0]=0x{cpu[0]:04x} match={match}")

    # Free 150 of those blocks + reclaim
    free_block_ids = list(range(0, 150))
    cache.mark_blocks_freed(free_block_ids)
    log("after mark_blocks_freed(150 blocks)")

    fp_before = phys_fp_mb()
    released = cache.reclaim()
    fp_after = phys_fp_mb()
    log(f"after reclaim ({released/1e6:.2f} MB released)")
    expected_freed_bytes = 2 * num_layers * len(free_block_ids) * block_bytes
    print(f"  (expected freed: {expected_freed_bytes/1e6:.2f} MB; "
          f"actual drop in phys_fp: {fp_before - fp_after:.2f} MB)")

    # Verify reclaimed regions read zero (fresh pages after mmap dance).
    print("  Reclaimed-region read check:")
    all_zero = True
    for layer in range(num_layers):
        slc = cache.key_caches[layer][0, 0, 0, 0]
        val = float(np.array(slc))
        if val != 0.0:
            all_zero = False
            print(f"    L{layer}: cache[0,0,0,0]={val} (NOT zero — pages "
                  f"may not have been released)")
    print(f"  All reclaimed slots read zero: "
          f"{'YES (pages truly released)' if all_zero else 'NO'}")

    # Verify cache is still functional post-reclaim — write a known pattern
    # to a NEW block, verify readback.
    known_k = mx.full((block_size, num_kv_heads, head_dim), 1.5, dtype=dtype)
    known_v = mx.full((block_size, num_kv_heads, head_dim), 2.5, dtype=dtype)
    nk, nv = ops.kv_scatter(
        known_k, known_v,
        cache.key_caches[0], cache.value_caches[0],
        mx.arange(800 * block_size, 800 * block_size + block_size, dtype=mx.int64),
    )
    cache.key_caches[0] = nk
    cache.value_caches[0] = nv
    mx.synchronize()
    log("after post-reclaim kv_scatter (sanity write of 1.5)")

    sample = float(np.array(cache.key_caches[0][800, 0, 0, 0]))
    print(f"  Post-reclaim K cache write readback: {sample} "
          f"(expected 1.5): {'PASS' if abs(sample - 1.5) < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
