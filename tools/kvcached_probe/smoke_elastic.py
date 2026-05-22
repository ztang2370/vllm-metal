"""Smoke test for ElasticKVPool end-to-end through Python.

Verifies:
  1. Pool can be constructed via the new nanobind binding.
  2. publish_array() lands a live mlx.core.array bound to our pool.
  3. CPU writes through the buffer pointer are visible to MLX kernels.
  4. mark_freed + reclaim drops phys_fp and gives us a fresh array that
     reads zero after the reclaim.
  5. Recreated array stays usable for fresh CPU writes + MLX reads.
"""

from __future__ import annotations

import ctypes
import os
import resource

import mlx.core as mx
import psutil
from vllm_metal.metal import get_ops

ops = get_ops()
Pool = ops.ElasticKVPool


def phys_fp() -> float:
    # macOS task_vm_info.phys_footprint is what Activity Monitor / `sample`
    # show. psutil's memory_info exposes it as rss but rss != phys_footprint
    # on macOS for IOKit-mapped memory. Use task_info via ctypes.
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    TASK_VM_INFO = 22
    TASK_VM_INFO_COUNT = 87  # depends on macOS version; oversize is safe

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
    # phys_footprint is at offset 16 in task_vm_info_data_t
    return float(int.from_bytes(bytes(info.data[16:24]), "little")) / 1e6


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def log(label: str) -> None:
    print(f"  {label:<48s} rss={rss_mb():7.2f} MB  phys_fp={phys_fp():7.2f} MB")


def main() -> None:
    print("ElasticKVPool smoke test\n")

    log("baseline")

    # Allocate a 256 MB pool, viewed as uint32.
    total = 256 * 1024 * 1024  # 256 MB
    n_elems = total // 4
    pool = Pool(total, [n_elems], "uint32")
    log(f"after Pool(256 MB) construct")

    # Bind into an mx.array placeholder via publish_array.
    arr = mx.array(0)
    pool.publish_array(arr)
    log("after publish_array")
    print(f"  base_address: 0x{pool.base_address:x}, total_bytes: "
          f"{pool.total_bytes/1e6:.2f} MB")
    print(f"  array.shape: {arr.shape}, dtype: {arr.dtype}, "
          f"nbytes: {arr.nbytes/1e6:.2f} MB")

    # Write a pattern via CPU using ctypes (unified memory is host-readable).
    n_write = 4 * 1024 * 1024  # write 4 MB of uint32s = 1M cells
    base_ptr = ctypes.cast(pool.base_address, ctypes.POINTER(ctypes.c_uint32))
    PATTERN = 0xABCD0000
    for i in range(n_write // 4):
        base_ptr[i] = PATTERN + i
    log(f"after CPU wrote {n_write/1e6:.1f} MB pattern")

    # Read through MLX: slice the first 1M cells, add 1, eval.
    slice_view = arr[: n_write // 4]
    out = slice_view + 1
    mx.eval(out)
    log("after MLX read+add kernel")

    # Verify
    import numpy as np
    out_np = np.array(out)
    expected_first = PATTERN + 0 + 1
    expected_last = PATTERN + (n_write // 4 - 1) + 1
    ok = out_np[0] == expected_first and out_np[-1] == expected_last
    print(f"  MLX read of CPU writes: {'PASS' if ok else 'FAIL'} "
          f"(first={out_np[0]:#x}, expected={expected_first:#x})")
    if not ok:
        return

    # Free the range we wrote, reclaim.
    print()
    pool.mark_freed(0, n_write)
    log("after mark_freed(0, 4MB)")
    print(f"  pending: {pool.freed_pending_bytes/1e6:.2f} MB")

    mx.synchronize()
    new_arr = mx.array(0)
    freed = pool.reclaim(new_arr)
    print(f"  reclaim released {freed/1e6:.2f} MB (page-aligned)")
    log("after reclaim")

    # The new array should read zero in the freed region (fresh pages).
    slice2 = new_arr[: n_write // 4]
    mx.eval(slice2)
    s2_np = np.array(slice2)
    all_zero = bool((s2_np == 0).all())
    print(f"  Reclaimed region reads zero: {'PASS' if all_zero else 'FAIL'} "
          f"(first 4: {s2_np[:4].tolist()})")

    # Write fresh data via CPU into reclaimed region; MLX should see it.
    NEW_PATTERN = 0xCAFE0000
    base_ptr2 = ctypes.cast(pool.base_address, ctypes.POINTER(ctypes.c_uint32))
    for i in range(n_write // 4):
        base_ptr2[i] = NEW_PATTERN + i
    log("after CPU wrote new pattern into reclaimed range")

    slice3 = new_arr[: n_write // 4]
    out3 = slice3 + 2
    mx.eval(out3)
    log("after MLX read+add of new pattern")
    out3_np = np.array(out3)
    expected3 = NEW_PATTERN + 0 + 2
    ok3 = out3_np[0] == expected3
    print(f"  MLX read of fresh-after-reclaim data: "
          f"{'PASS' if ok3 else 'FAIL'} (first={out3_np[0]:#x}, "
          f"expected={expected3:#x})")

    print("\nSummary:")
    print(f"  Pool constructed:                  yes")
    print(f"  CPU→MLX visibility:                {'yes' if ok else 'NO'}")
    print(f"  Reclaim released pages:            "
          f"{'yes' if freed > 0 else 'NO'}")
    print(f"  Reclaimed region zeroed:           "
          f"{'yes' if all_zero else 'NO'}")
    print(f"  GPU sees fresh post-reclaim data:  "
          f"{'yes' if ok3 else 'NO'}")


if __name__ == "__main__":
    main()
