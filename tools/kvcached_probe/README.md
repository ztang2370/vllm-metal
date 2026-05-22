# kvcached-on-Metal viability probe

Tests whether `mmap(MAP_FIXED) + MTLBuffer.newBufferWithBytesNoCopy:` can be used
to implement kvcached-style elastic GPU memory on Apple Silicon — see
`probe.mm` for full details.

## Build & run

```bash
cd tools/kvcached_probe
clang++ -std=c++17 -fobjc-arc -O2 probe.mm \
    -framework Metal -framework Foundation -o probe
./probe
```

## What success looks like

Each step prints `rss` (regular VM working set) and `phys_fp` (the metric
Activity Monitor shows, which includes IOKit-mapped Metal memory). A successful
run answers four questions:

1. **Does Metal accept `newBufferWithBytesNoCopy:` on a `PROT_NONE` region?**
   If not, fail at step 2.
2. **Does `mmap(MAP_FIXED, PROT_READ|WRITE)` over a sub-range work after the
   buffer is wrapped?** If not, fail at step 3.
3. **Does a Metal compute kernel see the freshly-mapped pages?** Verified by
   step 4 + 5: kernel writes a pattern, CPU reads it back.
4. **Does unmapping a sub-range release physical pages?** Verified by step 6:
   `PROT_NONE` the window, re-map with `PROT_READ|WRITE`, check that the pages
   are zero (a fresh anonymous mapping).

If all four succeed, the kvcached approach is viable on Metal and we can build
an elastic KV pool that mirrors the CUDA design.
