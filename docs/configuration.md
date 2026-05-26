# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_MEMORY_FRACTION` | `auto` | `auto` allocates just enough memory plus a minimal KV cache, or `0.?` for fraction of memory |
| `VLLM_METAL_USE_MLX` | `1` | Use MLX for compute (1=yes, 0=no) |
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_USE_PAGED_ATTENTION` | `1` | Enable experimental paged KV cache |
| `VLLM_METAL_KV_SHARING_FAST_PREFILL` | `1` | Enable Gemma4 YOCO fast prefill for eligible Gemma4 models on the paged KV path |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |
| `VLLM_METAL_MULTIMODAL_MODE` | `auto` | Multimodal serve mode: `auto`, `text-only-compat`, or `multimodal-native` |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_PREFIX_CACHE` | (unset) | Set to enable prefix caching for shared prompt reuse |
| `VLLM_METAL_PREFIX_CACHE_FRACTION` | `0.05` | Fraction of MLX working set for prefix cache (0, 1] |
| `VLLM_METAL_GDN_LAZY_DECODE` | `1` | Enable lazy GDN decode kernels for eligible decode-only hybrid batches. Set to `0` to force the eager conv / C++ recurrent fallback path. |
| `VLLM_METAL_MLA_KERNEL` | `0` | Enable the experimental absorbed-MLA single-pass Metal decode kernel ([RFC #360](https://github.com/vllm-project/vllm-metal/issues/360)). Off by default; the MLA wrapper falls back to the MLX SDPA per-request slow path. Set to `1` to route absorbed-MLA decode through the kernel when the workload matches the instantiated specialization (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `block_size ∈ {16, 32}`, fp16/bf16, decode-only). |
| `VLLM_METAL_ELASTIC_KV` | `0` | Experimental elastic KV cache. Backs each per-layer paged KV array with an `mmap`'d region wrapped as an `MTL::Buffer`; freed-request blocks are unmapped + remapped (page-aligned) and the buffer is rebuilt so the GPU sees the fresh mapping. Currently MHA/GQA only — no-op for MLA and hybrid models. See [Elastic KV Cache](#elastic-kv-cache). |
| `VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION` | (unset) | Fraction of the elastic KV pool that vLLM's prefix cache may retain. Unset = unbounded (default; cached blocks keep pages backed indefinitely). A value in `[0, 1]` caps retention via LRU eviction so elastic reclaim continues under prefix-cache pressure. Only meaningful when `VLLM_METAL_ELASTIC_KV=1` and `--enable-prefix-caching` is on. |

## Multimodal Serve Modes

- `auto`: use native multimodal loading by default, but fall back to the text-only compatibility path for known-incompatible checkpoints such as Gemma4 and Qwen3.5/Qwen3.6 FP8 conditional-generation wrappers.
- `text-only-compat`: force the text-only compatibility path only for known-safe checkpoints such as Gemma4 and Qwen3.5/Qwen3.6 FP8 conditional-generation wrappers. Other multimodal checkpoints stay on the native multimodal loader.
- `multimodal-native`: disable the compatibility fallback and keep the native multimodal path active when validating or developing real multimodal support.

## Gemma4 YOCO Fast Prefill

Gemma4 YOCO fast prefill is enabled by default for eligible Gemma4 text models on the paged KV path. It runs the YOCO KV-shared decoder layers only on the selected logits positions during prefill, then scatters those hidden states back before the final norm and LM head. Set `VLLM_METAL_KV_SHARING_FAST_PREFILL=0` to disable it.

This path requires `VLLM_METAL_USE_PAGED_ATTENTION=1` and is currently limited to Gemma4/Gemma4 text models with KV-shared layers. Ineligible models continue without fast prefill; if `VLLM_METAL_KV_SHARING_FAST_PREFILL=1` was explicitly set, vllm-metal logs a warning for the skipped enablement.

## Paged KV vs MLX KV Memory Settings

- MLX path (`VLLM_METAL_USE_PAGED_ATTENTION=0`): `VLLM_METAL_MEMORY_FRACTION` must be `auto`.
- Paged KV path (`VLLM_METAL_USE_PAGED_ATTENTION=1`): `VLLM_METAL_MEMORY_FRACTION` can be `auto` or a numeric fraction in `(0, 1]`.
- For paged KV with `VLLM_METAL_MEMORY_FRACTION=auto`, vllm-metal uses a default fraction of `0.9`.

| `VLLM_METAL_MEMORY_FRACTION` | `VLLM_METAL_USE_PAGED_ATTENTION` | Valid? | Notes |
|--|--|--|--|
| `auto` | `0` | Yes | MLX path |
| `auto` | `1` | Yes | Paged KV path (default); defaults to 0.9 internally |
| `0.7` | `1` | Yes | Paged KV path with explicit memory budget |
| `0.7` | `0` | No | Explicit fraction without paged KV is invalid |

## Elastic KV Cache

`VLLM_METAL_ELASTIC_KV=1` enables experimental elastic KV semantics on the paged-attention path. The pool is sized the same as today; what changes is when pages are physically backed:

- **Allocation.** Each per-layer K and V cache is replaced by an `ElasticKVPool`. The pool reserves its bytes with `mmap(MAP_ANON|MAP_PRIVATE, PROT_READ|WRITE)` — virtual address space only, with no physical pages committed — and wraps that region as an `MTL::Buffer` via `newBufferWithBytesNoCopy:`. The MLX array view is constructed over that buffer. macOS commits a physical page the first time a KV scatter writes to it.
- **Request completion.** When a request finishes, the model runner translates its scheduler-assigned block ids into byte ranges and queues them on each layer's pool. Reclaim then runs the **mmap dance** per range — `mmap(MAP_FIXED, PROT_NONE)` followed by `mmap(MAP_FIXED, PROT_READ|WRITE)` — which forcibly drops the physical pages, and rebuilds the `MTL::Buffer` so the GPU's IOMMU mapping captures the fresh (unbacked) pages. The next write to a reclaimed range silently re-commits a zero page.

The net effect is that steady-state RSS tracks actually-touched KV blocks rather than the full pool, leaving more headroom for the rest of the system.

Caveats:
- The mmap dance is **synchronous** — `reclaim()` calls `mx.synchronize()` first so the old `MTL::Buffer` is no longer referenced by in-flight kernels. Reclaim cost scales with the number of pending ranges; it runs on every request-completion batch in the model runner cleanup hook.
- Page-aligned only. Ranges are rounded **inward** to system page boundaries (16 KB on Apple Silicon), so partial pages at the ends of a freed range remain backed.
- Only the MHA/GQA paged backend is wired today. MLA and hybrid (SDPA + recurrent) backends fall back to a no-op for now.
- Not compatible with TurboQuant in the same cache (`MetalPagedKVCache(elastic=True, turboquant=True)` raises).
- Requires `VLLM_METAL_USE_PAGED_ATTENTION=1`. Enabling elastic mode without paged attention raises at startup.

### Interaction with prefix caching

Elastic mode composes with vLLM's stock paged prefix caching (`--enable-prefix-caching`). vllm-metal installs a once-per-process hook on `BlockPool.free_blocks` that forwards block releases to the per-layer `ElasticKVPool` reclaim queue — but only for blocks whose `block_hash` is `None` after the free. Blocks the prefix cache retains for cross-request reuse keep their pages backed (otherwise a later cache hit would read garbage), while partial last blocks and other uncached releases drop their pages as usual. Blocks evicted from the prefix cache during a subsequent allocation are immediately reused by the new request, so the hook intentionally does not reclaim them.

By default the prefix cache is uncapped, so a workload with high prefix diversity can fill the entire pool with retained blocks and effectively neutralize elastic reclaim. `VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION` bounds this:

| Setting | Effect |
|--|--|
| unset / empty | Unbounded retention (default). |
| `0` | No retention — every cached block is evicted on free. Equivalent to disabling prefix caching for RSS purposes; cache lookups still hit while a request holds the block. |
| `0.25` | Up to 25% of the pool may be held by the cache. Excess blocks evict LRU-oldest and their pages reclaim. |
| `1.0` | Equivalent to unbounded. |

The cap is enforced inside the same `free_blocks` call that pushed the pool over: the LRU walker pops the oldest cached-and-freed block, clears its hash via `BlockPool._maybe_evict_cached_block` (so future cache lookups miss), and forwards the block id to the elastic backend. The walker tolerates stale LRU entries — blocks re-acquired by `touch` or already un-cached by `get_new_blocks` — so the hook does not need to patch those code paths.

The legacy `VLLM_METAL_PREFIX_CACHE` flag controls a separate, contiguous (non-paged) prefix cache that is structurally inactive whenever paged attention is on, and is therefore unaffected by elastic mode.
