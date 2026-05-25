# vLLM Metal Plugin

> **High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

**Documentation**: https://docs.vllm.ai/projects/vllm-metal/en/latest/

---
*Latest News* 🔥

- [2026/04] We released the new version v0.2.0! Unified paged varlen Metal kernel is now the default attention backend. 83x TTFT, 3.6x throughput compared to v0.1.0.

---

## Requirements

- macOS on Apple Silicon

## Supported Models

vllm-metal supports a growing set of text-only language models on Apple Silicon. See the full matrix in [docs/supported_models.md](docs/supported_models.md).

## Installation
Using the install script, the following will be installed under the `~/.venv-vllm-metal` directory (the default).
- vllm-metal plugin
- vllm core
- Related libraries

If you run `source ~/.venv-vllm-metal/bin/activate`, the `vllm` CLI becomes available and you can access the vLLM right away.

For how to use the `vllm` CLI, please refer to the official vLLM guide.
https://docs.vllm.ai/en/latest/cli/

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

### Editable install

If you're hacking on vllm-metal locally, do an editable install so Python edits take effect without reinstalling. Activate the venv first, then:

```bash
uv pip install --no-deps -e .
```

`--no-deps` skips re-resolving dependencies (already installed by `install.sh`). Python changes under [vllm_metal/](vllm_metal/) are live on next import; Rust changes require rerunning the command.

## Elastic KV Cache (kvcached)

Experimental KV cache mode that lets steady-state memory track actually-used KV blocks instead of the full pool. The pool is sized the same as today; the difference is when pages are physically backed.

Each per-layer K and V cache is backed by an `mmap`'d region wrapped as an `MTL::Buffer`. macOS commits a physical page the first time a KV scatter writes to it. When a request finishes, the model runner unmaps the block ranges that backed it (page-aligned, 16 KB on Apple Silicon) and rebuilds the `MTL::Buffer` so the GPU's IOMMU mapping captures the freshly-unbacked pages. The next write to a reclaimed range silently re-commits a zero page.

**When to use it.** Workloads with high request churn where you'd otherwise size the pool generously and waste RSS on idle blocks. Net effect: more headroom for the rest of the system.

**Enable.** Set the env var at server launch (paged attention is already on by default; no install changes required — the native code JIT-builds on first import):

```bash
VLLM_METAL_ELASTIC_KV=1 vllm serve <MODEL>
```

**Status / caveats.**
- MHA/GQA paged backend only. MLA and hybrid (SDPA + recurrent) fall back to no-op.
- Not compatible with TurboQuant in the same cache (`MetalPagedKVCache(elastic=True, turboquant=True)` raises).
- Prefix caching (`VLLM_METAL_PREFIX_CACHE`) is not supported in elastic mode yet.
- Reclaim is synchronous: it calls `mx.synchronize()` then runs the mmap dance per range, so cost scales with the number of ranges finishing per scheduler tick.
- Partial pages at the ends of a freed range stay backed (ranges round inward to 16 KB).

See [docs/configuration.md](docs/configuration.md#elastic-kv-cache) for the full description.

### Optional: Rust frontend (experimental)

Pass `--with-vllm-rs` to also install [`vllm-frontend-rs`](https://github.com/Inferact/vllm-frontend-rs), an experimental Rust drop-in for vLLM's serving layer. Requires the Rust toolchain (https://rustup.rs):

```bash
./install.sh --with-vllm-rs
```

See [docs/rust_frontend.md](docs/rust_frontend.md) for usage and architecture.

