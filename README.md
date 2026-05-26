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
- vLLM prefix caching (`--enable-prefix-caching`) is supported: blocks held by the prefix cache keep their pages backed for reuse; only blocks that vLLM actually releases (cache miss, partial last block, eviction) get reclaimed. Set `VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION=<0..1>` to cap how much of the pool the cache may hold — excess cached blocks evict LRU-oldest and their pages are reclaimed. The legacy non-paged `VLLM_METAL_PREFIX_CACHE` is unrelated and stays a no-op when paged attention is on.
- Reclaim is synchronous: it calls `mx.synchronize()` then runs the mmap dance per range, so cost scales with the number of ranges finishing per scheduler tick.
- Partial pages at the ends of a freed range stay backed (ranges round inward to 16 KB).

See [docs/configuration.md](docs/configuration.md#elastic-kv-cache) for the full description.

### Reproducing the agent workload benchmark

End-to-end setup we use to compare kvcached against plain vllm-metal on a real tool-calling agent workload (Hermes CLI driving `Qwen/Qwen3-8B-MLX-4bit`).

**1. Launch the server.** Run one of the two commands below in an activated `~/.venv-vllm-metal` shell. The only difference is the `VLLM_METAL_ELASTIC_KV=1` env var — everything else is identical so the comparison is apples-to-apples.

The model's stock `max_position_embeddings` is 40960, so vLLM would otherwise cap `--max-model-len` below Hermes's 64k minimum. `--hf-overrides` applies the YaRN rope-scaling config from the [Qwen3-8B-MLX-4bit model card](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit) (`factor=4`, base 32768 → max position 131072), which is what unlocks `--max-model-len 64000`. Prefix caching is on by default for both runs (vLLM auto-enables it for supported models).

Ours (kvcached on):
```bash
VLLM_METAL_ELASTIC_KV=1 \
vllm serve Qwen/Qwen3-8B-MLX-4bit \
  --max-model-len 64000 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-log-requests
```

Baseline (kvcached off):
```bash
vllm serve Qwen/Qwen3-8B-MLX-4bit \
  --max-model-len 64000 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-log-requests
```

Flag notes:
- `--enable-auto-tool-choice` + `--tool-call-parser hermes` make vLLM emit OpenAI-style `tool_calls` parsed out of the Qwen3 `<tool_call>...</tool_call>` blocks, which is the format the Hermes agent expects.
- `--enable-log-requests` logs each request's prompt and generated output — handy for inspecting what the agent is sending.
- Optional: cap how much of the KV pool the prefix cache may pin with `VLLM_METAL_ELASTIC_KV_MAX_CACHED_FRACTION=0.5` (defaults to no cap).

**2. Configure the Hermes agent.** Point Hermes at the local vLLM endpoint by writing `~/.hermes/config.yaml`. Minimal `model:` block (the rest of the file can stay at defaults):

```yaml
model:
  provider: vllm
  model: Qwen/Qwen3-8B-MLX-4bit
  base_url: http://127.0.0.1:8000/v1
  api_key: none
  context_length: 64000
```

`model` must match the served model id exactly. `context_length` is technically optional when `--max-model-len ≥ 64000` (Hermes will pick up the model's max from the server), but we set it explicitly to make the config self-documenting.

If your Mac doesn't have enough memory to serve at `--max-model-len 64000`, you can drop the server below 64k (e.g. `--max-model-len 40000`) and keep `context_length: 64000` in the config purely to satisfy Hermes's ≥64k startup check. It's an imperfect workaround — Hermes will think it has 64k available and may send prompts the server then rejects — but for short-to-medium traces it works in practice.

**3. Drive the workload.** Start `hermes` in a second terminal and run whichever task you want to measure.

## Optional: Rust frontend (experimental)

Pass `--with-vllm-rs` to also install [`vllm-frontend-rs`](https://github.com/Inferact/vllm-frontend-rs), an experimental Rust drop-in for vLLM's serving layer. Requires the Rust toolchain (https://rustup.rs):

```bash
./install.sh --with-vllm-rs
```

See [docs/rust_frontend.md](docs/rust_frontend.md) for usage and architecture.

