# MVT-CS — Marginal Value of Thinking

Research code for a paper on test-time compute budget selection.
Claude Code operates this repo. Read this file before every task.

## Hardware (fixed, non-negotiable)

| GPU | VRAM | Arch | Notes |
|---|---|---|---|
| RTX 4090 x1 | 24 GB | sm_89 (Ada) | fastest, `CUDA_VISIBLE_DEVICES=0` [^xorg] |
| RTX 3090 x2 | 24 GB | sm_86 (Ampere) | `CUDA_VISIBLE_DEVICES=1,2` |

[^xorg]: GPU 0 carries a permanent ~19 MiB VRAM footprint from Xorg (9 MiB)
and gnome-shell (10 MiB). GPU 1 and GPU 2 each have ~4 MiB from Xorg only.
No iGPU is present; display cannot be offloaded. All three GPUs run at
`gpu_memory_utilization=0.85`; the asymmetry (~15 MiB, 0.06% of VRAM) is
negligible and documented here for reproducibility. (0.92 causes OOM in
vLLM 0.19.0 because CUDA graph memory is not profiled accurately by default;
0.85 is the fixed value for all three cards.)

No NVLink assumed. No multi-node. Single machine.

## Model requirement

**Primary models MUST have a native thinking mode.** Instruct-only models
(Qwen2.5-Instruct, Llama-3-Instruct, etc.) are invalid for this project —
the budget grid presupposes a reasoning trace. Using an instruct-only model
produces numbers silently, but those numbers are meaningless.

Primary model: **Qwen3-8B** (HF: `Qwen/Qwen3-8B`).

## Hard rules — violating any of these invalidates the paper

1. **Never use tensor parallelism for 7B/8B models.** They fit on one 24 GB
   card in BF16. Run three independent single-GPU vLLM processes and shard
   work with a queue. TP across 4090+3090 is slower, not faster.
2. **`enable_prefix_caching=True` always.** Checkpoint probes reuse the
   trajectory KV. Without it, cost is ~6x and the project is infeasible.
3. **`kv_cache_dtype="auto"` (BF16) — fp8_e5m2 is permanently forbidden.**
   D3 diagnostic (2026-08-02): fp8_e5m2 changed the top-1 token on 4/50
   prompts (8%) compared to BF16 on the same GPU. That is a different model,
   not a rounding error. This paper measures sub-0.5 pp Brier differences;
   logit corruption at the token level is a correctness bug, not a storage
   trade-off. Do not restore fp8 to recover KV cache capacity.
4. **Never mix GPU architectures inside one (model x dataset) cell** until
   `/m0-probe` has confirmed logit agreement. This is a calibration paper;
   pooled logits from different kernels are a correctness bug.
5. **Never change `BUDGET_GRID` or `SAMPLING`** in
   `m0_capacity_probe.py`. Any change invalidates every cached generation.
   If a change seems necessary, stop and ask.
6. **BF16 only for primary results.** 4-bit results go in an appendix
   ablation, never in a calibration table.
7. **Generate once, replay many.** Online-algorithm evaluation (stream
   permutations, policy racing, confidence sequences) runs on cached
   checkpoint tables on CPU. Never regenerate to vary stream order.

## Layout

```
m0_capacity_probe.py     # step 0: throughput + cross-GPU logit check
out/probe_<tag>.json     # one per GPU, written by the probe
```

## Environment

vLLM 0.19.0 + transformers, Python 3.12, venv at `/home/sclab/paper2/.venv`.
If vLLM is missing, report that and stop — do not silently fall back to
HF `generate`, the timings would be meaningless.

**Run command** (all three GPUs):
```bash
source /home/sclab/paper2/.venv/bin/activate
export CUDA_HOME=/tmp/claude-1000/-home-sclab-paper2/*/scratchpad/cuda_home
export PATH="$PATH:/home/sclab/miniconda3/envs/dualmoe/bin"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<0|1|2> python m0_capacity_probe.py \
    --gpu-tag <4090|3090a|3090b> --util 0.85
```

**FlashInfer JIT** (sm_89 Ada, sm_86 Ampere): vLLM 0.19.0 JIT-compiles
attention kernels for these architectures (no prebuilt cubins).
Requires `nvcc` (from `dualmoe` conda env, read-only) via CUDA_HOME fake dir:
```bash
FAKE=/tmp/.../cuda_home
mkdir -p $FAKE
ln -sfn /home/sclab/miniconda3/envs/dualmoe/bin          $FAKE/bin
ln -sfn .../dualmoe/targets/x86_64-linux/include          $FAKE/include
ln -sfn .../dualmoe/targets/x86_64-linux/lib              $FAKE/lib64
```
JIT cache: `~/.cache/flashinfer/0.6.6/{89,86}/` — survives reboots.
After first run per architecture the CUDA_HOME fake dir is no longer needed.
The `cuda_home` symlinks live in the Claude scratchpad (`/tmp/`) and must be
recreated if the scratchpad is cleared.

## Style

- No new dependencies without asking.
- Do not "improve" measurement code by adding warmups, retries, or
  averaging unless asked. Timing fidelity beats robustness here.
- Report raw numbers. Do not round away differences under 5%.
