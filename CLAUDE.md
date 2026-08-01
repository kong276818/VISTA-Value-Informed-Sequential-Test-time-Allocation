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
`gpu_memory_utilization=0.92`; the asymmetry (~15 MiB, 0.06% of VRAM) is
negligible and documented here for reproducibility.

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
3. **`kv_cache_dtype="fp8_e5m2"`.** Roughly doubles concurrent sequences at
   8k context. Storage-only quantization — works on Ampere.
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

vLLM + transformers, Python 3.10+. If vLLM is missing, report that and
stop — do not silently fall back to HF `generate`, the timings would be
meaningless.

## Style

- No new dependencies without asking.
- Do not "improve" measurement code by adding warmups, retries, or
  averaging unless asked. Timing fidelity beats robustness here.
- Report raw numbers. Do not round away differences under 5%.
