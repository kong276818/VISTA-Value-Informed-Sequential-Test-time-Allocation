# VISTA: Value-Informed Sequential Test-time Allocation

Research code for a paper studying how MCQ accuracy varies with reasoning-token budget across a discrete grid.
We generate checkpoint prefixes at each budget level, teacher-force the answer cue, and measure option Brier scores — all from cached trajectories, never re-generating.

---

## Overview

Modern reasoning models (Qwen3, DeepSeek-R1) allow explicit control over how many tokens are spent thinking before an answer is produced. This project asks: **at what budget does further thinking stop paying off, and does that knee point differ by task difficulty or model family?**

The corpus is built once, stored as prefix checkpoints, and all downstream analysis (budget policies, confidence sequences, model comparisons) runs offline on CPU from the frozen tables.

---

## Hardware

| GPU | VRAM | Arch | Role |
|---|---|---|---|
| RTX 4090 × 1 | 24 GB | sm\_89 (Ada) | M1 (Qwen3-8B), all datasets |
| RTX 3090 × 2 | 24 GB | sm\_86 (Ampere) | M2 (DeepSeek-R1-Distill-Llama-8B), split across datasets |

All runs use `gpu_memory_utilization=0.85`, BF16 weights, BF16 KV cache (`kv_cache_dtype=auto`), and `enable_prefix_caching=True`.

---

## Models

| ID | HF path | Thinking format |
|----|---------|----------------|
| M1 | `Qwen/Qwen3-8B` | `<think>…</think>` (token 151667/151668) |
| M2 | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `<think>…</think>` (token 128014) |

Both models must have **native thinking mode**. Instruct-only models are invalid — the budget grid presupposes a reasoning trace.

---

## Datasets

| ID | Dataset | HF path | Items |
|----|---------|---------|------:|
| DS1 | MMLU-Pro | `TIGER-Lab/MMLU-Pro` | 1 000 |
| DS2 | ARC-Challenge | `allenai/ai2_arc` (ARC-Challenge) | 1 000 |
| DS3 | MedMCQA | `medmcqa` | 500 |
| DS4 | MedQA USMLE | `bigbio/med_qa` | 500 |

Total: **2 698 items × 2 models = 5 396 items**.  
Budget grid: **[256, 512, 1024, 2048, 4096, 8192]** reasoning tokens (6 levels).  
Total prefix slots: 5 396 × 6 = **32 376**.

---

## Key Diagnostics (completed)

### Logit Fingerprinting (D1–D4)

| ID | Test | Result |
|----|------|--------|
| D1 | Run-to-run determinism, same GPU | ✅ 50/50, Δ = 0.00 |
| D4 | Eager vs batch scheduler, same GPU | ✅ 50/50, Δ = 0.00 |
| D3 | fp8 vs BF16 KV cache, same GPU | ❌ 46/50 (8% top-1 change) — **fp8 permanently banned** |
| D2 | Cross-card fp8 | ❌ max\|Δlogprob\| = 1.37 — explained by fp8 noise |

### MCQ Stability (D5, MMLU-Pro 300 items × 3 budgets)

| Comparison | b=256 | b=2048 | b=8192 |
|-----------|-------|--------|--------|
| 4090 batch vs 4090 eager×1 | PASS (paired Brier < 0.002) | PASS | PASS |
| 3090a vs 3090b (same Ampere arch) | PASS (byte-identical) | PASS | PASS |

**Rule**: 4090 (sm\_89) and 3090 (sm\_86) results must never be pooled within the same (model × dataset) cell.

### M0 Throughput Probe (BF16)

| GPU | s/item | traj tok/s |
|---|---:|---:|
| 4090 | 9.19 | 424.4 |
| 3090a | 11.74 | 328.2 |
| 3090b | 10.00 | 385.0 |

Combined (3 GPUs parallel): **~1 058 items/hour**, wall ~6.9 h for the full M1 corpus.

---

## Repository Layout

```
m0_capacity_probe.py   # Step 0: throughput probe + cross-GPU logit agreement
diag_logits.py         # Logit fingerprint diagnostics (D1–D5) + gen-prefixes + run-d5
prep_corpus.py         # Download and serialise dataset items to data/items_*.json
run_corpus.py          # Coordinator: launches gen-prefixes + run-d5 across all GPU tracks
run_ablations.py       # Offline ablation runner (CPU, operates on cached prefix tables)

data/
  items_ds{1..4}.json                  # prepared dataset items
  prefixes_m1_ds{1..4}_4090.jsonl      # prefix checkpoint tables (LFS for large files)
  results_m1_ds{1..4}_4090.json        # option_probs + Brier per (item, budget)

docs/
  diagnostics.md   # full D1–D5 results and decisions
  corpus_plan.md   # GPU assignment, timeline, pre-flight checklist
```

---

## Quickstart

### 0. Environment

```bash
source /home/sclab/paper2/.venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="$PATH:/home/sclab/miniconda3/envs/dualmoe/bin"  # nvcc for FlashInfer JIT
```

### 1. Capacity probe (run once per GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python m0_capacity_probe.py --gpu-tag 4090  --util 0.85
CUDA_VISIBLE_DEVICES=1 python m0_capacity_probe.py --gpu-tag 3090a --util 0.85
CUDA_VISIBLE_DEVICES=2 python m0_capacity_probe.py --gpu-tag 3090b --util 0.85
python m0_capacity_probe.py --compare out/probe_*.json
```

### 2. Prepare dataset items

```bash
python prep_corpus.py
```

### 3. Run corpus generation (all GPU tracks in parallel)

```bash
python run_corpus.py          # full run
DRY_RUN=1 python run_corpus.py  # 4-item smoke test
```

Each cell runs `gen-prefixes` (checkpoint prefix table) then `run-d5` (teacher-forced logprobs). Outputs are written incrementally to `data/` with config-hash guards; a crashed run can be resumed by re-running the same command.

---

## Hard Rules (violating any invalidates the paper)

1. **No tensor parallelism for 8B models** — each fits on one 24 GB card; TP is slower.
2. **`enable_prefix_caching=True` always** — without it, cost is ~6× and the project is infeasible.
3. **BF16 KV cache only** — fp8_e5m2 changes top-1 tokens on 8% of prompts (D3); permanently banned.
4. **No architecture mixing per cell** — 4090 (sm\_89) and 3090 (sm\_86) produce logit differences up to |Δ| = 0.374; each (model × dataset) cell is pinned to a single GPU.
5. **`BUDGET_GRID` and `SAMPLING` are frozen** — any change invalidates all cached generations.
6. **Generate once, replay many** — all analysis (policy comparison, confidence sequences) runs on cached tables; never regenerate to vary stream order.

---

## Environment

- Python 3.12, vLLM 0.19.0
- FlashInfer JIT-compiled for sm\_89 (Ada) and sm\_86 (Ampere); first run per architecture takes ~5 min, results cached in `~/.cache/flashinfer/`
- Large data files tracked via Git LFS (`data/prefixes_m1_ds1_4090.{json,jsonl}`, `data/prefixes_m1_ds2_4090.jsonl`)
