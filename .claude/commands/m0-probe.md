---
description: Run the M0 capacity probe on all three GPUs and decide the corpus size
---

Run the M0 step-0 capacity probe. Follow this exactly.

## 1. Preflight

- `nvidia-smi` — confirm three GPUs are visible and idle. If any card has
  existing processes or is above 50 C, report it and ask before starting.
  A hot 3090 throttles and the throughput number will be wrong.
- Confirm `vllm` imports and print its version. If it is missing, stop and
  report. Do not install anything.
- Confirm the model weights are present locally or note that the first run
  will download ~16 GB.

## 2. Run sequentially, one GPU at a time

Do not run these in parallel. Concurrent runs contend for PCIe and host
RAM and corrupt the timing measurement.

```
CUDA_VISIBLE_DEVICES=0 python m0_capacity_probe.py --gpu-tag 4090
CUDA_VISIBLE_DEVICES=1 python m0_capacity_probe.py --gpu-tag 3090a
CUDA_VISIBLE_DEVICES=2 python m0_capacity_probe.py --gpu-tag 3090b
```

Each takes roughly 10-25 minutes. Stream progress. If one fails with OOM,
do not lower `--util` on your own — report the failure with the vLLM
memory profile line and ask.

## 3. Compare

```
python m0_capacity_probe.py --compare
```

## 4. Report a decision, not a data dump

Give me exactly this, in a short table plus three verdicts:

**Table** — per GPU: `full_grid_s_per_item`, trajectory `tok_per_s`,
`probe_overhead_frac`; then combined items/hour for the rig.

**Verdict A — corpus size.** Using combined items/hour, state how many
hours a 2,700-item and an 8,100-item full-grid pass would take.
- under 20 h for 8,100 -> proceed with 3 primary models
- 20-40 h -> drop to 2 primary models
- over 40 h -> also cut MMLU-Pro subset from 1500 to 800, and say so

**Verdict B — prefix caching.** If `probe_overhead_frac` > 0.20 on any
card, prefix caching is not taking effect. Investigate: confirm
`enable_prefix_caching` is on, and check whether probe prompts are being
built from the exact detokenized prefix (a tokenizer round-trip mismatch
silently breaks cache hits). Report the cause. Do not reduce the budget
grid without asking.

**Verdict C — GPU pooling.** Report max and median `|Δ logprob|` for
4090-vs-3090a and 3090a-vs-3090b.
- max < 1e-3 -> all three cards poolable, say so explicitly
- otherwise -> we must pin each (model x dataset) cell to one card;
  state which pairs disagree and by how much

Then flag anything you noticed that I did not ask about — thermal
throttling mid-run, unexpected vLLM warnings, KV concurrency far below
the printed memory plan. Those matter more than the headline number.

Do not modify `m0_capacity_probe.py` during this task.
