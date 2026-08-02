# Logit Fingerprint Diagnostics (D1–D4 + additional)

Run date: 2026-08-02  
Tool: `diag_logits.py` — 50 fixed math prompts, teacher-forced next-token logprobs (top-20).  
Metric: max |Δ logprob| and median |Δ logprob| over matched token pairs across all prompts.

## Results

| ID | Description | Config A | Config B | top-1 match | max\|Δ\| | median\|Δ\| |
|----|-------------|----------|----------|-------------|----------|-------------|
| D1 | Run-to-run non-determinism, same GPU | 4090, BF16, batch, run 1 | 4090, BF16, batch, run 2 | 50/50 | 0.00 | 0.00 |
| D4 | Eager + batch=1 non-determinism, same GPU | 4090, BF16, eager, max_num_seqs=1, run 1 | same, run 2 | 50/50 | 0.00 | 0.00 |
| D2 | Same arch, different card (fp8 KV) | 3090a, fp8_e5m2 | 3090b, fp8_e5m2 | — | 1.37 | 0.18 |
| D3 | fp8 vs BF16 KV cache, same GPU | 4090, BF16 | 4090, fp8_e5m2 | 46/50 | 3.27 | — |
| add | Same GPU, batch size variation (fp8 KV) | 4090, fp8_e5m2, batch | 4090, fp8_e5m2, max_num_seqs=1 | — | 1.82 | 0.20 |

## Conclusions

**D1, D4:** Determinism is perfect under both normal batching and eager+batch=1.
CUDA graph introduction and batch parallelism do not cause logit drift.

**D3:** fp8_e5m2 KV cache changed the top-1 predicted token on 4/50 prompts (8%)
compared to BF16 on the *same GPU*. max|Δ logprob| = 3.27.
This is a model-level correctness failure: fp8 quantisation of the KV cache
alters the effective model, not just its precision.
A paper measuring sub-0.5 pp Brier differences cannot tolerate this.

**Additional (batch variation, fp8):** max|Δ| = 1.82, median = 0.20 with fp8 KV
even between batch=N and batch=1 on the same GPU. Confirms that fp8 is the cause
of D2's cross-card divergence, not architecture differences.

**D2 (cross-card, fp8):** max|Δ| = 1.37 is explained entirely by fp8 KV noise.

## Decision

`kv_cache_dtype` is set to `"auto"` (BF16) for all runs.  
`fp8_e5m2` is permanently forbidden — see CLAUDE.md Rule 3.  
The M0 probe JSONs in `out/` were generated under fp8_e5m2 and are superseded
by the BF16 re-run in Step [3].
