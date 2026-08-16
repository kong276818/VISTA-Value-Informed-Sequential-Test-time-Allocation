# M0 Capacity Probe — BF16 (판정 A 갱신)

Run date: 2026-08-04  
Tool: `m0_capacity_probe.py`, `--util 0.85 --kv-cache-dtype auto`.  
Output: `out/probe_{tag}_bf16.json`. N_TIMING_ITEMS=24, N_LOGIT_ITEMS=50.  
이전 fp8 수치(824 items/hour, util=0.92)는 폐기.

## 판정 A: 처리량

| GPU | s/item | traj tok/s | probe overhead | 2700항목 (단독) | 8100항목 (단독) |
|-----|-------:|----------:|:--------------:|----------------:|----------------:|
| 4090  (sm_89) | 9.19 | 424.4 | 7.0% | 6.9 h | 20.7 h |
| 3090a (sm_86) | 11.74 | 328.2 | 8.5% | 8.8 h | 26.4 h |
| 3090b (sm_86) | 10.00 | 385.0 | 8.4% | 7.5 h | 22.5 h |

리그 합산 (세 GPU 병렬): **1058 items/hour**, wall 3.40 s/item.  
BF16 동시 시퀀스 (8192-tok): 4090 ~4개 (KV 1.12 GiB/seq, weights 15.3 GiB, budget 20.0 GiB).

## 판정 B: 아키텍처 간 로짓 일치 (BF16)

참조 GPU: 4090 (sm_89).

| 비교 | top-1 일치 | max\|Δlogprob\| | median\|Δlogprob\| | 판정 |
|------|:----------:|----------------:|-------------------:|------|
| 4090 vs 3090a | 49/50 | 3.74e-01 | 3.75e-02 | **셀 고정 필수** |
| 4090 vs 3090b | 50/50 | 3.74e-01 | 3.53e-02 | **셀 고정 필수** |

max\|Δ\| = 0.374 ≫ 1e-3 임계값 → **4090과 3090을 동일 (model × dataset) 셀에 혼용 불가.**  
각 셀은 단일 GPU 아키텍처에 고정. CLAUDE.md rule 4 적용.

---

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

---

# D5: MCQ Logit Stability (Qwen3-8B, BF16, MMLU-Pro)

Run date: 2026-08-04  
Tool: `diag_logits.py` — `run-d5` (4×) + `compare-d5` (2×).  
300 items × 3 budgets [256, 2048, 8192]. Metric: paired mean |Brier(A)−Brier(B)|, threshold < 0.005.  
Answer cue: `"\n</think>\n\nThe answer is **"` → bare letter tokens A–J (IDs 32–41) at rank 1 (top-1 rate 99.4%, mean mass 0.984).

## D5-a: 4090 batch vs 4090 eager×1 (same GPU, different vLLM scheduler config)

| budget | top-1 match | max\|Δlogprob\| | paired mean \|Brier\| | CI.hi | VERDICT |
|--------|:-----------:|----------------:|---------------------:|------:|---------|
| 256    | 298/300     | 0.853           | 0.00165              | 0.00205 | **PASS** |
| 2048   | 298/300     | 0.625           | 0.00056              | 0.00079 | **PASS** |
| 8192   | 299/300     | 0.500           | 0.00019              | 0.00032 | **PASS** |

sanity check (perturbed B[0]): mean Brier diff = 0.000802 ≠ 0 ✓

## D5-b: 3090a vs 3090b (same Ampere architecture, different cards)

| budget | top-1 match | max\|Δlogprob\| | paired mean \|Brier\| | CI.hi | VERDICT |
|--------|:-----------:|----------------:|---------------------:|------:|---------|
| 256    | 300/300     | 0.218           | 0.00001              | 0.00004 | **PASS** |
| 2048   | 300/300     | 0.125           | 0.00000              | 0.00000 | **PASS** |
| 8192   | 300/300     | 0.065           | 0.00000              | 0.00000 | **PASS** |

sanity check (perturbed B[0]): mean Brier diff = 0.000005 ≠ 0 ✓

## D5 Conclusions

**D5-b (same arch):** 3090a and 3090b produce byte-identical logits at all budgets.
Pooling within the same architecture is confirmed safe.

**D5-a (batch config):** CUDA graph + continuous batching vs eager + max\_num\_seqs=1
on the same GPU produces small but non-zero logit differences: up to 0.853 logprob
units at b=256. Two of 300 items (b=256) and one item (b=8192) have different top-1
forced-answer tokens. Brier differences are below threshold at every budget and decay
with prefix length. All corpus generation uses a single fixed scheduler config per cell.

## Limitation Note (for paper)

BF16 KV cache 기준, 동일 GPU에서 배치 구성(CUDA graph + continuous batching vs eager + max_num_seqs=1)이 달라지면 MMLU-Pro 300문항 중 2문항에서 강제 답변 토큰이 바뀌었다 (298/300 at b=256, 299/300 at b=8192). paired mean Brier 차이는 0.00165 이하이며 prefix 길이가 길수록 감소한다. 동일 아키텍처 카드 간에는 300/300 일치. 본 실험의 모든 생성은 단일 설정으로 수행하나, 시스템 구성 변경이 개별 문항 수준에서 답을 바꿀 수 있음을 명시한다.
