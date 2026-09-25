#!/usr/bin/env python3
"""
M2 smoke test — DeepSeek-R1-Distill-Llama-8B, THINK_END_ID=128014.

Checks (in order):
  1. Tokenizer: "</think>" encodes to [128014]
  2. Tokenizer: decode([128014]) == "</think>"
  3. A-J option letter token IDs (must be stable across tokenizers)
  4. gen-prefixes: 5 items, budget=[256], generation stops at 128014 or hits max
  5. run-d5:       forced-prefix logprob extraction for A-J options

Run on GPU 1 (3090a):
  CUDA_VISIBLE_DEVICES=1 python m2_smoke_test.py

Results written to:
  data/smoke_m2_prefixes.json
  data/smoke_m2_results.json
  logs/smoke_m2.log  (stdout/stderr when run via run_corpus wrapper)
"""

import json, sys, time
from pathlib import Path

MODEL       = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
THINK_END   = 128014   # </think> in DeepSeek-R1-Llama tokenizer
BUDGET      = 256      # single small budget for smoke
N_ITEMS     = 5
UTIL        = 0.85

OPTION_LETTERS = list("ABCDEFGHIJ")

# Answer cue appended after thinking block to elicit answer letter.
# DeepSeek-R1 uses the same "</think>" boundary; the cue wording matches
# what diag_logits.py uses for M1 so prefix structures are comparable.
ANSWER_CUE = "\n</think>\n\nThe answer is **"

ITEMS_FILE = "data/items_ds1.json"
OUT_PREFIX = "data/smoke_m2_prefixes.json"
OUT_RESULT = "data/smoke_m2_results.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_probs(log_probs: list[float]) -> list[float]:
    import math
    m = max(log_probs)
    exp = [math.exp(lp - m) for lp in log_probs]
    s = sum(exp)
    return [e / s for e in exp]


def build_mcq_prompt(item: dict) -> str:
    opts = "\n".join(
        f"{'ABCDEFGHIJ'[i]}. {opt}"
        for i, opt in enumerate(item["options"])
    )
    return (
        f"Question: {item['question']}\n\nOptions:\n{opts}\n\n"
        "Think carefully, then output your answer."
    )


# ---------------------------------------------------------------------------
# Step 1+2: tokenizer checks (CPU only)
# ---------------------------------------------------------------------------

def check_tokenizer():
    print("=" * 60)
    print("STEP 1-3: Tokenizer checks (CPU)")
    print("=" * 60)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # 1. </think> encoding
    encoded = tok.encode("</think>", add_special_tokens=False)
    if encoded == [THINK_END]:
        print(f"[PASS] '</think>' → {encoded}  (== [{THINK_END}])")
    else:
        print(f"[FAIL] '</think>' → {encoded}  (expected [{THINK_END}])")
        sys.exit(1)

    # 2. token 128014 decoding
    decoded = tok.decode([THINK_END])
    if decoded == "</think>":
        print(f"[PASS] decode([{THINK_END}]) == '</think>'")
    else:
        print(f"[FAIL] decode([{THINK_END}]) == {repr(decoded)}  (expected '</think>')")
        sys.exit(1)

    # 3. A-J option IDs
    option_ids = []
    for letter in OPTION_LETTERS:
        ids = tok.encode(letter, add_special_tokens=False)
        option_ids.append(ids)
        status = "ok" if len(ids) == 1 else "MULTI-TOKEN"
        print(f"  {letter} → {ids}  [{status}]")

    single_token = [ids for ids in option_ids if len(ids) == 1]
    if len(single_token) != 10:
        print(f"[FAIL] Not all A-J are single-token in this tokenizer")
        sys.exit(1)
    print(f"[PASS] All A-J are single-token")

    flat_ids = [ids[0] for ids in option_ids]
    print(f"[INFO] OPTION_IDS = {flat_ids}")

    # 4. answer cue encoding
    cue_ids = tok.encode(ANSWER_CUE, add_special_tokens=False)
    print(f"[INFO] answer_cue len = {len(cue_ids)} tokens")

    return flat_ids, cue_ids


# ---------------------------------------------------------------------------
# Step 4: gen-prefixes smoke (GPU)
# ---------------------------------------------------------------------------

def run_gen_prefixes(option_ids: list[int], answer_cue_ids: list[int]):
    print()
    print("=" * 60)
    print("STEP 4: gen-prefixes smoke (GPU)")
    print("=" * 60)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load 5 items
    raw = json.loads(Path(ITEMS_FILE).read_text())
    items = raw["items"][:N_ITEMS]
    print(f"[INFO] Using {len(items)} items from {ITEMS_FILE}")

    # Build prompt token IDs via tokenizer
    tok_pre = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def build_prompt_ids(item):
        # Apply chat template: system message optional; use raw prompt to stay
        # consistent with diag_logits.py which uses build_mcq_prompt_ids.
        prompt_text = build_mcq_prompt(item)
        # Use chat template with thinking enabled (DeepSeek-R1 style)
        messages = [{"role": "user", "content": prompt_text}]
        text = tok_pre.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tok_pre.encode(text, add_special_tokens=False)

    prompt_ids_list = [build_prompt_ids(it) for it in items]
    max_prompt_len  = max(len(ids) for ids in prompt_ids_list)
    cue_len         = len(answer_cue_ids)
    MARGIN          = 64
    max_model_len   = max_prompt_len + BUDGET + cue_len + MARGIN
    print(f"[INFO] max_prompt_len={max_prompt_len}  BUDGET={BUDGET}  "
          f"cue_len={cue_len}  → max_model_len={max_model_len}")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        gpu_memory_utilization=UTIL,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()
    cue_ids_gpu = tok.encode(ANSWER_CUE, add_special_tokens=False)

    traj_params = SamplingParams(
        max_tokens=BUDGET,
        min_tokens=BUDGET,
        stop_token_ids=[THINK_END],
        temperature=0.0,
    )

    print(f"[INFO] Generating {len(items)} items with THINK_END_ID={THINK_END} ...")
    t0 = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": list(ids)} for ids in prompt_ids_list],
        traj_params,
        use_tqdm=True,
    )
    elapsed = time.time() - t0
    print(f"[INFO] Generation done in {elapsed:.1f}s")

    # Analyse stop tokens
    prefixes = []
    all_pass = True
    for i, (item, p_ids, out) in enumerate(zip(items, prompt_ids_list, outputs)):
        thinking = list(out.outputs[0].token_ids)
        stop_reason = out.outputs[0].stop_reason
        finish_reason = out.outputs[0].finish_reason

        # Strip trailing THINK_END if present
        ended_with_think = thinking and thinking[-1] == THINK_END
        if ended_with_think:
            thinking_clean = thinking[:-1]
        else:
            thinking_clean = thinking

        last_tok = thinking[-1] if thinking else None

        if ended_with_think:
            stop_status = f"[PASS] stopped at THINK_END ({THINK_END})"
        elif len(thinking) >= BUDGET:
            stop_status = f"[INFO] hit max_tokens={BUDGET} (thinking still running)"
        else:
            stop_status = f"[WARN] stopped at token {last_tok} (not THINK_END)"
            all_pass = False

        print(f"  item[{i}] thinking_len={len(thinking)}  last_tok={last_tok}"
              f"  finish={finish_reason}  stop_reason={stop_reason}  → {stop_status}")

        # Build prefix: prompt + thinking[:BUDGET] + answer_cue
        budget_ids = list(p_ids) + thinking_clean[:BUDGET] + list(cue_ids_gpu)
        prefixes.append({
            "question_id":  item["question_id"],
            "category":     item["category"],
            "n_options":    len(item["options"]),
            "answer":       item["answer"],
            "answer_index": item["answer_index"],
            "thinking_len": len(thinking_clean),
            "budgets":      {str(BUDGET): budget_ids},
        })

    if all_pass:
        print("[PASS] All items stopped correctly")
    else:
        print("[WARN] Some items did not stop at THINK_END — check budget size")

    out_data = {
        "model":         MODEL,
        "think_end_id":  THINK_END,
        "budgets":       [BUDGET],
        "max_model_len": max_model_len,
        "n_items":       len(prefixes),
        "items":         prefixes,
    }
    Path(OUT_PREFIX).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PREFIX).write_text(json.dumps(out_data, indent=2))
    print(f"[INFO] Wrote {OUT_PREFIX}")

    return option_ids, out_data


# ---------------------------------------------------------------------------
# Step 5: run-d5 smoke (GPU, same process)
# ---------------------------------------------------------------------------

def run_d5_smoke(option_ids: list[int], prefix_data: dict):
    print()
    print("=" * 60)
    print("STEP 5: run-d5 smoke (forced-prefix logprob extraction)")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    items = prefix_data["items"]
    all_ids = [
        ids
        for item in items
        for ids in item["budgets"].values()
        if ids is not None
    ]
    max_prefix_len = max(len(ids) for ids in all_ids)
    max_model_len  = max_prefix_len + 1
    print(f"[INFO] max_prefix_len={max_prefix_len}  → max_model_len={max_model_len}")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        gpu_memory_utilization=UTIL,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        max_logprobs=200,
    )
    tok = llm.get_tokenizer()

    # Verify option IDs are single-token in model tokenizer too
    mismatch = False
    for letter, expected_id in zip(OPTION_LETTERS[:len(option_ids)], option_ids):
        actual = tok.encode(letter, add_special_tokens=False)
        if actual != [expected_id]:
            print(f"[FAIL] Option ID mismatch for {letter}: "
                  f"expected [{expected_id}], got {actual}")
            mismatch = True
    if mismatch:
        sys.exit(1)
    print(f"[PASS] Option IDs consistent between tokenizer load and vLLM tokenizer")

    # Build prompts
    all_prompts, all_meta = [], []
    for item in items:
        ids = item["budgets"].get(str(BUDGET))
        if ids is None:
            continue
        all_prompts.append({"prompt_token_ids": ids})
        all_meta.append({
            "question_id":  item["question_id"],
            "category":     item["category"],
            "budget":       BUDGET,
            "answer":       item["answer"],
            "answer_index": item["answer_index"],
            "n_options":    item["n_options"],
        })

    lp_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=200)

    print(f"[INFO] Running {len(all_prompts)} prefix queries ...")
    lp_out = llm.generate(all_prompts, lp_params, use_tqdm=True)

    results = []
    all_pass = True
    for meta, o in zip(all_meta, lp_out):
        lp_map  = o.outputs[0].logprobs[0]
        n       = meta["n_options"]
        threshold = min(v.logprob for v in lp_map.values())
        raw_lp  = [
            float(lp_map[option_ids[i]].logprob) if option_ids[i] in lp_map else None
            for i in range(n)
        ]
        n_missing = sum(x is None for x in raw_lp)
        raw_lp_for_norm = [x if x is not None else float("-inf") for x in raw_lp]
        probs = _norm_probs(raw_lp_for_norm)
        pred_idx = probs.index(max(probs))
        correct  = pred_idx == meta["answer_index"]

        status = "PASS" if n_missing == 0 else "WARN"
        if n_missing > 0:
            all_pass = False
        print(f"  qid={meta['question_id']}  n_options={n}  n_missing={n_missing}"
              f"  pred={OPTION_LETTERS[pred_idx]}  ans={meta['answer']}"
              f"  correct={correct}  [{status}]")

        results.append({
            "question_id":  meta["question_id"],
            "category":     meta["category"],
            "budget":       BUDGET,
            "answer":       meta["answer"],
            "answer_index": meta["answer_index"],
            "n_options":    n,
            "option_probs": probs,
            "raw_logprobs": raw_lp,
            "threshold":    threshold,
            "n_missing":    n_missing,
        })

    if all_pass:
        print("[PASS] All option logprobs extracted (n_missing=0 for all items)")
    else:
        print("[WARN] Some option logprobs missing — option IDs may not be in top-200")

    output = {
        "model":       MODEL,
        "think_end_id": THINK_END,
        "gpu_tag":     "3090a",
        "n_results":   len(results),
        "results":     results,
    }
    Path(OUT_RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_RESULT).write_text(json.dumps(output, indent=2))
    print(f"[INFO] Wrote {OUT_RESULT}")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"M2 Smoke Test — {MODEL}")
    print(f"THINK_END_ID={THINK_END}  BUDGET={BUDGET}  N_ITEMS={N_ITEMS}")
    print()

    option_ids, cue_ids = check_tokenizer()
    option_ids, prefix_data = run_gen_prefixes(option_ids, cue_ids)
    logprob_ok = run_d5_smoke(option_ids, prefix_data)

    print()
    print("=" * 60)
    final = "PASS" if logprob_ok else "PARTIAL (check WARNs above)"
    print(f"SMOKE TEST: {final}")
    print("=" * 60)
