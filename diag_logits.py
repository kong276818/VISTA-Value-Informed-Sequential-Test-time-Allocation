#!/usr/bin/env python3
"""
Logit-fingerprint diagnostic — D1 through D5.

Free-generation track (math prompts, 50 items):
  run      Load model, generate logit fingerprint, save JSON.
  compare  Compare two fingerprint JSONs (max/median |Δ logprob|).

  D1  run-to-run non-determinism, same GPU:
        python diag_logits.py run --gpu-tag 4090 --out /tmp/d1a.json
        python diag_logits.py run --gpu-tag 4090 --out /tmp/d1b.json
        python diag_logits.py compare /tmp/d1a.json /tmp/d1b.json
  D2  same arch, different card:
        (run 3090a, 3090b)  then compare
  D3  fp8 vs BF16 KV cache, 4090:
        python diag_logits.py run --gpu-tag 4090 --kv-cache-dtype auto  --out /tmp/d3bf.json
        python diag_logits.py compare /tmp/d1a.json /tmp/d3bf.json
  D4  eager mode, batch=1, 4090:
        python diag_logits.py run --gpu-tag 4090 --enforce-eager --max-num-seqs 1 --out /tmp/d4a.json

MCQ track (MMLU-Pro, 300 items):
  prep-d5        (CPU) Sample 300 MMLU-Pro items → data/d5_items.json
  gen-prefixes   (GPU, once on 4090) Generate reference thinking traces → data/d5_prefixes.json
  run-d5         (GPU) Compute A–J logits from saved prefix IDs → data/d5_<tag>_<cfg>.json
  compare-d5     (CPU) Full comparison: Brier, L1, L∞, bootstrap CI

  D5 workflow:
    python diag_logits.py prep-d5
    CUDA_VISIBLE_DEVICES=0 python diag_logits.py gen-prefixes --gpu-tag 4090
    # D5-a: same GPU, two batch configs
    CUDA_VISIBLE_DEVICES=0 python diag_logits.py run-d5 --gpu-tag 4090 --out data/d5_4090_batch.json
    CUDA_VISIBLE_DEVICES=0 python diag_logits.py run-d5 --gpu-tag 4090 --enforce-eager \\
        --max-num-seqs 1 --out data/d5_4090_eager1.json
    python diag_logits.py compare-d5 data/d5_4090_batch.json data/d5_4090_eager1.json
    # D5-b: cross-card, same arch
    CUDA_VISIBLE_DEVICES=1 python diag_logits.py run-d5 --gpu-tag 3090a --out data/d5_3090a_batch.json
    CUDA_VISIBLE_DEVICES=2 python diag_logits.py run-d5 --gpu-tag 3090b --out data/d5_3090b_batch.json
    python diag_logits.py compare-d5 data/d5_3090a_batch.json data/d5_3090b_batch.json

KV memory report (run separately per config):
  CUDA_VISIBLE_DEVICES=0 python diag_logits.py mem-report --config a --gpu-tag 4090
  CUDA_VISIBLE_DEVICES=0 python diag_logits.py mem-report --config b --gpu-tag 4090
"""

import argparse, json, math, random, statistics, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared constants — must match m0_capacity_probe.py exactly
# ---------------------------------------------------------------------------

THINK_START_ID = 151667   # <think>
THINK_END_ID   = 151668   # </think>
TOP_LOGPROBS   = 20
DEFAULT_MODEL  = "Qwen/Qwen3-8B"
B_MAX          = 8192     # matches m0_capacity_probe.BUDGET_GRID[-1]

# D5 configuration (MCQ track)
D5_BUDGETS     = [256, 2048, 8192]   # thinking-token checkpoints to probe
D5_ITEM_COUNT  = 300
D5_SEED        = 42

# A–J option letter token IDs in Qwen3 tokenizer (verified 2026-08-02).
# tok.encode("A", add_special_tokens=False) == [32], ..., "J" == [41].
OPTION_IDS     = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
OPTION_LETTERS = list("ABCDEFGHIJ")

# Exactly the same 50 prompts as m0_capacity_probe._PROMPTS.
# Kept in sync by copy; both lists must be identical for cross-run comparison.
PROMPTS = [
    # geometry / measurement
    "A regular hexagon has area 96. Find the area of the triangle formed by three alternating vertices. Show all steps.",
    "A circle of radius 5 is tangent to both coordinate axes in the first quadrant. Find its equation and verify.",
    "A trapezoid has parallel sides of length 6 and 10, and a height of 4. Find its area and the length of each non-parallel side if the trapezoid is isosceles.",
    "A ladder 10 m long leans against a vertical wall. Its foot is 6 m from the wall. How high does the top reach and what angle does it make with the ground?",
    "Find the number of diagonals of a convex polygon with 12 sides. Derive the general formula for n sides.",
    "A rectangle has perimeter 36 cm. Find the dimensions that maximize its area and verify it is indeed a maximum.",
    "A square of side 8 has a circle inscribed in it and a circle circumscribed around it. Find the area between the two circles.",
    "A cylinder and a cone share the same base radius r and height h. Find the ratio of their volumes and total surface areas.",
    "Two circles of radii 3 and 5 are externally tangent. Find the length of their common external tangent.",
    "In triangle ABC, angle A = 45°, angle B = 60°, and side c = 10. Find sides a and b using the sine rule.",
    # algebra / functions
    "Let f(x)=x^4-6x^2+8. Find every real root and prove none were missed.",
    "Factor x^6 - 64 completely over the integers into irreducible factors.",
    "If the polynomial p(x) = x^3 + ax^2 + bx + 6 has roots 1, 2, and 3, find a, b, and verify by expanding.",
    "Solve the system: 3x + 2y - z = 7, x - y + 2z = 1, 2x + 3y + z = 12. Verify your solution.",
    "Find all real solutions to |3x - 2| + |x + 1| = 9.",
    "Find the range of f(x) = (x^2 - 1)/(x^2 + 1). Show that every value in the range is achieved.",
    "Solve: 2^(2x) - 5 * 2^x + 4 = 0. Find all real solutions, showing all steps.",
    "Find all x such that log_3(x) + log_3(x-2) = 1. Check for extraneous solutions.",
    "Find all local maxima and minima of f(x) = x^3 - 6x^2 + 9x + 2. Confirm using the second derivative test.",
    "A geometric sequence has a_1 = 3 and a_4 = 81. Find the common ratio, a_7, and the sum of the first 6 terms.",
    # number theory
    "Prove that for all positive integers n, 6 divides n^3 - n. Use factoring and divisibility arguments.",
    "Find all integers n such that n^2 + 3n - 18 is a perfect square.",
    "Find all positive integer solutions to the Diophantine equation 3x + 5y = 47.",
    "Find the remainder when 17^100 is divided by 13. Use Fermat's little theorem.",
    "Prove that sqrt(3) + sqrt(5) is irrational. (Hint: square it and reach a contradiction.)",
    "Seven distinct integers sum to 0 and their product is 5040. Determine all possible multisets, justifying exhaustiveness.",
    "Find all prime numbers p such that p^2 + 2 is also prime. Prove no others exist.",
    "Find all n such that n! + 1 is divisible by n + 1, for 1 <= n <= 10. Check each case.",
    "The digits of a two-digit number are reversed to form a new number. The sum of original and new is 121, difference is 45. Find both numbers.",
    "Find all positive integer pairs (x, y) satisfying 1/x + 1/y = 1/6.",
    # probability / combinatorics
    "A fair coin is flipped until two consecutive heads appear. Compute the expected number of flips, deriving the recurrence.",
    "A bag contains 4 red, 3 blue, and 5 green balls. You draw 3 without replacement. What is the probability exactly 2 are red?",
    "Two dice are rolled. Given the sum is greater than 7, what is the conditional probability that at least one die shows a 6?",
    "In a class of 40 students, 25 study mathematics, 20 study physics, and 8 study both. How many study neither?",
    "How many 5-letter strings can be formed from A, B, C, D, E (no repeats) such that the string starts with A or ends with E?",
    "A committee of 5 is chosen from 8 men and 6 women. How many committees have at least 3 women?",
    "In how many ways can 8 people be seated at a round table? Account for rotational symmetry and explain your counting.",
    "A biased coin lands heads with probability 2/3. If flipped 5 times, what is the probability of exactly 3 heads?",
    "How many 4-digit positive integers have all distinct digits and digit sum equal to 12?",
    "From a group of 10 people, two teams of 4 are chosen simultaneously. How many ways can this be done?",
    # analysis / series / induction
    "Prove by mathematical induction that 1^2 + 2^2 + ... + n^2 = n(n+1)(2n+1)/6 for all positive integers n.",
    "Compute the sum of the infinite series 1/(1*2) + 1/(2*3) + 1/(3*4) + ... using partial fractions and telescoping.",
    "The sum of the first n terms of a sequence is S_n = n^2 + 2n. Find a_n for n >= 2 and check if n=1 fits.",
    "Given vectors a = (2, -1, 3) and b = (-1, 4, 2), find a.b, |a|, |b|, and the angle between them.",
    "A ball is thrown upward at 20 m/s. Using g = 10 m/s^2, find the max height, time to reach it, and time to return.",
    # word problems / logic / mixed
    "A farmer has 100 m of fencing to enclose a rectangular field along a river (no fence on the river side). What dimensions maximize the area?",
    "Three pipes A, B, C fill a tank in 4, 6, and 12 hours. A and B are open for the first 2 hours, then all three. How long total to fill?",
    "A store sells apples at $3 and oranges at $5. A customer spends exactly $44 buying some of each. Find all solutions.",
    "Prove that in any group of 13 people, at least two share a birth month. State which theorem you are applying.",
    "A ship goes 40 km north, then 30 km east, then 20 km south. How far is it from the starting point and in what direction?",
]

assert len(PROMPTS) == 50, f"expected 50 prompts, got {len(PROMPTS)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_prompt_ids(tok, problem):
    msgs = [{"role": "user", "content": problem}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def build_mcq_prompt_ids(tok, item):
    """Token IDs for a MCQ item (question + lettered options, no system prompt)."""
    opts = item["options"]
    opt_str = "\n".join(f"{OPTION_LETTERS[i]}. {o}" for i, o in enumerate(opts))
    content = f"{item['question']}\n\nOptions:\n{opt_str}"
    msgs = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def brier_score(probs, answer_index):
    """Multi-class Brier: summed squared deviation from one-hot (standard formulation, NOT ÷K)."""
    return sum((p - (1.0 if i == answer_index else 0.0)) ** 2
               for i, p in enumerate(probs))


def bootstrap_mean_ci(data, n_boot=10000, seed=0, level=0.95):
    """Bootstrap CI for the mean of data."""
    rng = random.Random(seed)
    n = len(data)
    boot = sorted(
        statistics.mean(data[rng.randrange(n)] for _ in range(n))
        for _ in range(n_boot)
    )
    lo = boot[int((1 - level) / 2 * n_boot)]
    hi = boot[int((1 + level) / 2 * n_boot)]
    return lo, hi


def _norm_probs(raw_logprobs):
    """Softmax over a list of log-probs (may include -inf)."""
    finite = [x for x in raw_logprobs if x > float("-inf")]
    if not finite:
        return [1.0 / len(raw_logprobs)] * len(raw_logprobs)
    max_lp = max(raw_logprobs)
    exp_lp = [math.exp(x - max_lp) if x > float("-inf") else 0.0 for x in raw_logprobs]
    total = sum(exp_lp)
    return [x / total for x in exp_lp]


# ---------------------------------------------------------------------------
# Free-generation track: run / compare (D1–D4)
# ---------------------------------------------------------------------------

def cmd_run(args):
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.util,
        max_model_len=8704,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    print(f"[diag] loading model  gpu_tag={args.gpu_tag}  "
          f"kv={args.kv_cache_dtype}  eager={args.enforce_eager}  "
          f"max_num_seqs={args.max_num_seqs}  util={args.util}")
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    think_empty_suffix = tok.encode(
        "<think>\n\n</think>\n\nFinal answer:", add_special_tokens=False
    )
    lp_prompts = [
        {"prompt_token_ids": list(format_prompt_ids(tok, p)) + list(think_empty_suffix)}
        for p in PROMPTS
    ]
    lp_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP_LOGPROBS)

    print(f"[diag] generating logit fingerprint for {len(PROMPTS)} prompts ...")
    lp_out = llm.generate(lp_prompts, lp_params)

    fingerprint = []
    for o in lp_out:
        lp = o.outputs[0].logprobs[0]
        fingerprint.append(sorted(
            [(int(k), round(float(v.logprob), 6)) for k, v in lp.items()],
            key=lambda kv: -kv[1]
        ))

    result = {
        "gpu_tag":        args.gpu_tag,
        "model":          args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "util":           args.util,
        "enforce_eager":  args.enforce_eager,
        "max_num_seqs":   args.max_num_seqs,
        "n_prompts":      len(PROMPTS),
        "logit_fingerprint": fingerprint,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[diag] wrote {args.out}")


def cmd_compare(args):
    results = []
    for path in args.files:
        d = json.loads(Path(path).read_text())
        results.append((path, d))

    (path_a, da), (path_b, db) = results
    fa, fb = da["logit_fingerprint"], db["logit_fingerprint"]

    if len(fa) != len(fb):
        sys.exit(f"fingerprint length mismatch: {len(fa)} vs {len(fb)}")

    top1_match, deltas = 0, []
    for ra, rb in zip(fa, fb):
        if ra[0][0] == rb[0][0]:
            top1_match += 1
        db_map = dict(rb)
        deltas += [abs(v - db_map[k]) for k, v in ra if k in db_map]

    print(f"\n  A : {path_a}")
    print(f"      gpu={da['gpu_tag']}  kv={da['kv_cache_dtype']}  "
          f"eager={da['enforce_eager']}  max_num_seqs={da['max_num_seqs']}")
    print(f"  B : {path_b}")
    print(f"      gpu={db['gpu_tag']}  kv={db['kv_cache_dtype']}  "
          f"eager={db['enforce_eager']}  max_num_seqs={db['max_num_seqs']}")
    print(f"\n  top-1 match : {top1_match}/{len(fa)}")
    print(f"  max  |Δ|    : {max(deltas):.2e}")
    print(f"  median |Δ|  : {statistics.median(deltas):.2e}")
    print(f"  p95  |Δ|    : {sorted(deltas)[int(0.95*len(deltas))]:.2e}")
    print(f"  n comparisons : {len(deltas)}")

    return max(deltas), statistics.median(deltas)


# ---------------------------------------------------------------------------
# MCQ track: prep-d5 (CPU only)
# ---------------------------------------------------------------------------

def cmd_prep_d5(args):
    """Stratified sample of D5_ITEM_COUNT MMLU-Pro items, seed-fixed."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "datasets library not found.\n"
            "Install with: pip install datasets\n"
            "(required for MMLU-Pro — not needed for D1–D4)"
        )

    print("[prep-d5] loading TIGER-Lab/MMLU-Pro test split ...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    print(f"[prep-d5] loaded {len(ds)} items across categories")

    # Collect indices per category
    by_cat: dict[str, list[int]] = {}
    for i, row in enumerate(ds):
        by_cat.setdefault(row["category"], []).append(i)

    cats = sorted(by_cat.keys())
    n_cats = len(cats)
    rng = random.Random(D5_SEED)

    # Stratified: distribute D5_ITEM_COUNT as evenly as possible
    base = D5_ITEM_COUNT // n_cats
    extra = D5_ITEM_COUNT - base * n_cats
    selected: list[int] = []
    for i, cat in enumerate(cats):
        n = base + (1 if i < extra else 0)
        pool = by_cat[cat]
        selected.extend(rng.sample(pool, min(n, len(pool))))

    rng.shuffle(selected)
    selected = selected[:D5_ITEM_COUNT]  # guard against pool < n cases

    items = []
    for idx in selected:
        row = ds[idx]
        ans_letter = row["answer"]  # e.g. "A"
        ans_idx = OPTION_LETTERS.index(ans_letter)
        items.append({
            "idx":          idx,
            "question_id":  str(row.get("question_id", idx)),
            "category":     row["category"],
            "question":     row["question"],
            "options":      list(row["options"]),
            "answer":       ans_letter,
            "answer_index": ans_idx,
        })

    out = {
        "source":   "TIGER-Lab/MMLU-Pro",
        "split":    "test",
        "seed":     D5_SEED,
        "n":        len(items),
        "items":    items,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[prep-d5] saved {len(items)} items to {args.out}")

    from collections import Counter
    cats_count = Counter(it["category"] for it in items)
    print(f"\n  category breakdown ({n_cats} categories):")
    for cat in sorted(cats_count):
        print(f"    {cats_count[cat]:3d}  {cat}")
    opts_count = Counter(len(it["options"]) for it in items)
    print(f"\n  option counts: {dict(sorted(opts_count.items()))}")


# ---------------------------------------------------------------------------
# MCQ track: gen-prefixes (GPU, run once on reference GPU)
# ---------------------------------------------------------------------------

def cmd_gen_prefixes(args):
    """Generate reference BF16 thinking traces; save prefix token IDs at D5_BUDGETS.

    Must run on ONE GPU (the reference card, typically 4090).
    Uses temperature=0.0 for determinism.  min_tokens=B_MAX forces every item
    to reach the largest checkpoint; items that can't are marked None and
    skipped by run-d5 / compare-d5.

    Incremental checkpointing: results are flushed to a JSONL file after each
    batch so at most one batch is lost on crash.  A config hash in the header
    prevents mixing items generated under different settings — the same failure
    mode that caused the D5 data-contamination incident.
    """
    import hashlib, threading, time
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    items_data = json.loads(Path(args.items).read_text())
    items = items_data["items"]
    budgets = [int(b) for b in args.budgets.split(",")]
    print(f"[gen-prefixes] {len(items)} items, budgets={budgets}, gpu={args.gpu_tag}")

    # Load tokenizer before the model to compute context budget up front.
    print(f"[gen-prefixes] loading tokenizer for {args.model} ...")
    tok_pre = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    answer_cue_ids_pre = tok_pre.encode("\n</think>\n\nThe answer is **", add_special_tokens=False)
    prompt_ids_pre = [build_mcq_prompt_ids(tok_pre, it) for it in items]

    # Derive max_model_len from actual prompt data — never hard-code.
    max_prompt_len = max(len(ids) for ids in prompt_ids_pre)
    cue_len = len(answer_cue_ids_pre)
    MARGIN = 64
    max_model_len = max_prompt_len + B_MAX + cue_len + MARGIN
    print(
        f"[gen-prefixes] max_prompt_len={max_prompt_len}  B_MAX={B_MAX}  "
        f"cue_len={cue_len}  margin={MARGIN}  → max_model_len={max_model_len}"
    )

    # Pre-validate: abort if any item would silently overflow.
    violations = [
        (i, items[i]["question_id"], len(ids), len(ids) + B_MAX + cue_len)
        for i, ids in enumerate(prompt_ids_pre)
        if len(ids) + B_MAX + cue_len > max_model_len
    ]
    if violations:
        print(f"[gen-prefixes] FATAL: {len(violations)} item(s) exceed max_model_len={max_model_len}:")
        for idx, qid, plen, total in violations[:10]:
            print(f"  item[{idx}] qid={qid} prompt_len={plen} total_needed={total}")
        sys.exit(1)
    print(f"[gen-prefixes] pre-validation PASS — all {len(items)} items fit in {max_model_len} tokens")

    # Config hash — changes to any generation-determining parameter invalidate the
    # checkpoint and require a fresh run.  This prevents mixing items from
    # different max_model_len / budget / sampling settings.
    cfg_key = json.dumps({
        "model":          args.model,
        "max_model_len":  max_model_len,
        "budgets":        budgets,
        "max_tokens":     B_MAX,
        "min_tokens":     B_MAX,
        "temperature":    0.0,
        "stop_token_ids": [args.think_end_id],
    }, sort_keys=True)
    config_hash = hashlib.sha256(cfg_key.encode()).hexdigest()[:16]
    print(f"[gen-prefixes] config_hash={config_hash}")

    # Incremental checkpoint: <out>.jsonl
    # Line 0 is a metadata object (config_hash, run params).
    # Lines 1+ are completed prefix records, one JSON object each.
    # The final <out>.json is written only after all items are done.
    ckpt_path = Path(args.out).with_suffix(".jsonl")
    out_path  = Path(args.out)

    done_ids: set[str] = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            first = f.readline().strip()
        stored_hash = ""
        if first:
            try:
                stored_hash = json.loads(first).get("config_hash", "")
            except json.JSONDecodeError:
                pass
        if stored_hash and stored_hash != config_hash:
            sys.exit(
                f"[gen-prefixes] FATAL: checkpoint config_hash mismatch\n"
                f"  stored : {stored_hash}\n"
                f"  current: {config_hash}\n"
                f"  Different max_model_len or sampling settings detected.\n"
                f"  Delete {ckpt_path} to force a clean run."
            )
        with open(ckpt_path) as f:
            f.readline()  # skip header
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"[gen-prefixes] checkpoint found: {len(done_ids)} items already done")
    else:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "w") as f:
            f.write(json.dumps({
                "_meta":         True,
                "config_hash":   config_hash,
                "ref_gpu_tag":   args.gpu_tag,
                "model":         args.model,
                "max_model_len": max_model_len,
                "budgets":       budgets,
            }) + "\n")

    pending_items = [it for it in items if it["question_id"] not in done_ids]

    if pending_items:
        print(f"[gen-prefixes] {len(pending_items)} items to generate"
              f" ({len(done_ids)} already checkpointed)")

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            kv_cache_dtype="auto",          # BF16 — never fp8
            gpu_memory_utilization=args.util,
            max_model_len=max_model_len,
            enable_prefix_caching=False,    # 300 unique prompts → 0% hit rate
            enforce_eager=False,
        )
        tok = llm.get_tokenizer()
        answer_cue_ids = tok.encode("\n</think>\n\nThe answer is **", add_special_tokens=False)
        pending_ids = [build_mcq_prompt_ids(tok, it) for it in pending_items]

        # Batch size = KV concurrency so no sequence ever needs to be preempted.
        # Submitting more than floor(kv_tokens / max_model_len) forces immediate
        # eviction of the excess, causing repeated re-prefill (observed: 9× overhead).
        try:
            ec         = llm.llm_engine.cache_config
            kv_tokens  = ec.num_gpu_blocks * ec.block_size
            batch_size = max(1, kv_tokens // max_model_len)
            print(f"[gen-prefixes] KV={kv_tokens} tokens  "
                  f"max_model_len={max_model_len}  batch_size={batch_size}")
        except Exception as exc:
            batch_size = 3
            print(f"[gen-prefixes] KV cache query failed ({exc}), using batch_size=3")

        traj_params = SamplingParams(
            max_tokens=B_MAX,
            min_tokens=B_MAX,               # force full budget so all 3 checkpoints exist
            stop_token_ids=[args.think_end_id],  # stop after thinking block, not mid-answer
            temperature=0.0,                # deterministic reference trace
        )

        t0_total     = time.time()
        n_total_done = len(done_ids)
        monitor_stop = threading.Event()

        def _monitor():
            import subprocess
            while not monitor_stop.wait(60):
                elapsed = time.time() - t0_total
                try:
                    r = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    gpu_line = r.stdout.strip().split("\n")[0] if r.stdout.strip() else "?"
                    sm_pct, mem_mib = (
                        gpu_line.split(", ") if ", " in gpu_line else ("?", "?")
                    )
                except Exception:
                    sm_pct, mem_mib = "?", "?"
                print(
                    f"[gen-prefixes] heartbeat  elapsed={elapsed:.0f}s"
                    f"  SM={sm_pct}%  mem={mem_mib}MiB",
                    flush=True,
                )

        threading.Thread(target=_monitor, daemon=True).start()
        print(
            f"[gen-prefixes] generating {len(pending_items)} items, batch={batch_size}"
            f" (checkpoint written after each batch) ...",
            flush=True,
        )

        for batch_start in range(0, len(pending_items), batch_size):
            batch_end   = min(batch_start + batch_size, len(pending_items))
            b_items     = pending_items[batch_start:batch_end]
            b_ids       = pending_ids[batch_start:batch_end]

            t0        = time.time()
            batch_out = llm.generate(
                [{"prompt_token_ids": list(ids)} for ids in b_ids],
                traj_params,
                use_tqdm=False,
            )
            elapsed     = time.time() - t0
            gen_lens    = [len(o.outputs[0].token_ids) for o in batch_out]
            short       = sum(1 for gl in gen_lens if gl < B_MAX)
            tok_per_sec = sum(gen_lens) / elapsed if elapsed > 0 else 0

            # Build prefix records and flush to checkpoint immediately.
            batch_recs = []
            for item, p_ids, out in zip(b_items, b_ids, batch_out):
                thinking = list(out.outputs[0].token_ids)
                if thinking and thinking[-1] == args.think_end_id:
                    thinking = thinking[:-1]
                budgets_d: dict[str, list[int] | None] = {}
                for b in budgets:
                    if b > len(thinking):
                        budgets_d[str(b)] = None
                    else:
                        # prefix: prompt + thinking[:b] + answer_cue
                        budgets_d[str(b)] = list(p_ids) + thinking[:b] + list(answer_cue_ids)
                batch_recs.append({
                    "question_id":  item["question_id"],
                    "category":     item["category"],
                    "n_options":    len(item["options"]),
                    "answer":       item["answer"],
                    "answer_index": item["answer_index"],
                    "thinking_len": len(thinking),
                    "budgets":      budgets_d,
                })

            with open(ckpt_path, "a") as f:
                for rec in batch_recs:
                    f.write(json.dumps(rec) + "\n")

            n_total_done += len(b_items)
            print(
                f"[gen-prefixes] batch {batch_start+1}–{batch_end}/{len(pending_items)}"
                f"  {elapsed:.1f}s  {tok_per_sec:.0f} tok/s"
                f"  total={n_total_done}/{len(items)}"
                + (f"  WARN short={short}" if short else ""),
                flush=True,
            )

        monitor_stop.set()
    else:
        print("[gen-prefixes] all items already checkpointed — consolidating to JSON ...")

    # Consolidate JSONL → final JSON, preserving original item order.
    print(f"[gen-prefixes] consolidating {ckpt_path.name} → {out_path.name} ...")
    order = {it["question_id"]: i for i, it in enumerate(items)}
    all_prefixes: list = []
    with open(ckpt_path) as f:
        f.readline()  # skip metadata header
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_prefixes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    all_prefixes.sort(key=lambda r: order.get(r["question_id"], 999999))

    n_full = sum(
        1 for r in all_prefixes
        if all(v is not None for v in r["budgets"].values())
    )
    n_partial = sum(
        1 for r in all_prefixes
        if any(v is not None for v in r["budgets"].values())
        and not all(v is not None for v in r["budgets"].values())
    )
    n_empty = len(all_prefixes) - n_full - n_partial
    print(f"[gen-prefixes] full={n_full}  partial={n_partial}  empty={n_empty}")

    out_data = {
        "ref_gpu_tag":   args.gpu_tag,
        "model":         args.model,
        "budgets":       budgets,
        "max_model_len": max_model_len,
        "config_hash":   config_hash,
        "n_items":       len(all_prefixes),
        "items":         all_prefixes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"[gen-prefixes] wrote {out_path}  (checkpoint kept at {ckpt_path})")


# ---------------------------------------------------------------------------
# MCQ track: run-d5 (GPU)
# ---------------------------------------------------------------------------

def cmd_run_d5(args):
    """Load saved prefix token IDs; compute A–J option logits (BF16 always)."""
    from vllm import LLM, SamplingParams

    prefix_data = json.loads(Path(args.prefixes).read_text())
    items = prefix_data["items"]

    # Compute max_model_len from the actual saved prefix lengths (+ 1 answer token).
    all_ids = [
        ids
        for item in items
        for ids in item["budgets"].values()
        if ids is not None
    ]
    if not all_ids:
        sys.exit("[run-d5] no valid prefix IDs found in prefixes file")
    max_prefix_len = max(len(ids) for ids in all_ids)
    max_model_len = max_prefix_len + 1   # only 1 answer token generated
    print(f"[run-d5] max_prefix_len={max_prefix_len}  → max_model_len={max_model_len}")

    # Pre-validate: confirm all prefixes fit.
    overflow = [len(ids) for ids in all_ids if len(ids) >= max_model_len]
    if overflow:
        sys.exit(f"[run-d5] FATAL: {len(overflow)} prefix(es) >= max_model_len={max_model_len}")

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        gpu_memory_utilization=args.util,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        max_logprobs=200,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    print(f"[run-d5] gpu={args.gpu_tag}  eager={args.enforce_eager}  "
          f"max_num_seqs={args.max_num_seqs}  util={args.util}")
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    # Verify option token IDs have not shifted
    for i, letter in enumerate(OPTION_LETTERS):
        actual = tok.encode(letter, add_special_tokens=False)
        if actual != [OPTION_IDS[i]]:
            sys.exit(f"Option ID mismatch for {letter}: expected [{OPTION_IDS[i]}], got {actual}")

    # Collect all (item, budget) pairs that have saved prefix IDs
    budgets = prefix_data["budgets"]
    all_prompts, all_meta = [], []
    for item in items:
        for b in budgets:
            ids = item["budgets"].get(str(b))
            if ids is None:
                continue
            all_prompts.append({"prompt_token_ids": ids})
            all_meta.append({
                "question_id":  item["question_id"],
                "category":     item["category"],
                "budget":       b,
                "answer":       item["answer"],
                "answer_index": item["answer_index"],
                "n_options":    item["n_options"],
            })

    # Request top-200 logprobs so all option letters (IDs 32-41) are always returned
    lp_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=200,
    )

    print(f"[run-d5] {len(all_prompts)} prefix+budget combinations ...")
    lp_out = llm.generate(all_prompts, lp_params)

    results = []
    for meta, o in zip(all_meta, lp_out):
        lp_map = o.outputs[0].logprobs[0]  # dict token_id → Logprob
        n = meta["n_options"]
        threshold = min(v.logprob for v in lp_map.values())
        raw_lp = [
            float(lp_map[OPTION_IDS[i]].logprob) if OPTION_IDS[i] in lp_map else None
            for i in range(n)
        ]
        n_missing = sum(x is None for x in raw_lp)
        raw_lp_for_norm = [x if x is not None else float("-inf") for x in raw_lp]
        probs = _norm_probs(raw_lp_for_norm)
        results.append({
            "question_id":  meta["question_id"],
            "category":     meta["category"],
            "budget":       meta["budget"],
            "answer":       meta["answer"],
            "answer_index": meta["answer_index"],
            "n_options":    n,
            "option_probs": probs,
            "raw_logprobs": raw_lp,
            "threshold":    threshold,
            "n_missing":    n_missing,
        })

    output = {
        "gpu_tag":      args.gpu_tag,
        "model":        args.model,
        "kv_cache_dtype": "auto",
        "util":         args.util,
        "enforce_eager": args.enforce_eager,
        "max_num_seqs": args.max_num_seqs,
        "n_results":    len(results),
        "results":      results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"[run-d5] wrote {args.out}")


# ---------------------------------------------------------------------------
# MCQ track: compare-d5 (CPU)
# ---------------------------------------------------------------------------

def _compare_d5_at_budget(pairs, budget):
    """Return per-budget comparison metrics for a list of (result_A, result_B) pairs."""
    # Legacy logprob metric
    raw_deltas = []
    for ra, rb in pairs:
        for lpa, lpb in zip(ra["raw_logprobs"], rb["raw_logprobs"]):
            if lpa > float("-inf") and lpb > float("-inf"):
                raw_deltas.append(abs(lpa - lpb))

    top1_match = sum(
        1 for ra, rb in pairs
        if (ra["option_probs"].index(max(ra["option_probs"])) ==
            rb["option_probs"].index(max(rb["option_probs"])))
    )

    ans_diffs = [
        abs(ra["option_probs"][ra["answer_index"]] - rb["option_probs"][rb["answer_index"]])
        for ra, rb in pairs
    ]
    l1_diffs = [
        sum(abs(pa - pb) for pa, pb in zip(ra["option_probs"], rb["option_probs"]))
        for ra, rb in pairs
    ]
    linf_diffs = [
        max(abs(pa - pb) for pa, pb in zip(ra["option_probs"], rb["option_probs"]))
        for ra, rb in pairs
    ]

    brier_a = [brier_score(ra["option_probs"], ra["answer_index"]) for ra, _ in pairs]
    brier_b = [brier_score(rb["option_probs"], rb["answer_index"]) for _, rb in pairs]
    paired_diff = [a - b for a, b in zip(brier_a, brier_b)]
    abs_paired = [abs(x) for x in paired_diff]
    mean_abs = statistics.mean(abs_paired)
    ci_lo, ci_hi = bootstrap_mean_ci(abs_paired)

    print(f"\n  --- budget b={budget} ({len(pairs)} items) ---")
    print(f"  top-1 match             : {top1_match}/{len(pairs)}")
    if raw_deltas:
        rd_sorted = sorted(raw_deltas)
        print(f"  max  |Δ logprob|        : {max(raw_deltas):.2e}")
        print(f"  median |Δ logprob|      : {statistics.median(raw_deltas):.2e}")
        print(f"  p95  |Δ logprob|        : {rd_sorted[int(0.95 * len(rd_sorted))]:.2e}")
    ans_sorted = sorted(ans_diffs)
    print(f"  ans prob diff mean/p95  : {statistics.mean(ans_diffs):.5f} / "
          f"{ans_sorted[int(0.95 * len(ans_sorted))]:.5f}")
    print(f"  L1  prob diff (mean)    : {statistics.mean(l1_diffs):.5f}")
    print(f"  L∞  prob diff (mean)    : {statistics.mean(linf_diffs):.5f}")
    print(f"  paired Brier diff mean  : {statistics.mean(paired_diff):+.5f}")
    print(f"  |paired Brier| mean     : {mean_abs:.5f}  "
          f"95% CI [{ci_lo:.5f}, {ci_hi:.5f}]")

    verdict = "PASS" if (mean_abs < 0.005 and ci_hi < 0.005) else "FAIL"
    print(f"  VERDICT                 : {verdict}  "
          f"(threshold: mean<0.005 and CI.hi<0.005)")
    return mean_abs, ci_hi


def cmd_compare_d5(args):
    a_data = json.loads(Path(args.files[0]).read_text())
    b_data = json.loads(Path(args.files[1]).read_text())

    def index(data):
        return {(r["question_id"], r["budget"]): r for r in data["results"]}

    a_idx = index(a_data)
    b_idx = index(b_data)
    common = set(a_idx.keys()) & set(b_idx.keys())

    print(f"\n  A : {args.files[0]}")
    print(f"      gpu={a_data['gpu_tag']}  eager={a_data['enforce_eager']}  "
          f"max_num_seqs={a_data['max_num_seqs']}")
    print(f"  B : {args.files[1]}")
    print(f"      gpu={b_data['gpu_tag']}  eager={b_data['enforce_eager']}  "
          f"max_num_seqs={b_data['max_num_seqs']}")
    print(f"\n  common (question_id, budget) pairs: {len(common)}")

    any_fail = False
    for b in D5_BUDGETS:
        pairs = [(a_idx[k], b_idx[k]) for k in common if k[1] == b]
        if not pairs:
            print(f"\n  --- budget b={b}: no common pairs, skipping ---")
            continue
        mean_abs, ci_hi = _compare_d5_at_budget(pairs, b)
        if mean_abs >= 0.005 or ci_hi >= 0.005:
            any_fail = True

    print()
    if any_fail:
        print("  OVERALL: FAIL — stop and investigate before proceeding to [3].")
    else:
        print("  OVERALL: PASS — BF16 logits are stable at all budget levels.")


# ---------------------------------------------------------------------------
# KV memory report
# ---------------------------------------------------------------------------

def cmd_mem_report(args):
    """Report KV cache capacity for a single BF16 config (run separately per config).

    Config codes:
      a  BF16 + CUDA graph + util=0.85  (baseline)
      b  BF16 + eager    + util=0.92
      c  analytical: fixed KV bytes = (a)'s KV budget applied to each GPU
    """
    if args.config == "c":
        print("\n  (c) Fixed absolute KV memory — analytical")
        print("  Strategy: equalise KV token budget across GPUs by setting util such")
        print("  that (model_bytes + target_kv_bytes) / total_vram == util.")
        print("  In practice all 3 cards have identical 24 GB VRAM and the same")
        print("  16 GB BF16 model, so (a) already produces equal KV budgets.")
        print("  The Xorg asymmetry (15 MiB) is < 0.1% of VRAM and negligible.")
        print("  To hard-pin the budget: read num_gpu_blocks from config (a),")
        print("  then compute target_kv_bytes = num_gpu_blocks * block_size * kv_per_token,")
        print("  and set util = (weight_bytes + target_kv_bytes) / total_vram.")
        return

    cfg_map = {
        "a": dict(enforce_eager=False, gpu_memory_utilization=0.85, label="BF16+CUDA-graph+util=0.85"),
        "b": dict(enforce_eager=True,  gpu_memory_utilization=0.92, label="BF16+eager+util=0.92"),
    }
    if args.config not in cfg_map:
        sys.exit(f"--config must be a, b, or c; got '{args.config}'")

    cfg = cfg_map[args.config]
    label = cfg.pop("label")

    from vllm import LLM
    print(f"\n  config ({args.config}): {label}  gpu={args.gpu_tag}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        max_model_len=B_MAX + 600,
        enable_prefix_caching=True,
        **cfg,
    )

    ec = llm.llm_engine.cache_config
    n_blocks   = ec.num_gpu_blocks
    block_size = ec.block_size
    total_kv_toks = n_blocks * block_size
    max_conc = total_kv_toks // (B_MAX + 600)

    print(f"  GPU blocks      : {n_blocks}")
    print(f"  block_size      : {block_size} tokens")
    print(f"  total KV tokens : {total_kv_toks:,}")
    print(f"  max concurrency : {max_conc}  (at {B_MAX}+600 tok context)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Logit fingerprint diagnostic (D1–D5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- run (D1–D4 free-generation track) ---------------------------------
    rp = sub.add_parser("run")
    rp.add_argument("--model",          default=DEFAULT_MODEL)
    rp.add_argument("--gpu-tag",        default="gpu0")
    rp.add_argument("--kv-cache-dtype", default="auto",
                    help="auto (BF16, default) or fp8_e5m2 (D3 only — forbidden for primary runs)")
    rp.add_argument("--util",           type=float, default=0.85)
    rp.add_argument("--enforce-eager",  action="store_true")
    rp.add_argument("--max-num-seqs",   type=int, default=None)
    rp.add_argument("--out",            required=True)

    # ---- compare (D1–D4) ---------------------------------------------------
    cp = sub.add_parser("compare")
    cp.add_argument("files", nargs=2, metavar="JSON")

    # ---- prep-d5 (CPU) -----------------------------------------------------
    pp = sub.add_parser("prep-d5")
    pp.add_argument("--out", default="data/d5_items.json")

    # ---- gen-prefixes (GPU, once) ------------------------------------------
    gp = sub.add_parser("gen-prefixes")
    gp.add_argument("--model",   default=DEFAULT_MODEL)
    gp.add_argument("--gpu-tag", default="4090")
    gp.add_argument("--items",   default="data/d5_items.json")
    gp.add_argument("--out",     default="data/d5_prefixes.json")
    gp.add_argument("--util",    type=float, default=0.85)
    gp.add_argument("--budgets", default="256,2048,8192",
                    help="comma-separated budget checkpoints (default: D5 original 3)")
    gp.add_argument("--think-end-id", type=int, default=151668,
                    help="token ID of </think> (default: 151668 = Qwen3)")

    # ---- run-d5 (GPU) ------------------------------------------------------
    rdp = sub.add_parser("run-d5")
    rdp.add_argument("--model",         default=DEFAULT_MODEL)
    rdp.add_argument("--gpu-tag",       default="gpu0")
    rdp.add_argument("--prefixes",      default="data/d5_prefixes.json")
    rdp.add_argument("--util",          type=float, default=0.85)
    rdp.add_argument("--enforce-eager", action="store_true")
    rdp.add_argument("--max-num-seqs",  type=int, default=None)
    rdp.add_argument("--out",           required=True)

    # ---- compare-d5 (CPU) --------------------------------------------------
    cd = sub.add_parser("compare-d5")
    cd.add_argument("files", nargs=2, metavar="JSON")

    # ---- mem-report (GPU) --------------------------------------------------
    mp = sub.add_parser("mem-report")
    mp.add_argument("--model",   default=DEFAULT_MODEL)
    mp.add_argument("--gpu-tag", default="gpu0")
    mp.add_argument("--config",  required=True, choices=["a", "b", "c"],
                    help="a=BF16+CUDA-graph+0.85  b=BF16+eager+0.92  c=analytical")

    args = ap.parse_args()

    dispatch = {
        "run":          cmd_run,
        "compare":      cmd_compare,
        "prep-d5":      cmd_prep_d5,
        "gen-prefixes": cmd_gen_prefixes,
        "run-d5":       cmd_run_d5,
        "compare-d5":   cmd_compare_d5,
        "mem-report":   cmd_mem_report,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
