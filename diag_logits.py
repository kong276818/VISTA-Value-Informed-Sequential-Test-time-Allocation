#!/usr/bin/env python3
"""
Logit-fingerprint diagnostic — used for D1 through D4.

Two sub-commands:
  run      Load model, generate logit fingerprint for 50 prompts, save JSON.
  compare  Load two JSON files and print max / median |Δ logprob|.

D1 (run-to-run non-determinism, same GPU):
  python diag_logits.py run --gpu-tag 4090 --out /tmp/d1a.json
  python diag_logits.py run --gpu-tag 4090 --out /tmp/d1b.json
  python diag_logits.py compare /tmp/d1a.json /tmp/d1b.json

D2 (same arch, different card):
  (run 3090a)  python diag_logits.py run --gpu-tag 3090a --out /tmp/d2a.json
  (run 3090b)  python diag_logits.py run --gpu-tag 3090b --out /tmp/d2b.json
  python diag_logits.py compare /tmp/d2a.json /tmp/d2b.json

D3 (fp8 vs BF16 KV cache, 4090):
  python diag_logits.py run --gpu-tag 4090 --kv-cache-dtype auto --out /tmp/d3.json
  python diag_logits.py compare /tmp/d1a.json /tmp/d3.json

D4 (eager mode, batch=1, 4090):
  python diag_logits.py run --gpu-tag 4090 --enforce-eager --max-num-seqs 1 --out /tmp/d4.json
  python diag_logits.py compare /tmp/d1a.json /tmp/d4.json
"""

import argparse, json, statistics, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared constants — must match m0_capacity_probe.py exactly
# ---------------------------------------------------------------------------

THINK_START_ID = 151667   # <think>
THINK_END_ID   = 151668   # </think>
TOP_LOGPROBS   = 20
DEFAULT_MODEL  = "Qwen/Qwen3-8B"

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


# ---------------------------------------------------------------------------
# Run
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


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

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
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Logit fingerprint diagnostic (D1-D4)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="generate and save logit fingerprint")
    rp.add_argument("--model",          default=DEFAULT_MODEL)
    rp.add_argument("--gpu-tag",        default="gpu0")
    rp.add_argument("--kv-cache-dtype", default="fp8_e5m2",
                    help="fp8_e5m2 (default) or auto (BF16)")
    rp.add_argument("--util",           type=float, default=0.85)
    rp.add_argument("--enforce-eager",  action="store_true",
                    help="D4: disable CUDA graphs")
    rp.add_argument("--max-num-seqs",   type=int, default=None,
                    help="D4: set to 1 to remove batching non-determinism")
    rp.add_argument("--out",            required=True,
                    help="output JSON path")

    cp = sub.add_parser("compare", help="compare two fingerprint JSON files")
    cp.add_argument("files", nargs=2, metavar="JSON")

    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
