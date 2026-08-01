#!/usr/bin/env python3
"""
M0 Step 0: capacity probe for a 3-GPU heterogeneous rig
  RTX 4090 (24GB, sm_89) x1  +  RTX 3090 (24GB, sm_86) x2

Answers three questions BEFORE any dataset is chosen:
  (A) how many concurrent 8k sequences actually fit  -> max batch
  (B) wall-clock seconds per item for a full budget-grid run
  (C) do 4090 and 3090 produce comparable logits for the same prompt

Run once per GPU:
    CUDA_VISIBLE_DEVICES=0 python m0_capacity_probe.py --gpu-tag 4090
    CUDA_VISIBLE_DEVICES=1 python m0_capacity_probe.py --gpu-tag 3090a
    CUDA_VISIBLE_DEVICES=2 python m0_capacity_probe.py --gpu-tag 3090b

Then:
    python m0_capacity_probe.py --compare out/probe_*.json
"""

import argparse, json, os, time, glob, statistics, sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration -- keep every knob here so the paper can cite exact settings.
# ----------------------------------------------------------------------------

BUDGET_GRID = [256, 512, 1024, 2048, 4096, 8192]   # reasoning-token checkpoints
B_MAX = BUDGET_GRID[-1]
PROBE_MAX_TOKENS = 64          # forced final-answer probe after </think>
N_TIMING_ITEMS = 24            # enough to estimate throughput, cheap to run
N_LOGIT_ITEMS = 50             # cross-GPU numerical agreement check
TOP_LOGPROBS = 20              # what we persist per checkpoint

DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Deliberately fixed. Any change invalidates cached generations.
SAMPLING = dict(temperature=0.6, top_p=0.95, seed=1234)

# Single-token IDs verified against Qwen3 tokenizer_config.json.
THINK_START_ID = 151667   # <think>
THINK_END_ID   = 151668   # </think>


# ----------------------------------------------------------------------------
# Prompts -- a fixed synthetic set so the probe never depends on a dataset
# being downloaded. Real datasets come in M0 step 2.
# ----------------------------------------------------------------------------

def make_prompts(n):
    """Prompts long enough to force genuine multi-step reasoning."""
    base = [
        "A regular hexagon has area 96. Find the area of the triangle formed "
        "by three alternating vertices. Show all steps.",
        "Let f(x)=x^4-6x^2+8. Find every real root and prove none were missed.",
        "Seven distinct integers sum to 0 and their product is 5040. "
        "Determine all possible multisets, justifying exhaustiveness.",
        "A fair coin is flipped until two consecutive heads appear. "
        "Compute the expected number of flips, deriving the recurrence.",
    ]
    return [base[i % len(base)] for i in range(n)]


def format_prompt_ids(tok, problem):
    """Token IDs for Qwen3 chat-formatted prompt (assistant generation turn open).

    Applying the chat template rather than passing raw strings ensures the model
    reliably enters thinking mode and that the prompt encoding is stable across
    all three probe phases (no string round-trips).
    """
    msgs = [{"role": "user", "content": problem}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


# ----------------------------------------------------------------------------
# Memory math -- printed so you can sanity-check vLLM's own allocation
# ----------------------------------------------------------------------------

def kv_bytes_per_token(cfg, kv_dtype_bytes):
    """2 (K and V) * layers * kv_heads * head_dim * bytes."""
    layers = cfg.num_hidden_layers
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return 2 * layers * kv_heads * head_dim * kv_dtype_bytes


def report_memory_plan(model, kv_cache_dtype, util):
    from transformers import AutoConfig
    import torch

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    kv_b = 1 if kv_cache_dtype.startswith("fp8") else 2
    per_tok = kv_bytes_per_token(cfg, kv_b)
    per_seq = per_tok * B_MAX

    # Estimate BF16 weight size from architecture (tie_word_embeddings=False for Qwen3).
    hid = cfg.hidden_size
    n_params = (
        cfg.vocab_size * hid * 2                                         # embeddings (untied)
        + cfg.num_hidden_layers * (
            cfg.num_attention_heads * cfg.head_dim * hid                 # Wq
            + cfg.num_key_value_heads * cfg.head_dim * hid * 2          # Wk, Wv
            + hid * hid                                                  # Wo
            + cfg.intermediate_size * hid * 3                           # gate, up, down
            + hid * 2                                                    # pre/post-attn norms
        )
        + hid                                                            # final norm
    )
    weight_gib = n_params * 2 / 2**30   # BF16

    total = torch.cuda.get_device_properties(0).total_memory
    budget = total * util
    print("--- memory plan -------------------------------------------------")
    print(f"  device             : {torch.cuda.get_device_name(0)}")
    print(f"  total VRAM         : {total/2**30:.1f} GiB")
    print(f"  vLLM budget        : {budget/2**30:.1f} GiB (util={util})")
    print(f"  kv dtype           : {kv_cache_dtype} ({kv_b} B/elem)")
    print(f"  kv per token       : {per_tok/1024:.1f} KiB")
    print(f"  kv per {B_MAX}-tok seq  : {per_seq/2**30:.2f} GiB")
    print(f"  est. weights (BF16): {weight_gib:.1f} GiB")
    print(f"  est. concurrent seqs (8192-tok): "
          f"{(budget - weight_gib * 2**30) / per_seq:.1f}")
    print("  -> compare against vLLM's reported max concurrency below.")
    print("-----------------------------------------------------------------")
    return per_seq


# ----------------------------------------------------------------------------
# Core measurement
# ----------------------------------------------------------------------------

def run_probe(args):
    from vllm import LLM, SamplingParams

    per_seq = report_memory_plan(args.model, args.kv_cache_dtype, args.util)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.util,
        # Headroom: prompt_ids (~40) + B_MAX + suffix (~10) + PROBE_MAX_TOKENS(64).
        # Fixed prompts are ~26 tokens; extending beyond ~438-token problems would
        # require raising this limit.
        max_model_len=B_MAX + 512,
        enable_prefix_caching=True,      # <-- the whole plan depends on this
        enforce_eager=False,
        swap_space=2,
    )
    tok = llm.get_tokenizer()

    raw_prompts = make_prompts(N_TIMING_ITEMS)

    # Tokenise prompts once and pass IDs directly everywhere.
    # String round-trips (tok.decode → retokenise) shift BPE boundaries at the
    # prompt/generation junction and silently break prefix-cache hits.
    fmt_ids_list = [format_prompt_ids(tok, p) for p in raw_prompts]

    # Suffix appended after each truncated thinking trace.
    suffix_ids = tok.encode("\n</think>\n\nFinal answer:", add_special_tokens=False)

    results = {
        "gpu_tag": args.gpu_tag,
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "util": args.util,
        "budget_grid": BUDGET_GRID,
        "kv_bytes_per_seq": per_seq,
    }

    # ---- (B) throughput: full trajectory until </think> or B_MAX -----------
    # stop_token_ids ensures ids contains only the thinking trace, never
    # post-</think> answer text.  Checkpoints beyond the natural thinking
    # length are skipped by the guard below rather than being contaminated.
    traj_params = SamplingParams(
        max_tokens=B_MAX,
        stop_token_ids=[THINK_END_ID],
        **SAMPLING,
    )

    t0 = time.perf_counter()
    traj_out = llm.generate(
        [{"prompt_token_ids": ids} for ids in fmt_ids_list],
        traj_params,
    )
    t_traj = time.perf_counter() - t0

    gen_tokens = sum(len(o.outputs[0].token_ids) for o in traj_out)
    results["trajectory"] = {
        "n_items": N_TIMING_ITEMS,
        "wall_s": round(t_traj, 2),
        "generated_tokens": gen_tokens,
        "tok_per_s": round(gen_tokens / t_traj, 1),
        "s_per_item": round(t_traj / N_TIMING_ITEMS, 2),
    }

    # ---- checkpoint probes reusing the cached prefix ----------------------
    # IDs passed directly: p_ids + ids[:b] + suffix_ids.
    # The prefix p_ids + ids[:b] is identical byte-for-byte to what the
    # trajectory cached, so every probe prefill is a full KV-cache hit.
    probe_params = SamplingParams(
        max_tokens=PROBE_MAX_TOKENS, temperature=0.0,
        logprobs=TOP_LOGPROBS,
    )
    probe_prompts = []
    for p_ids, o in zip(fmt_ids_list, traj_out):
        ids = list(o.outputs[0].token_ids)
        # vLLM may include the stop token in output; strip it if present.
        if ids and ids[-1] == THINK_END_ID:
            ids = ids[:-1]
        for b in BUDGET_GRID:
            if b > len(ids):
                continue
            probe_prompts.append({
                "prompt_token_ids": list(p_ids) + ids[:b] + list(suffix_ids)
            })

    t0 = time.perf_counter()
    probe_out = llm.generate(probe_prompts, probe_params)
    t_probe = time.perf_counter() - t0

    probe_tokens = sum(len(o.outputs[0].token_ids) for o in probe_out)
    results["probes"] = {
        "n_probes": len(probe_prompts),
        "probes_per_item": round(len(probe_prompts) / N_TIMING_ITEMS, 2),
        "wall_s": round(t_probe, 2),
        "generated_tokens": probe_tokens,
        "s_per_item": round(t_probe / N_TIMING_ITEMS, 2),
    }

    total_per_item = (t_traj + t_probe) / N_TIMING_ITEMS
    results["full_grid_s_per_item"] = round(total_per_item, 2)
    results["probe_overhead_frac"] = round(t_probe / (t_traj + t_probe), 3)

    # ---- extrapolation table ---------------------------------------------
    results["extrapolation_hours_single_gpu"] = {
        f"{n}_items": round(n * total_per_item / 3600, 2)
        for n in (500, 1000, 2700, 8100)
    }

    # ---- (C) logits for cross-GPU comparison ------------------------------
    # Zero-thinking baseline: chat-formatted prompt + empty <think></think> block.
    # Token IDs are identical across all three GPUs so the logit comparison is fair.
    think_empty_suffix = tok.encode(
        "<think>\n\n</think>\n\nFinal answer:", add_special_tokens=False
    )
    lp_prompts = [
        {"prompt_token_ids": list(format_prompt_ids(tok, p)) + list(think_empty_suffix)}
        for p in make_prompts(N_LOGIT_ITEMS)
    ]
    lp_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP_LOGPROBS)
    lp_out = llm.generate(lp_prompts, lp_params)

    fingerprint = []
    for o in lp_out:
        lp = o.outputs[0].logprobs[0]
        fingerprint.append(sorted(
            [(int(k), round(float(v.logprob), 6)) for k, v in lp.items()],
            key=lambda kv: -kv[1]
        ))
    results["logit_fingerprint"] = fingerprint

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out = Path(args.outdir) / f"probe_{args.gpu_tag}.json"
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps({k: v for k, v in results.items()
                      if k != "logit_fingerprint"}, indent=2))
    print(f"\nwrote {out}")


# ----------------------------------------------------------------------------
# Cross-GPU comparison
# ----------------------------------------------------------------------------

def compare(paths):
    runs = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        runs[d["gpu_tag"]] = d

    print("=== throughput ==================================================")
    for tag, d in runs.items():
        print(f"  {tag:8s}  {d['full_grid_s_per_item']:6.2f} s/item   "
              f"traj {d['trajectory']['tok_per_s']:7.1f} tok/s   "
              f"probe overhead {d['probe_overhead_frac']:.1%}")

    agg = sum(1.0 / d["full_grid_s_per_item"] for d in runs.values())
    print(f"\n  combined rig: {1/agg:.2f} s/item wall  "
          f"({agg*3600:.0f} items/hour)")
    for n in (2700, 8100):
        print(f"    {n:5d} items -> {n/(agg*3600):.1f} h")

    print("\n=== cross-GPU logit agreement ===================================")
    tags = list(runs)
    ref = tags[0]
    for tag in tags[1:]:
        a, b = runs[ref]["logit_fingerprint"], runs[tag]["logit_fingerprint"]
        top1_match, deltas = 0, []
        for fa, fb in zip(a, b):
            if fa[0][0] == fb[0][0]:
                top1_match += 1
            db = dict(fb)
            deltas += [abs(v - db[k]) for k, v in fa if k in db]
        print(f"  {ref} vs {tag}:")
        print(f"    top-1 token agreement : {top1_match}/{len(a)}")
        print(f"    max |Δ logprob|       : {max(deltas):.2e}")
        print(f"    median |Δ logprob|    : {statistics.median(deltas):.2e}")

    print("""
  READ THIS:
    max |Δ logprob| < 1e-3  -> pool all three GPUs freely.
    otherwise               -> pin each (model x dataset) cell to ONE card
                               and state it in the paper. Do NOT split a
                               calibration cell across architectures.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu-tag", default="gpu0")
    ap.add_argument("--kv-cache-dtype", default="fp8_e5m2",
                    help="fp8_e5m2 roughly doubles concurrency; auto = bf16")
    ap.add_argument("--util", type=float, default=0.92)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--compare", nargs="*", default=None)
    args = ap.parse_args()

    if args.compare is not None:
        paths = args.compare or sorted(glob.glob("out/probe_*.json"))
        if len(paths) < 2:
            sys.exit("need at least two probe_*.json files")
        compare(paths)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
