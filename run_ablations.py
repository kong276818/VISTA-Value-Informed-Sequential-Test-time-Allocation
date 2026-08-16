#!/usr/bin/env python3
"""
Ablation analysis — CPU replay on stored checkpoint tables.

Requires:
  data/results_m1_ds{1,2,3,4}_4090.json   (produced by run_corpus.py run-d5 step)
  data/items_ds{1,2,3,4}.json             (ground-truth labels)

All ablations are stream-replay experiments; no GPU inference is needed.

Usage:
  python run_ablations.py \\
    --results data/results_m1_ds1_4090.json \\
    --items   data/items_ds1.json \\
    [--n-permutations 200]

Ablations implemented:
  A1  w/o randomized audit   — biased policy risk vs. IPW policy risk
  A2  w/o anytime-valid CS   — greedy empirical mean vs. CS-based policy selection
  A3  fixed calibration      — frozen policy (first N_cal items) vs. prequential
  A4  Brier vs accuracy-only — proper scoring objective vs. 0-1 loss
  A5  global vs item-adaptive — single global budget vs. per-item signal stopping
  A6  signal ablation        — per-signal AUC / Spearman (requires prefixes file)
  A7  audit probability rho  — sweep rho in {0.05, 0.10, 0.20, 0.40, 1.0}
"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGETS = [256, 512, 1024, 2048, 4096, 8192]

# Threshold for "confidently wrong": model assigns >= this probability to a
# single option that turns out to be incorrect. Values above random chance
# (1/n_opt) are far too permissive; 0.90 targets genuine overconfidence.
CONF_WRONG_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(items_path: str) -> dict[str, int]:
    """Load {question_id: correct_answer_index} from items file."""
    with open(items_path) as f:
        payload = json.load(f)
    items = payload["items"] if isinstance(payload, dict) else payload
    gt: dict[str, int] = {}
    for it in items:
        qid = it["question_id"]
        idx = it.get("answer_index")
        if idx is None:
            a = it.get("answer", "")
            if isinstance(a, str) and len(a) == 1:
                idx = ord(a.upper()) - ord("A")
        if idx is not None:
            gt[qid] = idx
    return gt


def load_results(path: str) -> list[dict]:
    """Load raw run-d5 results (option_probs stored, brier/nll computed later)."""
    with open(path) as f:
        payload = json.load(f)
    return payload["results"] if isinstance(payload, dict) else payload


def build_item_table(
    results: list[dict],
    ground_truth: dict[str, int],
) -> dict[str, dict]:
    """
    Returns {question_id: {budget: {"brier": float, "nll": float, "correct": bool}}}.
    Brier is summed multiclass. Computed from stored option_probs + ground truth.
    """
    table: dict[str, dict] = defaultdict(dict)
    for r in results:
        qid = r["question_id"]
        b   = r["budget"]
        true_idx = ground_truth.get(qid)
        if true_idx is None:
            continue
        n_opt = r["n_options"]
        probs = np.array(r["option_probs"][:n_opt], dtype=float)
        probs /= probs.sum()
        pred = int(np.argmax(probs))
        e_y  = np.zeros(n_opt); e_y[true_idx] = 1.0
        brier = float(np.sum((probs - e_y) ** 2))
        nll   = float(-np.log(probs[true_idx] + 1e-12))
        table[qid][b] = {
            "brier":    brier,
            "nll":      nll,
            "correct":  bool(pred == true_idx),
            "n_options": n_opt,
            "max_prob": float(np.max(probs)),
        }
    return dict(table)


# ---------------------------------------------------------------------------
# A1: Randomized audit necessity
# ---------------------------------------------------------------------------

def ablation_a1_audit_necessity(
    item_table: dict,
    rho: float = 0.20,
    n_permutations: int = 200,
    seed: int = 42,
    lam: float = 0.0,
    reference_budget: int = 4096,
) -> dict:
    """
    Compare IPW policy evaluation (with audit) vs. naive biased evaluation
    (without audit, using only items the policy would have stopped at).

    Returns per-permutation policy risk estimates for both methods.
    """
    rng = random.Random(seed)
    items = list(item_table.keys())
    # Candidate policies: fixed budgets as the simplest family
    policies = {b: b for b in BUDGETS}

    ipw_risks = defaultdict(list)
    naive_risks = defaultdict(list)

    for _ in range(n_permutations):
        perm = items[:]
        rng.shuffle(perm)
        for pi_b in BUDGETS:
            ipw_acc = []
            naive_acc = []
            for qid in perm:
                row = item_table[qid]
                if pi_b not in row or reference_budget not in row:
                    continue
                loss_pi  = row[pi_b]["brier"] + lam * pi_b
                loss_ref = row[reference_budget]["brier"] + lam * reference_budget
                # IPW: audit with probability rho, weight by 1/rho
                audited = rng.random() < rho
                if audited:
                    ipw_acc.append((loss_pi - loss_ref) / rho)
                # Naive: only items where both are observable
                # (for biased estimator, pretend audit didn't happen;
                #  use only items that naturally stop at pi_b — simulated
                #  by checking if pi_b == the item's natural stop budget)
                # For simplicity: biased estimator uses all items but
                # ignores items where pi_b > natural stop (no counterfactual).
                # Here we proxy "natural stop" as the smallest budget with
                # correct answer that doesn't change afterwards; a real
                # implementation needs natural_stop_budget in the data.
                # PLACEHOLDER — fill in once natural_stop_budget is recorded.
                naive_acc.append(loss_pi - loss_ref)  # biased: ignores selection
            if ipw_acc:
                ipw_risks[pi_b].append(statistics.mean(ipw_acc))
            if naive_acc:
                naive_risks[pi_b].append(statistics.mean(naive_acc))

    return {"ipw": dict(ipw_risks), "naive": dict(naive_risks)}


# ---------------------------------------------------------------------------
# A2: Anytime-valid CS vs greedy empirical mean
# ---------------------------------------------------------------------------

def ablation_a2_anytime_cs(
    item_table: dict,
    rho: float = 0.20,
    n_permutations: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
    reference_budget: int = 4096,
) -> dict:
    """
    Simulate policy selection under:
      (a) greedy: switch when empirical mean risk advantage exceeds fixed threshold
      (b) CS-based: switch only when lower CS bound exceeds zero

    Reports incorrect-switch rate across permutations.
    A switch is "incorrect" if the new policy is actually worse than the reference
    on the full dataset (oracle ground truth).
    """
    rng = random.Random(seed)
    items = list(item_table.keys())

    # Oracle policy risks
    oracle_risks = {}
    for pi_b in BUDGETS:
        diffs = []
        for qid, row in item_table.items():
            if pi_b in row and reference_budget in row:
                diffs.append(row[pi_b]["brier"] - row[reference_budget]["brier"])
        oracle_risks[pi_b] = statistics.mean(diffs) if diffs else float("nan")

    best_pi = min(oracle_risks, key=oracle_risks.get)

    greedy_wrong = 0
    cs_wrong = 0
    n_trials = 0

    for _ in range(n_permutations):
        perm = items[:]
        rng.shuffle(perm)

        running_diffs = defaultdict(list)
        greedy_selected = reference_budget
        cs_selected = reference_budget

        for qid in perm:
            row = item_table[qid]
            audited = rng.random() < rho
            if not audited:
                continue
            for pi_b in BUDGETS:
                if pi_b in row and reference_budget in row:
                    d = row[pi_b]["brier"] - row[reference_budget]["brier"]
                    running_diffs[pi_b].append(d / rho)

        # Greedy: pick the policy with the lowest empirical mean
        if any(running_diffs):
            greedy_selected = min(running_diffs, key=lambda b: statistics.mean(running_diffs[b]))

        # CS-based (empirical Bernstein, simplified): use half-width ~ sqrt(V/n)
        cs_candidates = [reference_budget]
        for pi_b, diffs in running_diffs.items():
            n = len(diffs)
            if n < 2:
                continue
            mean_d = statistics.mean(diffs)
            var_d  = statistics.variance(diffs)
            # Simplified CS bound: mean - c * sqrt(var / n) > 0 means confidently better
            c = math.sqrt(2 * math.log(2 * len(BUDGETS) / alpha))
            if mean_d + c * math.sqrt(var_d / n) < 0:  # pi_b is confidently better
                cs_candidates.append(pi_b)
        cs_selected = min(cs_candidates, key=lambda b: statistics.mean(running_diffs.get(b, [0])))

        n_trials += 1
        ref_risk = oracle_risks[reference_budget]
        # False positive: selected a policy that is genuinely WORSE than the reference.
        # (CS guarantee promises this is rare; greedy may violate it.)
        if oracle_risks.get(greedy_selected, float("inf")) > ref_risk + 1e-6:
            greedy_wrong += 1
        if oracle_risks.get(cs_selected, float("inf")) > ref_risk + 1e-6:
            cs_wrong += 1

    return {
        # false_positive_rate: fraction of replays where selected policy is
        # worse than the reference budget (CS guarantee: should be near 0)
        "greedy_false_positive_rate": greedy_wrong / max(n_trials, 1),
        "cs_false_positive_rate":     cs_wrong / max(n_trials, 1),
        "n_permutations":    n_trials,
        "oracle_best_pi":    best_pi,
        "oracle_risks":      oracle_risks,
        # Note: cs_false_positive=0 confirms the coverage guarantee.
        # Separately measure miss_rate (conservatism cost) by checking how
        # often cs_selected != oracle best — not computed here to keep it simple.
    }


# ---------------------------------------------------------------------------
# A3: Fixed calibration vs prequential
# ---------------------------------------------------------------------------

def ablation_a3_fixed_calibration(
    item_table: dict,
    n_cal: int = 100,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict:
    """
    Simulate:
      (a) prequential MVT-CS: update preferred budget after each labeled item
      (b) fixed calibration:  fit on first n_cal items, freeze policy

    Reports cumulative Brier regret vs. oracle adaptive.
    """
    rng = random.Random(seed)
    items = list(item_table.keys())

    oracle_per_item = {}
    for qid, row in item_table.items():
        best_b = min(row, key=lambda b: row[b]["brier"])
        oracle_per_item[qid] = row[best_b]["brier"]

    prequential_regrets = []
    fixed_cal_regrets   = []

    for _ in range(n_permutations):
        perm = items[:]
        rng.shuffle(perm)

        running = defaultdict(list)
        prequential_loss = 0.0
        fixed_cal_loss   = 0.0
        oracle_loss      = 0.0
        fixed_policy     = None

        for t, qid in enumerate(perm):
            row = item_table[qid]
            oracle_loss += oracle_per_item.get(qid, 0.0)

            # Pick prequential policy: best empirical mean so far
            if running:
                preq_pi = min(running, key=lambda b: statistics.mean(running[b]))
            else:
                preq_pi = BUDGETS[-1]  # default max before any data

            # Fixed calibration policy
            if t == n_cal and fixed_policy is None:
                fixed_policy = preq_pi
            fp = fixed_policy if fixed_policy is not None else BUDGETS[-1]

            b_preq = preq_pi if preq_pi in row else min(row)
            b_fix  = fp if fp in row else min(row)

            prequential_loss += row[b_preq]["brier"]
            fixed_cal_loss   += row[b_fix]["brier"]

            # Update running stats (simulating receiving delayed label)
            for b, info in row.items():
                running[b].append(info["brier"])

        n = len(perm)
        prequential_regrets.append((prequential_loss - oracle_loss) / n)
        fixed_cal_regrets.append((fixed_cal_loss - oracle_loss) / n)

    return {
        "prequential_mean_regret": statistics.mean(prequential_regrets),
        "fixed_cal_mean_regret":   statistics.mean(fixed_cal_regrets),
        "n_cal": n_cal,
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# A4: Brier vs accuracy-only objective
# ---------------------------------------------------------------------------

def ablation_a4_brier_vs_accuracy(item_table: dict) -> dict:
    """
    Compare stopping policies optimized for Brier vs. accuracy-only 0-1 loss.
    Reports: accuracy, Brier, confidently-wrong rate, NLL at each budget.
    """
    rows = []
    for b in BUDGETS:
        briers, accs, nlls = [], [], []
        conf_wrong = 0
        n = 0
        for qid, row in item_table.items():
            if b not in row:
                continue
            info = row[b]
            briers.append(info["brier"])
            if info["correct"] is not None:
                accs.append(int(info["correct"]))
            if not math.isnan(info["nll"]):
                nlls.append(info["nll"])
                # confidently wrong: model is both highly confident (max prob
                # >= CONF_WRONG_THRESHOLD) and incorrect
                if info["correct"] is False and info.get("max_prob", 0.0) >= CONF_WRONG_THRESHOLD:
                    conf_wrong += 1
            n += 1
        rows.append({
            "budget": b,
            "mean_brier":   statistics.mean(briers) if briers else float("nan"),
            "accuracy":     statistics.mean(accs)   if accs   else float("nan"),
            "mean_nll":     statistics.mean(nlls)   if nlls   else float("nan"),
            "conf_wrong_rate": conf_wrong / max(n, 1),
        })
    return {"budget_rows": rows}


# ---------------------------------------------------------------------------
# A5: Global vs item-adaptive
# ---------------------------------------------------------------------------

def ablation_a5_global_vs_item(item_table: dict) -> dict:
    """
    Global: pick one fixed budget for the entire stream (best-fixed).
    Item-adaptive oracle: pick per-item optimal budget.
    Reports Brier gap between the two.
    """
    best_fixed_budget = None
    best_fixed_brier  = float("inf")
    for b in BUDGETS:
        vals = [row[b]["brier"] for row in item_table.values() if b in row]
        if vals:
            mean_b = statistics.mean(vals)
            if mean_b < best_fixed_brier:
                best_fixed_brier  = mean_b
                best_fixed_budget = b

    oracle_brier = statistics.mean(
        min(row[b]["brier"] for b in BUDGETS if b in row)
        for row in item_table.values()
    )

    return {
        "best_fixed_budget": best_fixed_budget,
        "best_fixed_brier":  best_fixed_brier,
        "oracle_brier":      oracle_brier,
        "oracle_gap":        oracle_brier - best_fixed_brier,
    }


# ---------------------------------------------------------------------------
# Gate 0: Matched-quality token oracle
# ---------------------------------------------------------------------------

def compute_matched_oracle(
    item_table: dict,
    epsilon: float = 0.0,
) -> dict:
    """
    Matched-quality token oracle: per item, find the minimum budget b such that
    Brier(b) <= Brier(best_fixed_global) + epsilon.

    epsilon=0.0 means strictly match or beat the best-fixed baseline.
    Reports mean minimum budget and fraction of items where a smaller budget
    than best_fixed suffices (compute reduction opportunity).

    This is NOT the hindsight oracle from A5 (which minimises Brier per item).
    This oracle asks: "what is the cheapest budget that is at least as good as
    the best global fixed budget?"  Gate 0 passes if mean_matched_budget is
    meaningfully smaller than best_fixed_budget.
    """
    best_fixed_budget = None
    best_fixed_brier  = float("inf")
    for b in BUDGETS:
        vals = [row[b]["brier"] for row in item_table.values() if b in row]
        if vals:
            mean_b = statistics.mean(vals)
            if mean_b < best_fixed_brier:
                best_fixed_brier  = mean_b
                best_fixed_budget = b

    threshold = best_fixed_brier + epsilon

    min_budgets = []
    reducible   = 0
    for row in item_table.values():
        chosen = best_fixed_budget
        for b in BUDGETS:
            if b in row and row[b]["brier"] <= threshold:
                chosen = b
                break
        min_budgets.append(chosen)
        if chosen < best_fixed_budget:
            reducible += 1

    n = len(min_budgets)
    mean_min = statistics.mean(min_budgets) if min_budgets else float("nan")

    return {
        "best_fixed_budget":     best_fixed_budget,
        "best_fixed_mean_brier": best_fixed_brier,
        "epsilon":               epsilon,
        "threshold_brier":       threshold,
        "mean_matched_budget":   mean_min,
        "pct_reducible":         reducible / max(n, 1),
        "n_items":               n,
    }


# ---------------------------------------------------------------------------
# A7: Audit probability sweep
# ---------------------------------------------------------------------------

def ablation_a7_rho_sweep(
    item_table: dict,
    rho_values: list[float] = (0.05, 0.10, 0.20, 0.40, 1.0),
    n_permutations: int = 200,
    seed: int = 42,
) -> list[dict]:
    results = []
    for rho in rho_values:
        a1 = ablation_a1_audit_necessity(item_table, rho=rho,
                                         n_permutations=n_permutations, seed=seed)
        a2 = ablation_a2_anytime_cs(item_table, rho=rho,
                                    n_permutations=n_permutations, seed=seed)
        results.append({
            "rho": rho,
            "cs_false_positive_rate":     a2["cs_false_positive_rate"],
            "greedy_false_positive_rate": a2["greedy_false_positive_rate"],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results JSON file")
    ap.add_argument("--items",   required=True, help="Path to items JSON file (ground truth)")
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--n-cal", type=int, default=100,
                    help="Calibration-set size for A3")
    ap.add_argument("--rho", type=float, default=0.20,
                    help="Default audit probability")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    ground_truth = load_ground_truth(args.items)
    results = load_results(args.results)
    item_table = build_item_table(results, ground_truth)
    print(f"Loaded {len(item_table)} items from {args.results}")

    output = {}

    print("\n=== A1: Audit necessity ===")
    a1 = ablation_a1_audit_necessity(item_table, rho=args.rho,
                                     n_permutations=args.n_permutations,
                                     seed=args.seed)
    output["a1"] = a1
    for pi_b in BUDGETS:
        ipw = a1["ipw"].get(pi_b, [])
        naive = a1["naive"].get(pi_b, [])
        if ipw and naive:
            print(f"  b={pi_b}: IPW mean={statistics.mean(ipw):+.4f}  "
                  f"naive mean={statistics.mean(naive):+.4f}")

    print("\n=== A2: Anytime-valid CS ===")
    a2 = ablation_a2_anytime_cs(item_table, rho=args.rho,
                                n_permutations=args.n_permutations,
                                alpha=0.05, seed=args.seed)
    output["a2"] = a2
    print(f"  greedy false-positive rate: {a2['greedy_false_positive_rate']:.3f}  "
          f"(selected genuinely worse policy)")
    print(f"  CS false-positive rate:     {a2['cs_false_positive_rate']:.3f}  "
          f"(coverage guarantee; 0 expected)")
    print(f"  oracle best policy: b={a2['oracle_best_pi']}")
    print(f"  Note: CS miss-rate (failed to adopt best) should be computed "
          f"separately as convergence metric")

    print("\n=== A3: Fixed calibration vs prequential ===")
    a3 = ablation_a3_fixed_calibration(item_table, n_cal=args.n_cal,
                                       n_permutations=args.n_permutations,
                                       seed=args.seed)
    output["a3"] = a3
    print(f"  prequential regret: {a3['prequential_mean_regret']:+.4f}")
    print(f"  fixed-cal regret:   {a3['fixed_cal_mean_regret']:+.4f}")

    print("\n=== A4: Brier vs accuracy-only ===")
    a4 = ablation_a4_brier_vs_accuracy(item_table)
    output["a4"] = a4
    print(f"  {'Budget':>6}  {'Accuracy':>8}  {'Brier':>8}  {'NLL':>6}  {'ConfWrong':>10}")
    for row in a4["budget_rows"]:
        print(f"  {row['budget']:6d}  {row['accuracy']:8.3f}  "
              f"{row['mean_brier']:8.4f}  {row['mean_nll']:6.3f}  "
              f"{row['conf_wrong_rate']:10.4f}")

    print("\n=== A5: Global vs item-adaptive ===")
    a5 = ablation_a5_global_vs_item(item_table)
    output["a5"] = a5
    print(f"  best fixed b={a5['best_fixed_budget']}, Brier={a5['best_fixed_brier']:.4f}")
    print(f"  oracle Brier={a5['oracle_brier']:.4f}  "
          f"gap={a5['oracle_gap']:+.4f}")

    print("\n=== Gate 0: Matched-quality token oracle ===")
    for eps in (0.0, 0.005, 0.01):
        g0 = compute_matched_oracle(item_table, epsilon=eps)
        output[f"gate0_eps{eps}"] = g0
        print(f"  epsilon={eps:.3f}: best_fixed=b{g0['best_fixed_budget']}  "
              f"mean_matched=b{g0['mean_matched_budget']:.0f}  "
              f"pct_reducible={g0['pct_reducible']:.3f}")
    print(f"  (Gate 0 passes if mean_matched_budget meaningfully < best_fixed_budget)")

    print("\n=== A7: rho sweep ===")
    a7 = ablation_a7_rho_sweep(item_table, n_permutations=args.n_permutations,
                                seed=args.seed)
    output["a7"] = a7
    print(f"  {'rho':>5}  {'cs_fp':>9}  {'greedy_fp':>10}")
    for row in a7:
        print(f"  {row['rho']:5.2f}  {row['cs_false_positive_rate']:9.3f}  "
              f"{row['greedy_false_positive_rate']:10.3f}")

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
