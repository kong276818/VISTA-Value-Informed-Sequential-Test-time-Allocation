import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
VISTA Final Experiment Campaign — 20 Ablations (CPU replay).

All ablations operate on stored checkpoint tables; no GPU inference required.
Results go to results/ablations/a{01..20}_*.json.

Usage:
  cd /home/sclab/paper2
  python run_ablation_campaign.py [--ablations 1,2,3] [--cells m1_ds1,m2_ds1]
                                   [--n-perm 200] [--n-boot 1000]

Ablation index (campaign doc numbering):
  A01  IPW vs Naive (extends run_ablations A1, all cells, rho sweep)
  A02  Counterfactual Missingness
  A03  CS vs Greedy (extends run_ablations A2, all cells)
  A04  Audit Rate Sensitivity (extends run_ablations A7, all cells)
  A05  Accuracy vs Brier vs NLL objective divergence (all 8 cells)
  A06  Prefix Signal Ablation — gate1 heatmap (all cells)
  A07  Signal Combination (logistic regression / linear, cross-validated)
  A08  Global vs Item-Adaptive Policy
  A09  Matched-Quality Policy Comparison (main table candidate)
  A10  Lambda Sensitivity — Pareto frontier
  A11  Prequential vs Fixed Calibration (extends run_ablations A3)
  A12  Distribution Shift (synthetic mixed streams)
  A13  Cross-Model Transfer (M1→M2 and M2→M1 threshold transfer)
  A14  Cross-Dataset Transfer (threshold transfer across datasets)
  A15  BLOCKED — Forced Checkpoint vs Independent Generation (needs GPU)
  A16  Delayed Label Latency
  A17  Policy Family Size (CS width vs K)
  A18  Audit Schedule (fixed / decay / warm-start rho)
  A19  Bootstrap / Random-Order Robustness (1000 resamples)
  A20  Component Removal (VISTA − each component)
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent
RESULTS_DIR   = ROOT / "results" / "ablations"
TABLES_DIR    = ROOT / "tables"  / "ablations"
FIGURES_DIR   = ROOT / "figures" / "ablations"

for d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUDGETS    = [256, 512, 1024, 2048, 4096, 8192]
B_MAX      = 8192
CONF_WRONG = 0.90

ALL_CELLS = [
    ("m1", "ds1", "MMLU-Pro",      "Qwen3-8B",       "4090"),
    ("m1", "ds2", "ARC-Challenge", "Qwen3-8B",       "4090"),
    ("m1", "ds3", "MedMCQA",       "Qwen3-8B",       "4090"),
    ("m1", "ds4", "MedQA-USMLE",   "Qwen3-8B",       "4090"),
    ("m2", "ds1", "MMLU-Pro",      "DeepSeek-R1-Llama-8B", "3090a"),
    ("m2", "ds2", "ARC-Challenge", "DeepSeek-R1-Llama-8B", "3090a"),
    ("m2", "ds3", "MedMCQA",       "DeepSeek-R1-Llama-8B", "3090b"),
    ("m2", "ds4", "MedQA-USMLE",   "DeepSeek-R1-Llama-8B", "3090b"),
]

RESULTS_FILES = {
    ("m1", "ds1"): "data/results_m1_ds1_4090.json",
    ("m1", "ds2"): "data/results_m1_ds2_4090.json",
    ("m1", "ds3"): "data/results_m1_ds3_4090.json",
    ("m1", "ds4"): "data/results_m1_ds4_4090.json",
    ("m2", "ds1"): "data/results_m2_ds1_3090a.json",
    ("m2", "ds2"): "data/results_m2_ds2_3090a.json",
    ("m2", "ds3"): "data/results_m2_ds3_3090b.json",
    ("m2", "ds4"): "data/results_m2_ds4_3090b.json",
}

ITEMS_FILES = {
    "ds1": "data/items_ds1.json",
    "ds2": "data/items_ds2.json",
    "ds3": "data/items_ds3.json",
    "ds4": "data/items_ds4.json",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results_raw(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    return payload["results"] if isinstance(payload, dict) else payload


def load_ground_truth(items_path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(items_path).read_text())
    items = payload["items"] if isinstance(payload, dict) else payload
    gt = {}
    for it in items:
        qid = it["question_id"]
        idx = it.get("answer_index")
        if idx is None:
            a = it.get("answer", "")
            if isinstance(a, str) and len(a) == 1:
                idx = ord(a.upper()) - ord("A")
        if idx is not None:
            gt[qid] = int(idx)
    return gt


def build_item_table(raw_results: list[dict], ground_truth: dict[str, int]) -> dict:
    """
    item_table[qid][budget] = {
        brier, nll, correct, max_prob, n_options, option_probs
    }
    """
    table = defaultdict(dict)
    for r in raw_results:
        qid = r["question_id"]
        b   = int(r["budget"])
        true_idx = ground_truth.get(qid)
        if true_idx is None:
            continue
        n_opt = int(r["n_options"])
        probs = np.array(r["option_probs"][:n_opt], dtype=float)
        probs /= probs.sum()
        e_y  = np.zeros(n_opt); e_y[true_idx] = 1.0
        bs   = float(np.sum((probs - e_y) ** 2))
        nl   = float(-np.log(probs[true_idx] + 1e-12))
        pred = int(np.argmax(probs))
        table[qid][b] = {
            "brier":       bs,
            "nll":         nl,
            "correct":     pred == true_idx,
            "max_prob":    float(np.max(probs)),
            "option_probs": probs.tolist(),
            "n_options":   n_opt,
            "category":    r.get("category", ""),
        }
    return dict(table)


def load_cell(model_tag: str, ds_tag: str) -> dict:
    """Load and return item_table for a cell."""
    key = (model_tag, ds_tag)
    rpath = ROOT / RESULTS_FILES[key]
    ipath = ROOT / ITEMS_FILES[ds_tag]
    raw  = load_results_raw(rpath)
    gt   = load_ground_truth(ipath)
    return build_item_table(raw, gt)


def _percentile(data, q):
    s = sorted(data)
    n = len(s)
    idx = q * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] * (1 - idx + lo) + s[hi] * (idx - lo)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def brier_of_rec(r): return r["brier"]
def acc_of_rec(r):   return float(r["correct"])
def nll_of_rec(r):   return r["nll"]

def mean_metric(table: dict, b: int, fn) -> float:
    vals = [fn(row[b]) for row in table.values() if b in row]
    return statistics.mean(vals) if vals else float("nan")

def best_budget(table: dict, fn, want_min: bool = True) -> int:
    vals = {b: mean_metric(table, b, fn) for b in BUDGETS}
    vals = {b: v for b, v in vals.items() if not math.isnan(v)}
    return min(vals, key=vals.get) if want_min else max(vals, key=vals.get)


# ---------------------------------------------------------------------------
# A01 — IPW vs Naive
# ---------------------------------------------------------------------------

def a01_ipw_vs_naive(
    table: dict,
    rho_values=(0.05, 0.10, 0.20, 0.40, 1.00),
    n_perm: int = 200,
    seed: int = 42,
    reference_b: int = 4096,
    stop_tau: float = 0.85,
) -> dict:
    """
    Demonstrate necessity of IPW for unbiased policy-risk estimation.

    Scenario: a stopping policy halts inference when confidence >= stop_tau.
    Under this policy, items that stop early (before reference_b) do NOT have
    their reference_b Brier computed — it is a counterfactual.

    Estimators of E[Brier(pi_b) − Brier(reference_b)]:
      - oracle:        full-information mean over all stored checkpoints
      - naive:         mean over only naturally-continued items
                       (items where natural_stop >= reference_b); selection-biased
      - audited_naive: randomly reveal reference_b for rho-fraction of stopped items,
                       average over revealed (no reweighting); unbiased but high-var
      - ipw:           (1/n) × sum_{audited} diff / rho; unbiased, matches oracle
    """
    rng = random.Random(seed)
    items = list(table.keys())

    # Natural stop: first budget where confidence >= stop_tau
    def natural_stop(bmap):
        for b in BUDGETS:
            if b in bmap and bmap[b]["max_prob"] >= stop_tau:
                return b
        return max(b for b in BUDGETS if b in bmap)

    stop_budgets = {qid: natural_stop(row) for qid, row in table.items()}

    result_by_rho = []
    for rho in rho_values:
        # Oracle risk difference E[Brier(pi_b) - Brier(ref_b)] for each pi_b
        oracle_risk = {}
        for pi_b in BUDGETS:
            diffs = [row[pi_b]["brier"] - row[reference_b]["brier"]
                     for row in table.values()
                     if pi_b in row and reference_b in row]
            oracle_risk[pi_b] = statistics.mean(diffs) if diffs else float("nan")
        true_best = min((b for b in BUDGETS if not math.isnan(oracle_risk[b])),
                        key=oracle_risk.get)

        ipw_bias   = []
        naive_bias = []
        audited_naive_bias = []
        wrong_best_ipw   = 0
        wrong_best_naive = 0
        wrong_best_anv   = 0
        n_trials = 0

        for _ in range(n_perm):
            perm = items[:]
            rng.shuffle(perm)
            n = len(perm)

            # Accumulate per-policy estimates
            ipw_sum    = defaultdict(float)   # sum of d/rho for audited items
            naive_vals = defaultdict(list)    # diffs for naturally-continued items
            anv_vals   = defaultdict(list)    # diffs for audited items (no reweight)

            for qid in perm:
                row = table[qid]
                b_stop = stop_budgets[qid]

                # Audit decision (random, independent of item)
                audited = rng.random() < rho

                for pi_b in BUDGETS:
                    if pi_b not in row or reference_b not in row:
                        continue
                    d = row[pi_b]["brier"] - row[reference_b]["brier"]

                    # Naive: only items where reference_b is naturally observable
                    # (i.e., item ran past reference_b without stopping)
                    if b_stop >= reference_b:
                        naive_vals[pi_b].append(d)

                    # Audited naive: revealed reference_b via random audit
                    if audited:
                        anv_vals[pi_b].append(d)
                        # IPW: normalize by n (total items), not n_audited
                        ipw_sum[pi_b] += d / rho  # accumulate; divide by n below

            # Compute per-permutation bias vs oracle
            for pi_b in BUDGETS:
                true_val = oracle_risk.get(pi_b, float("nan"))
                if math.isnan(true_val):
                    continue
                # IPW estimate: (1/n) * sum_{audited} d/rho
                if pi_b in ipw_sum:
                    ipw_est = ipw_sum[pi_b] / n
                    ipw_bias.append(abs(ipw_est - true_val))
                # Naive estimate
                if naive_vals[pi_b]:
                    naive_est = statistics.mean(naive_vals[pi_b])
                    naive_bias.append(abs(naive_est - true_val))
                # Audited naive estimate
                if anv_vals[pi_b]:
                    anv_est = statistics.mean(anv_vals[pi_b])
                    audited_naive_bias.append(abs(anv_est - true_val))

            # Wrong-best selection
            # IPW: best policy according to (1/n)*sum(d/rho)
            ipw_risks = {b: ipw_sum[b] / n for b in ipw_sum}
            if ipw_risks:
                ipw_best = min(ipw_risks, key=ipw_risks.get)
                if oracle_risk.get(ipw_best, float("inf")) > oracle_risk.get(true_best, 0) + 1e-6:
                    wrong_best_ipw += 1

            # Naive: best policy among naturally-continued items
            if naive_vals:
                naive_best = min(naive_vals, key=lambda b: statistics.mean(naive_vals[b]))
                if oracle_risk.get(naive_best, float("inf")) > oracle_risk.get(true_best, 0) + 1e-6:
                    wrong_best_naive += 1

            n_trials += 1

        result_by_rho.append({
            "rho":                       rho,
            "ipw_mean_bias":             statistics.mean(ipw_bias)          if ipw_bias          else float("nan"),
            "naive_mean_bias":           statistics.mean(naive_bias)         if naive_bias         else float("nan"),
            "audited_naive_mean_bias":   statistics.mean(audited_naive_bias) if audited_naive_bias else float("nan"),
            "wrong_best_ipw_rate":       wrong_best_ipw  / max(n_trials, 1),
            "wrong_best_naive_rate":     wrong_best_naive / max(n_trials, 1),
            "oracle_risk":               oracle_risk,
            "true_best_budget":          true_best,
            "fraction_naturally_continued":
                sum(1 for b in stop_budgets.values() if b >= reference_b) / max(len(stop_budgets), 1),
        })
    return {
        "rho_sweep":    result_by_rho,
        "n_perm":       n_perm,
        "reference_b":  reference_b,
        "stop_tau":     stop_tau,
    }


# ---------------------------------------------------------------------------
# A02 — Counterfactual Missingness
# ---------------------------------------------------------------------------

def a02_counterfactual_missingness(
    table: dict,
    stop_tau: float = 0.85,
    rho: float = 0.20,
    n_perm: int = 200,
    seed: int = 42,
) -> dict:
    """
    Simulate realistic stopping: stop at first b where confidence >= stop_tau.
    Future checkpoints are hidden (counterfactual).
    Compare estimators of E[Brier(B_max)]:
      - full_info oracle
      - observed_only (use B_stop Brier, biased)
      - naive_audited (mean of audited B_max Briers, no weighting)
      - ipw (inverse-prob weight by audit rate rho)
    """
    rng = random.Random(seed)
    items = list(table.keys())

    def natural_stop(bmap):
        for b in BUDGETS:
            if b in bmap and bmap[b]["max_prob"] >= stop_tau:
                return b
        return max(b for b in BUDGETS if b in bmap)

    # Compute natural stop budget per item
    stop_budgets = {qid: natural_stop(row) for qid, row in table.items()}

    # Full-info target: mean Brier at B_max
    full_info_brier = statistics.mean(
        row[B_MAX]["brier"] for row in table.values() if B_MAX in row
    )

    bias_obs_list   = []
    bias_naive_list = []
    bias_ipw_list   = []
    risk_bias_obs   = []
    risk_bias_naive = []
    risk_bias_ipw   = []

    for _ in range(n_perm):
        perm = items[:]
        rng.shuffle(perm)
        obs_brirs  = []
        naive_brirs = []
        ipw_brirs   = []

        for qid in perm:
            row = table[qid]
            b_stop = stop_budgets[qid]
            if b_stop not in row:
                continue
            # Observed-only estimator: use Brier at stop budget (biased proxy for B_max)
            obs_brirs.append(row[b_stop]["brier"])
            # Audit: reveal B_max Brier for fraction rho
            audited = rng.random() < rho
            if audited and B_MAX in row:
                naive_brirs.append(row[B_MAX]["brier"])
                ipw_brirs.append(row[B_MAX]["brier"] / rho)

        n_total   = len(perm)
        obs_est   = statistics.mean(obs_brirs)   if obs_brirs   else float("nan")
        naive_est = statistics.mean(naive_brirs) if naive_brirs else float("nan")
        # IPW: (1/n_total) × Σ_audited (Brier / rho)
        # Dividing by n_total (not n_audited) makes this an unbiased estimator.
        ipw_est   = sum(ipw_brirs) / n_total     if ipw_brirs   else float("nan")

        if not math.isnan(obs_est):   bias_obs_list.append(obs_est   - full_info_brier)
        if not math.isnan(naive_est): bias_naive_list.append(naive_est - full_info_brier)
        if not math.isnan(ipw_est):   bias_ipw_list.append(ipw_est   - full_info_brier)

    def _summarize(lst):
        if not lst: return {"mean_bias": float("nan"), "rmse": float("nan")}
        return {
            "mean_bias": statistics.mean(lst),
            "rmse":      math.sqrt(statistics.mean(x**2 for x in lst)),
            "std":       statistics.stdev(lst) if len(lst) > 1 else 0.0,
        }

    return {
        "stop_tau":         stop_tau,
        "rho":              rho,
        "full_info_brier":  full_info_brier,
        "n_perm":           n_perm,
        "stop_budget_distribution": {
            b: sum(1 for v in stop_budgets.values() if v == b)
            for b in BUDGETS
        },
        "observed_only": _summarize(bias_obs_list),
        "naive_audited": _summarize(bias_naive_list),
        "ipw":           _summarize(bias_ipw_list),
    }


# ---------------------------------------------------------------------------
# A03 — CS vs Greedy
# ---------------------------------------------------------------------------

def a03_cs_vs_greedy(
    table: dict,
    rho: float = 0.20,
    alpha: float = 0.05,
    n_perm: int = 200,
    seed: int = 42,
    reference_b: int = 4096,
) -> dict:
    rng = random.Random(seed)
    items = list(table.keys())

    oracle_risks = {}
    for pi_b in BUDGETS:
        diffs = [row[pi_b]["brier"] - row[reference_b]["brier"]
                 for row in table.values() if pi_b in row and reference_b in row]
        oracle_risks[pi_b] = statistics.mean(diffs) if diffs else float("nan")
    oracle_best = min((b for b in BUDGETS if not math.isnan(oracle_risks[b])),
                      key=oracle_risks.get)

    greedy_wrong = 0; cs_wrong = 0; n_t = 0
    greedy_time_to_decide = []; cs_time_to_decide = []

    for _ in range(n_perm):
        perm = items[:]
        rng.shuffle(perm)
        running = defaultdict(list)
        greedy_decided = False; cs_decided = False
        greedy_t = len(perm); cs_t = len(perm)

        for t, qid in enumerate(perm):
            row = table[qid]
            if rng.random() >= rho:
                continue
            for pi_b in BUDGETS:
                if pi_b in row and reference_b in row:
                    running[pi_b].append(
                        (row[pi_b]["brier"] - row[reference_b]["brier"]) / rho
                    )
            if not greedy_decided and running and t > 10:
                greedy_decided = True
                greedy_t = t

            # CS-based: anytime-valid empirical Bernstein check
            if not cs_decided and t > 20:
                for pi_b, diffs in running.items():
                    n = len(diffs)
                    if n < 5: continue
                    m   = statistics.mean(diffs)
                    var = statistics.variance(diffs) if n > 1 else 1.0
                    c   = math.sqrt(2 * math.log(2 * len(BUDGETS) / alpha))
                    # pi_b is confidently better than reference_b if upper CB < 0
                    if m + c * math.sqrt(var / n) < 0:
                        cs_decided = True
                        cs_t = t
                        break

        greedy_sel = min(running, key=lambda b: statistics.mean(running[b])) if running else reference_b
        cs_sel = reference_b
        for pi_b, diffs in running.items():
            n = len(diffs)
            if n < 5: continue
            m   = statistics.mean(diffs)
            var = statistics.variance(diffs) if n > 1 else 1.0
            c   = math.sqrt(2 * math.log(2 * len(BUDGETS) / alpha))
            if m + c * math.sqrt(var / n) < 0:
                cs_sel = pi_b
                break

        ref_risk = oracle_risks[reference_b]
        if oracle_risks.get(greedy_sel, float("inf")) > ref_risk + 1e-6:
            greedy_wrong += 1
        if oracle_risks.get(cs_sel, float("inf")) > ref_risk + 1e-6:
            cs_wrong += 1
        greedy_time_to_decide.append(greedy_t)
        cs_time_to_decide.append(cs_t)
        n_t += 1

    return {
        "rho": rho, "alpha": alpha, "n_perm": n_perm,
        "reference_b": reference_b,
        "oracle_best": oracle_best,
        "oracle_risks": oracle_risks,
        "greedy_false_positive_rate": greedy_wrong / max(n_t, 1),
        "cs_false_positive_rate":     cs_wrong     / max(n_t, 1),
        "greedy_mean_time_to_decide": statistics.mean(greedy_time_to_decide) if greedy_time_to_decide else float("nan"),
        "cs_mean_time_to_decide":     statistics.mean(cs_time_to_decide)     if cs_time_to_decide     else float("nan"),
    }


# ---------------------------------------------------------------------------
# A04 — Audit Rate Sensitivity
# ---------------------------------------------------------------------------

def a04_audit_rate_sensitivity(
    table: dict,
    rho_values=(0.025, 0.05, 0.10, 0.20, 0.40, 1.00),
    n_perm: int = 200,
    seed: int = 42,
    reference_b: int = 4096,
) -> dict:
    rows = []
    for rho in rho_values:
        a1 = a01_ipw_vs_naive(table, rho_values=[rho], n_perm=n_perm, seed=seed,
                              reference_b=reference_b)["rho_sweep"][0]
        a3 = a03_cs_vs_greedy(table, rho=rho, n_perm=n_perm, seed=seed,
                              reference_b=reference_b)
        # Audit overhead = rho * B_MAX tokens per item (fraction of total)
        rows.append({
            "rho": rho,
            "audit_overhead_tokens_fraction": rho,
            "ipw_mean_bias":               a1["ipw_mean_bias"],
            "naive_mean_bias":             a1["naive_mean_bias"],
            "wrong_best_ipw_rate":         a1["wrong_best_ipw_rate"],
            "wrong_best_naive_rate":       a1["wrong_best_naive_rate"],
            "cs_false_positive_rate":      a3["cs_false_positive_rate"],
            "greedy_false_positive_rate":  a3["greedy_false_positive_rate"],
            "cs_mean_time_to_decide":      a3["cs_mean_time_to_decide"],
        })
    return {"rho_rows": rows, "n_perm": n_perm}


# ---------------------------------------------------------------------------
# A05 — Objective Divergence (Accuracy vs Brier vs NLL)
# ---------------------------------------------------------------------------

def a05_objective_divergence(table: dict) -> dict:
    b_acc   = best_budget(table, acc_of_rec, want_min=False)
    b_brier = best_budget(table, brier_of_rec, want_min=True)
    b_nll   = best_budget(table, nll_of_rec, want_min=True)

    budget_rows = []
    for b in BUDGETS:
        records = [row[b] for row in table.values() if b in row]
        if not records: continue
        budget_rows.append({
            "budget":     b,
            "n":          len(records),
            "accuracy":   statistics.mean(acc_of_rec(r) for r in records),
            "mean_brier": statistics.mean(brier_of_rec(r) for r in records),
            "mean_nll":   statistics.mean(nll_of_rec(r) for r in records),
        })

    # Pairwise budget divergence
    return {
        "best_budget_accuracy": b_acc,
        "best_budget_brier":    b_brier,
        "best_budget_nll":      b_nll,
        "objectives_agree":     (b_acc == b_brier == b_nll),
        "acc_brier_diverge":    (b_acc != b_brier),
        "brier_nll_diverge":    (b_brier != b_nll),
        "budget_rows":          budget_rows,
        "n_items":              len(table),
    }


# ---------------------------------------------------------------------------
# A06 — Prefix Signal Ablation (gate1 heatmap)
# ---------------------------------------------------------------------------

def a06_prefix_signal(table: dict) -> dict:
    from analysis.gate1 import gate1_full
    from analysis.data_loader import load_item_table

    # Rebuild full item_table with option_probs (as gate1 expects it)
    # gate1 uses 'option_probs' and 'answer_index' keys
    gate1_table = {}
    for qid, bmap in table.items():
        gate1_table[qid] = {}
        for b, r in bmap.items():
            gate1_table[qid][b] = {
                "option_probs": r["option_probs"],
                "answer_index": int(r["correct"]),   # can't recover; load from raw
            }

    # Load proper item_table via data_loader (has answer_index)
    return {"note": "Use analysis.gate1.gate1_full on data_loader.load_item_table output"}


def a06_prefix_signal_proper(model_tag: str, ds_tag: str) -> dict:
    """Load via data_loader and run gate1_full."""
    sys.path.insert(0, str(ROOT))
    from analysis.gate1 import gate1_full, gate1_pass
    from analysis.data_loader import load_cell as dl_load_cell

    item_table, _, entry = dl_load_cell(model_tag, ds_tag, root=ROOT)
    g1 = gate1_full(item_table)
    edges = []
    SIGNALS = ("confidence", "entropy", "margin", "answer_stability",
               "conf_slope", "prob_slope")
    for (b_lo, b_hi), edge in g1.items():
        row = {"b_lo": b_lo, "b_hi": b_hi,
               "n": edge["n"],
               "mean_G": edge["mean_G"],
               "frac_G_pos": edge["frac_G_pos"],
               "frac_G_neg": edge["frac_G_neg"]}
        for s in SIGNALS:
            sd = edge.get(s, {})
            row[f"{s}_auc"]      = sd.get("auc", float("nan"))
            row[f"{s}_spearman"] = sd.get("spearman", float("nan"))
        edges.append(row)
    return {
        "edges": edges,
        "gate1_pass": gate1_pass(g1),
        "n_items": len(item_table),
    }


# ---------------------------------------------------------------------------
# A07 — Signal Combination
# ---------------------------------------------------------------------------

def a07_signal_combination(table: dict, seed: int = 42) -> dict:
    """
    Cross-validated logistic regression combining prefix signals to predict
    whether more compute helps (Brier decreases from b_lo to b_hi).
    Compare: single best signal vs. combination.
    """
    from analysis.gate1 import gate1_full
    sys.path.insert(0, str(ROOT))
    from analysis.data_loader import load_item_table

    rng = random.Random(seed)

    # Build feature matrix per adjacent edge
    results_per_edge = []
    budgets = BUDGETS
    for i in range(len(budgets) - 1):
        b_lo, b_hi = budgets[i], budgets[i+1]
        b_prev = budgets[i-1] if i > 0 else None

        X, y = [], []
        for row in table.values():
            if b_lo not in row or b_hi not in row:
                continue
            r_lo = row[b_lo]
            r_hi = row[b_hi]
            improvement = int(r_hi["brier"] < r_lo["brier"] - 1e-6)

            probs = r_lo["option_probs"]
            conf  = max(probs)
            entr  = -sum(p * math.log(max(p, 1e-12)) for p in probs)
            s     = sorted(probs, reverse=True)
            marg  = s[0] - s[1] if len(s) >= 2 else s[0]

            feats = [conf, entr, marg]
            if b_prev is not None and b_prev in row:
                r_prev = row[b_prev]
                stab  = float(probs.index(max(probs)) == r_prev["option_probs"].index(max(r_prev["option_probs"])))
                cslope = conf - max(r_prev["option_probs"])
                feats += [stab, cslope]
            else:
                feats += [float("nan"), float("nan")]

            X.append(feats)
            y.append(improvement)

        if len(X) < 20:
            continue

        # Simple 5-fold cross-validated AUC (using manual logistic regression)
        n = len(X)
        # Only use non-NaN features for single-signal baselines
        auc_conf = _simple_auc([-row[0] for row in X], y)  # high conf → less improvement
        auc_entr = _simple_auc([row[1] for row in X], y)   # high entropy → more improvement
        auc_marg = _simple_auc([-row[2] for row in X], y)  # high margin → less improvement

        # Combination: cross-validated logistic regression
        fold_aucs = []
        indices = list(range(n))
        rng.shuffle(indices)
        folds = [indices[k::5] for k in range(5)]
        for f_idx in range(5):
            test_idx  = set(folds[f_idx])
            train_idx = [i for i in range(n) if i not in test_idx]
            X_tr = [[X[i][j] for j in range(3)] for i in train_idx]  # use 3 non-NaN features
            y_tr = [y[i] for i in train_idx]
            X_te = [[X[i][j] for j in range(3)] for i in test_idx]
            y_te = [y[i] for i in test_idx]
            w, b_c = _logistic_train(X_tr, y_tr, n_iter=200, lr=0.01, seed=seed+f_idx)
            scores = [_logistic_score(x, w, b_c) for x in X_te]
            fold_aucs.append(_auc_from_scores(scores, y_te))

        combo_auc = statistics.mean(fold_aucs) if fold_aucs else float("nan")

        results_per_edge.append({
            "b_lo": b_lo, "b_hi": b_hi,
            "n": n,
            "auc_confidence":  auc_conf,
            "auc_entropy":     auc_entr,
            "auc_margin":      auc_marg,
            "auc_combination_cv5": combo_auc,
            "best_single_auc": max(auc_conf, auc_entr, auc_marg),
            "combo_gain":      combo_auc - max(auc_conf, auc_entr, auc_marg),
        })

    return {"edges": results_per_edge, "n_items": len(table)}


def _logistic_train(X, y, n_iter=200, lr=0.01, seed=42):
    rng = random.Random(seed)
    n_feat = len(X[0]) if X else 0
    if n_feat == 0 or not X:
        return [0.0] * n_feat, 0.0
    w = [rng.gauss(0, 0.01) for _ in range(n_feat)]
    b = 0.0
    for _ in range(n_iter):
        dw = [0.0] * n_feat
        db = 0.0
        for xi, yi in zip(X, y):
            z = sum(w[j] * xi[j] for j in range(n_feat)) + b
            sig = 1 / (1 + math.exp(-z))
            err = sig - yi
            for j in range(n_feat):
                dw[j] += err * xi[j]
            db += err
        n = len(X)
        w = [w[j] - lr * dw[j] / n for j in range(n_feat)]
        b = b - lr * db / n
    return w, b


def _logistic_score(x, w, b):
    z = sum(w[j] * x[j] for j in range(len(w))) + b
    return 1 / (1 + math.exp(-z))


def _simple_auc(scores, labels):
    return _auc_from_scores(scores, labels)


def _auc_from_scores(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    c = sum(sp > sn for sp in pos for sn in neg)
    t = sum(sp == sn for sp in pos for sn in neg)
    return (c + 0.5 * t) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# A08 — Global vs Item-Adaptive Policy
# ---------------------------------------------------------------------------

def a08_global_vs_item_adaptive(table: dict) -> dict:
    """
    Compare:
    - Best fixed global budget (calibrated on all items — oracle global)
    - Item-adaptive oracle (per-item optimal budget)
    - VISTA-global (conf+margin threshold)
    - VISTA-item (per-category threshold)
    At matched Brier quality constraint.
    """
    from analysis.baselines import (
        run_all_policies, evaluate_policy,
        conf_thresh_policy, oracle_brier, oracle_accuracy,
    )
    from analysis.gate0 import gate0_analysis

    # Convert local table back to format baselines expects
    # baselines.py expects item_table[qid][b] = {'option_probs':..., 'answer_index':...}
    # We have item_table[qid][b] = {'brier':..., 'correct':..., 'option_probs':..., ...}

    # Build gate0 for oracle
    from analysis.gate0 import hindsight_oracle_budget
    oracle_budgets = []
    for bmap in table.values():
        # Reconstruct brier-minimizing budget
        briers = {b: r["brier"] for b, r in bmap.items()}
        best_b = min(briers, key=briers.get)
        oracle_budgets.append(best_b)

    g0 = {
        "n_items":         len(table),
        "mean_oracle":     statistics.mean(oracle_budgets),
        "median_oracle":   statistics.median(oracle_budgets),
        "p90_oracle":      _percentile(oracle_budgets, 0.90),
        "saving_vs_max":   1.0 - statistics.mean(oracle_budgets) / B_MAX,
        "budget_dist":     {b: oracle_budgets.count(b) for b in BUDGETS},
    }

    # Fixed policies
    fixed_results = {}
    for b in BUDGETS:
        records = [row[b] for row in table.values() if b in row]
        if not records: continue
        fixed_results[b] = {
            "budget": b,
            "n": len(records),
            "accuracy":   statistics.mean(r["correct"] for r in records),
            "mean_brier": statistics.mean(r["brier"] for r in records),
            "mean_nll":   statistics.mean(r["nll"] for r in records),
            "token_saving": 1.0 - b / B_MAX,
        }

    best_fixed_b = min(fixed_results, key=lambda b: fixed_results[b]["mean_brier"])

    # Threshold policies evaluated
    def eval_threshold(tau_conf, tau_margin=None, tau_entropy=None):
        assignments = {}
        for qid, bmap in table.items():
            chosen = max(b for b in BUDGETS if b in bmap)
            for b in BUDGETS:
                if b not in bmap: continue
                r = bmap[b]
                probs = r["option_probs"]
                conf = max(probs)
                s = sorted(probs, reverse=True)
                marg = s[0] - s[1] if len(s) >= 2 else s[0]
                entr = -sum(p * math.log(max(p, 1e-12)) for p in probs)
                cond = conf >= tau_conf
                if tau_margin is not None: cond = cond and (marg >= tau_margin)
                if tau_entropy is not None: cond = cond and (entr <= tau_entropy)
                if cond:
                    chosen = b
                    break
            assignments[qid] = chosen
        records = []
        budgets_used = []
        for qid, b in assignments.items():
            if b in table[qid]:
                records.append(table[qid][b])
                budgets_used.append(b)
        return {
            "mean_brier": statistics.mean(r["brier"] for r in records) if records else float("nan"),
            "accuracy":   statistics.mean(r["correct"] for r in records) if records else float("nan"),
            "mean_tokens": statistics.mean(budgets_used) if budgets_used else float("nan"),
            "token_saving": 1.0 - statistics.mean(budgets_used) / B_MAX if budgets_used else float("nan"),
        }

    # VISTA global (tau_conf=0.85, tau_margin=0.45)
    vista_global_res  = eval_threshold(0.85, tau_margin=0.45)
    # Confidence-only threshold policies (sweep)
    conf_sweep = []
    for tau in [0.70, 0.80, 0.85, 0.90, 0.95]:
        res = eval_threshold(tau)
        res["tau"] = tau
        conf_sweep.append(res)

    return {
        "gate0": g0,
        "fixed": fixed_results,
        "best_fixed_b": best_fixed_b,
        "best_fixed_brier": fixed_results[best_fixed_b]["mean_brier"],
        "oracle_mean_brier": statistics.mean(
            min(row[b]["brier"] for b in BUDGETS if b in row)
            for row in table.values()
        ),
        "oracle_token_saving": g0["saving_vs_max"],
        "vista_global": vista_global_res,
        "conf_threshold_sweep": conf_sweep,
    }


# ---------------------------------------------------------------------------
# A09 — Matched-Quality Policy Comparison (main table candidate)
# ---------------------------------------------------------------------------

def a09_matched_quality(table: dict) -> dict:
    """
    For every policy, compute metrics under matched-quality constraint:
    'at most epsilon Brier degradation vs. B_max baseline'.
    """
    # B_max baseline
    bmax_records = [row[B_MAX] for row in table.values() if B_MAX in row]
    bmax_brier   = statistics.mean(r["brier"]   for r in bmax_records)
    bmax_acc     = statistics.mean(r["correct"] for r in bmax_records)
    bmax_nll     = statistics.mean(r["nll"]     for r in bmax_records)

    def eval_policy_by_threshold(tau_conf, tau_margin=None):
        assignments = {}
        for qid, bmap in table.items():
            chosen = max(b for b in BUDGETS if b in bmap)
            for b in BUDGETS:
                if b not in bmap: continue
                probs = bmap[b]["option_probs"]
                conf  = max(probs)
                s     = sorted(probs, reverse=True)
                marg  = s[0] - s[1] if len(s) >= 2 else s[0]
                cond  = conf >= tau_conf
                if tau_margin is not None: cond = cond and (marg >= tau_margin)
                if cond:
                    chosen = b
                    break
            assignments[qid] = chosen
        records = [table[qid][b] for qid, b in assignments.items() if b in table[qid]]
        buds    = [assignments[qid] for qid in assignments if assignments[qid] in table[qid]]
        if not records:
            return {"feasible": False}
        br  = statistics.mean(r["brier"] for r in records)
        ac  = statistics.mean(r["correct"] for r in records)
        nl  = statistics.mean(r["nll"] for r in records)
        mt  = statistics.mean(buds)
        sav = 1.0 - mt / B_MAX
        return {
            "feasible": True,
            "mean_brier": br, "accuracy": ac, "mean_nll": nl,
            "mean_tokens": mt, "token_saving": sav,
            "brier_delta_vs_bmax": br - bmax_brier,
            "acc_delta_vs_bmax":   ac - bmax_acc,
        }

    policies = {}
    for b in BUDGETS:
        recs = [row[b] for row in table.values() if b in row]
        buds = [b] * len(recs)
        if not recs: continue
        policies[f"Fixed B={b}"] = {
            "feasible": True,
            "mean_brier": statistics.mean(r["brier"] for r in recs),
            "accuracy":   statistics.mean(r["correct"] for r in recs),
            "mean_nll":   statistics.mean(r["nll"] for r in recs),
            "mean_tokens": float(b),
            "token_saving": 1.0 - b / B_MAX,
            "brier_delta_vs_bmax": statistics.mean(r["brier"] for r in recs) - bmax_brier,
            "acc_delta_vs_bmax":   statistics.mean(r["correct"] for r in recs) - bmax_acc,
        }

    for tau, label in [(0.90, "Conf≥0.90"), (0.85, "Conf≥0.85"), (0.80, "Conf≥0.80")]:
        policies[label] = eval_policy_by_threshold(tau)

    policies["VISTA-Global(0.85,0.45)"] = eval_policy_by_threshold(0.85, 0.45)
    policies["VISTA-Global(0.90,0.50)"] = eval_policy_by_threshold(0.90, 0.50)

    # Oracle Brier
    oracle_recs = []
    oracle_buds = []
    for qid, bmap in table.items():
        best_b = min((b for b in BUDGETS if b in bmap), key=lambda b: bmap[b]["brier"])
        oracle_recs.append(bmap[best_b])
        oracle_buds.append(best_b)
    policies["Oracle-Brier"] = {
        "feasible": True,
        "mean_brier":   statistics.mean(r["brier"] for r in oracle_recs),
        "accuracy":     statistics.mean(r["correct"] for r in oracle_recs),
        "mean_nll":     statistics.mean(r["nll"] for r in oracle_recs),
        "mean_tokens":  statistics.mean(oracle_buds),
        "token_saving": 1.0 - statistics.mean(oracle_buds) / B_MAX,
        "brier_delta_vs_bmax": statistics.mean(r["brier"] for r in oracle_recs) - bmax_brier,
        "acc_delta_vs_bmax":   statistics.mean(r["correct"] for r in oracle_recs) - bmax_acc,
    }

    return {
        "bmax_brier": bmax_brier,
        "bmax_acc":   bmax_acc,
        "bmax_nll":   bmax_nll,
        "n_items":    len(table),
        "policies":   policies,
    }


# ---------------------------------------------------------------------------
# A10 — Lambda Sensitivity
# ---------------------------------------------------------------------------

def a10_lambda_sensitivity(table: dict) -> dict:
    """
    Deployment loss: L(lambda) = Brier + lambda * (mean_tokens / B_MAX)
    Sweep lambda on log scale.
    """
    lambdas = [10**x for x in np.linspace(-4, 0, 25).tolist()]
    pareto_rows = []

    for lam in lambdas:
        # For each candidate fixed policy, compute combined loss
        best_b = None
        best_loss = float("inf")
        for b in BUDGETS:
            recs = [row[b] for row in table.values() if b in row]
            if not recs: continue
            br = statistics.mean(r["brier"] for r in recs)
            compute_frac = b / B_MAX
            loss = br + lam * compute_frac
            if loss < best_loss:
                best_loss = loss
                best_b = b

        # Also check threshold policies
        for tau_conf in [0.80, 0.85, 0.90]:
            assignments = {}
            for qid, bmap in table.items():
                chosen = max(b for b in BUDGETS if b in bmap)
                for b in BUDGETS:
                    if b not in bmap: continue
                    if bmap[b]["max_prob"] >= tau_conf:
                        chosen = b
                        break
                assignments[qid] = chosen
            recs = [table[qid][b] for qid, b in assignments.items() if b in table[qid]]
            buds = [assignments[qid] for qid in assignments if assignments[qid] in table[qid]]
            if not recs: continue
            br = statistics.mean(r["brier"] for r in recs)
            compute_frac = statistics.mean(buds) / B_MAX
            loss = br + lam * compute_frac
            if loss < best_loss:
                best_loss = loss
                best_b = f"conf≥{tau_conf}"

        pareto_rows.append({
            "lambda": lam,
            "best_policy": str(best_b),
            "best_combined_loss": best_loss,
        })

    # Build Pareto frontier: (mean_tokens, mean_brier) per fixed policy
    pareto_points = []
    for b in BUDGETS:
        recs = [row[b] for row in table.values() if b in row]
        if not recs: continue
        pareto_points.append({
            "policy": f"Fixed B={b}",
            "mean_tokens": float(b),
            "mean_brier":  statistics.mean(r["brier"] for r in recs),
            "accuracy":    statistics.mean(r["correct"] for r in recs),
        })

    return {
        "n_items": len(table),
        "lambda_sweep": pareto_rows,
        "pareto_points": pareto_points,
    }


# ---------------------------------------------------------------------------
# A11 — Prequential vs Fixed Calibration
# ---------------------------------------------------------------------------

def a11_prequential_vs_fixed(
    table: dict,
    n_cal_values=(50, 100, 200),
    n_perm: int = 200,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    items = list(table.keys())
    n = len(items)

    oracle_per_item = {
        qid: min((r["brier"] for r in row.values()), default=0.0)
        for qid, row in table.items()
    }

    results_by_ncal = []
    for n_cal in n_cal_values:
        preq_regrets = []
        fixed_regrets = []

        for _ in range(n_perm):
            perm = items[:]
            rng.shuffle(perm)
            # Track running mean via cumulative sum + count (O(1) per update)
            run_sum = defaultdict(float)
            run_cnt = defaultdict(int)
            preq_loss = 0.0
            fixed_loss = 0.0
            oracle_loss = 0.0
            fixed_policy = None

            for t, qid in enumerate(perm):
                row = table[qid]
                oracle_loss += oracle_per_item.get(qid, 0.0)

                if run_cnt:
                    preq_pi = min(run_cnt, key=lambda b: run_sum[b] / max(run_cnt[b], 1))
                else:
                    preq_pi = BUDGETS[-1]

                if t == n_cal and fixed_policy is None:
                    fixed_policy = preq_pi
                fp = fixed_policy if fixed_policy is not None else BUDGETS[-1]

                b_preq = preq_pi if preq_pi in row else min(row)
                b_fix  = fp if fp in row else min(row)
                preq_loss  += row[b_preq]["brier"]
                fixed_loss += row[b_fix]["brier"]

                for b, info in row.items():
                    run_sum[b] += info["brier"]
                    run_cnt[b] += 1

            N = len(perm)
            preq_regrets.append((preq_loss - oracle_loss) / N)
            fixed_regrets.append((fixed_loss - oracle_loss) / N)

        results_by_ncal.append({
            "n_cal": n_cal,
            "prequential_mean_regret": statistics.mean(preq_regrets),
            "fixed_cal_mean_regret":   statistics.mean(fixed_regrets),
            "advantage_of_preq": statistics.mean(fixed_regrets) - statistics.mean(preq_regrets),
            "n_perm": n_perm,
        })

    return {"n_cal_sweep": results_by_ncal, "n_items": n}


# ---------------------------------------------------------------------------
# A12 — Distribution Shift
# ---------------------------------------------------------------------------

def a12_distribution_shift(
    tables: dict[str, dict],
    shift_configs: list[dict],
    n_perm: int = 100,
    seed: int = 42,
) -> dict:
    """
    tables: {ds_tag: item_table}
    shift_configs: [{"label": "ARC→MMLU", "phase1": "ds2", "phase2": "ds1", "split": 0.5}]
    Compare: fixed-best-on-phase1, prequential-VISTA, oracle.
    """
    rng = random.Random(seed)
    results = []

    for cfg in shift_configs:
        p1_tag, p2_tag = cfg["phase1"], cfg["phase2"]
        if p1_tag not in tables or p2_tag not in tables:
            results.append({"label": cfg["label"], "skipped": True, "reason": "missing table"})
            continue
        t1 = tables[p1_tag]
        t2 = tables[p2_tag]
        n_split = int(cfg.get("split", 0.5) * min(len(t1), len(t2)))

        # Oracle on full merged stream
        items1 = list(t1.keys())[:n_split]
        items2 = list(t2.keys())[:n_split]
        stream = [(qid, t1) for qid in items1] + [(qid, t2) for qid in items2]

        # Calibrate fixed policy on phase1
        brier_by_b = defaultdict(list)
        for qid in items1:
            for b, r in t1[qid].items():
                brier_by_b[b].append(r["brier"])
        fixed_b = min(
            (b for b in BUDGETS if brier_by_b[b]),
            key=lambda b: statistics.mean(brier_by_b[b])
        )

        fixed_losses  = []; preq_losses  = []; oracle_losses = []
        running = defaultdict(list)

        for _ in range(n_perm):
            rng.shuffle(stream)
            fixed_l = 0.0; preq_l = 0.0; ora_l = 0.0
            local_run = defaultdict(list)

            for t, (qid, tbl) in enumerate(stream):
                row = tbl[qid]
                ora_l += min(row[b]["brier"] for b in BUDGETS if b in row)

                # Fixed
                fb = fixed_b if fixed_b in row else max(b for b in BUDGETS if b in row)
                fixed_l += row[fb]["brier"]

                # Prequential
                if local_run:
                    preq_b = min(local_run, key=lambda b: statistics.mean(local_run[b]))
                else:
                    preq_b = BUDGETS[-1]
                pb = preq_b if preq_b in row else max(b for b in BUDGETS if b in row)
                preq_l += row[pb]["brier"]

                for b in BUDGETS:
                    if b in row:
                        local_run[b].append(row[b]["brier"])

            N = len(stream)
            fixed_losses.append(fixed_l / N)
            preq_losses.append(preq_l / N)
            oracle_losses.append(ora_l / N)

        results.append({
            "label": cfg["label"],
            "phase1": p1_tag, "phase2": p2_tag,
            "fixed_policy_budget": fixed_b,
            "fixed_mean_loss":     statistics.mean(fixed_losses),
            "preq_mean_loss":      statistics.mean(preq_losses),
            "oracle_mean_loss":    statistics.mean(oracle_losses),
            "preq_advantage_over_fixed": statistics.mean(fixed_losses) - statistics.mean(preq_losses),
            "n_stream": len(stream),
            "n_perm": n_perm,
        })

    return {"shift_results": results}


# ---------------------------------------------------------------------------
# A13 — Cross-Model Transfer
# ---------------------------------------------------------------------------

def a13_cross_model_transfer(
    tables_m1: dict[str, dict],
    tables_m2: dict[str, dict],
    ds_tags: list[str],
    tau_grid=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
) -> dict:
    """
    Calibrate threshold on M1, apply to M2 (and vice versa).
    Measure degradation in Brier and token savings.
    """
    results = []
    for ds_tag in ds_tags:
        if ds_tag not in tables_m1 or ds_tag not in tables_m2:
            continue
        t1 = tables_m1[ds_tag]
        t2 = tables_m2[ds_tag]

        def eval_thresh(table, tau):
            recs = []; buds = []
            for qid, bmap in table.items():
                chosen = max(b for b in BUDGETS if b in bmap)
                for b in BUDGETS:
                    if b in bmap and bmap[b]["max_prob"] >= tau:
                        chosen = b; break
                if chosen in bmap:
                    recs.append(bmap[chosen])
                    buds.append(chosen)
            return {
                "brier":   statistics.mean(r["brier"] for r in recs) if recs else float("nan"),
                "acc":     statistics.mean(r["correct"] for r in recs) if recs else float("nan"),
                "tokens":  statistics.mean(buds) if buds else float("nan"),
                "saving":  1.0 - statistics.mean(buds) / B_MAX if buds else float("nan"),
            }

        # Calibrate on M1: best tau by Brier
        best_tau_m1 = min(tau_grid, key=lambda tau: eval_thresh(t1, tau)["brier"])
        m1_tuned    = eval_thresh(t1, best_tau_m1)
        m2_transfer = eval_thresh(t2, best_tau_m1)  # apply M1 tau to M2

        # Calibrate on M2: best tau by Brier
        best_tau_m2 = min(tau_grid, key=lambda tau: eval_thresh(t2, tau)["brier"])
        m2_tuned    = eval_thresh(t2, best_tau_m2)
        m1_transfer = eval_thresh(t1, best_tau_m2)  # apply M2 tau to M1

        # Fixed B_max baseline per model
        m1_bmax = {k: statistics.mean(r[k] for r in [row[B_MAX] for row in t1.values() if B_MAX in row])
                   for k in ("brier", "correct")}
        m2_bmax = {k: statistics.mean(r[k] for r in [row[B_MAX] for row in t2.values() if B_MAX in row])
                   for k in ("brier", "correct")}

        results.append({
            "ds_tag": ds_tag,
            "m1_best_tau": best_tau_m1,
            "m1_tuned": m1_tuned,
            "m2_transfer_from_m1": m2_transfer,
            "m2_best_tau": best_tau_m2,
            "m2_tuned": m2_tuned,
            "m1_transfer_from_m2": m1_transfer,
            "m1_bmax": m1_bmax,
            "m2_bmax": m2_bmax,
            "m1to_m2_brier_degradation": m2_transfer["brier"] - m2_tuned["brier"],
            "m2to_m1_brier_degradation": m1_transfer["brier"] - m1_tuned["brier"],
        })
    return {"transfer_results": results}


# ---------------------------------------------------------------------------
# A14 — Cross-Dataset Transfer
# ---------------------------------------------------------------------------

def a14_cross_dataset_transfer(
    tables: dict[str, dict],
    tau_grid=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    ds_pairs=None,
) -> dict:
    if ds_pairs is None:
        all_ds = list(tables.keys())
        ds_pairs = [(a, b) for a in all_ds for b in all_ds if a != b]

    def eval_thresh(table, tau):
        recs = []; buds = []
        for qid, bmap in table.items():
            chosen = max(b for b in BUDGETS if b in bmap)
            for b in BUDGETS:
                if b in bmap and bmap[b]["max_prob"] >= tau:
                    chosen = b; break
            if chosen in bmap:
                recs.append(bmap[chosen]); buds.append(chosen)
        return {
            "brier":  statistics.mean(r["brier"] for r in recs) if recs else float("nan"),
            "acc":    statistics.mean(r["correct"] for r in recs) if recs else float("nan"),
            "tokens": statistics.mean(buds) if buds else float("nan"),
            "saving": 1.0 - statistics.mean(buds) / B_MAX if buds else float("nan"),
        }

    results = []
    for src, tgt in ds_pairs:
        if src not in tables or tgt not in tables:
            continue
        best_tau_src = min(tau_grid, key=lambda t: eval_thresh(tables[src], t)["brier"])
        src_tuned    = eval_thresh(tables[src], best_tau_src)
        tgt_transfer = eval_thresh(tables[tgt], best_tau_src)
        best_tau_tgt = min(tau_grid, key=lambda t: eval_thresh(tables[tgt], t)["brier"])
        tgt_tuned    = eval_thresh(tables[tgt], best_tau_tgt)
        results.append({
            "source": src, "target": tgt,
            "tau_from_source":        best_tau_src,
            "source_tuned":           src_tuned,
            "target_transferred":     tgt_transfer,
            "target_tuned":           tgt_tuned,
            "brier_degradation":      tgt_transfer["brier"] - tgt_tuned["brier"],
            "token_saving_delta":     tgt_transfer["saving"] - tgt_tuned["saving"],
        })
    return {"transfer_results": results}


# ---------------------------------------------------------------------------
# A15 — BLOCKED (needs GPU)
# ---------------------------------------------------------------------------

def a15_blocked() -> dict:
    return {
        "status": "BLOCKED",
        "reason": "Requires GPU inference: independent generation vs. forced-checkpoint comparison.",
        "recovery_script": "recover_m1_ds2.py can be extended for this experiment.",
        "estimated_items": "100-200 items × 2 models × 2 datasets × 4 budgets",
        "paper_placement": "APPENDIX",
    }


# ---------------------------------------------------------------------------
# A16 — Delayed Label Latency
# ---------------------------------------------------------------------------

def a16_delayed_label_latency(
    table: dict,
    delay_values=(0, 10, 50, 100, 500),
    n_perm: int = 100,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    items = list(table.keys())
    n = len(items)

    oracle_per_item = {
        qid: min((r["brier"] for r in row.values()), default=0.0)
        for qid, row in table.items()
    }

    results = []
    for delay in delay_values:
        regrets = []
        for _ in range(n_perm):
            perm = items[:]
            rng.shuffle(perm)
            run_sum = defaultdict(float)
            run_cnt = defaultdict(int)
            label_buffer = []   # (t_available, brier_per_budget dict)
            total_loss = 0.0
            oracle_loss = 0.0

            for t, qid in enumerate(perm):
                row = table[qid]
                oracle_loss += oracle_per_item.get(qid, 0.0)

                # Apply delayed labels: drain buffer (O(delay) amortized)
                remaining = []
                for (t_avail, brow) in label_buffer:
                    if t_avail <= t:
                        for b, bval in brow.items():
                            run_sum[b] += bval
                            run_cnt[b] += 1
                    else:
                        remaining.append((t_avail, brow))
                label_buffer = remaining

                # Choose policy
                preq_b = min(run_cnt, key=lambda b: run_sum[b] / max(run_cnt[b], 1)) if run_cnt else BUDGETS[-1]
                b_use  = preq_b if preq_b in row else max(b for b in BUDGETS if b in row)
                total_loss += row[b_use]["brier"]

                # Schedule label delivery
                label_buffer.append((t + delay, {b: r["brier"] for b, r in row.items()}))

            N = len(perm)
            oracle_loss_full = sum(oracle_per_item.get(qid, 0.0) for qid in perm)
            regrets.append((total_loss - oracle_loss_full) / N)

        results.append({
            "delay": delay,
            "mean_regret": statistics.mean(regrets),
            "std_regret":  statistics.stdev(regrets) if len(regrets) > 1 else 0.0,
            "n_perm": n_perm,
        })

    return {"delay_results": results, "n_items": n}


# ---------------------------------------------------------------------------
# A17 — Policy Family Size
# ---------------------------------------------------------------------------

def a17_policy_family_size(
    table: dict,
    k_values=(2, 3, 4, 6, 10, 25, 50),
    rho: float = 0.20,
    alpha: float = 0.05,
    n_perm: int = 100,
    seed: int = 42,
) -> dict:
    """
    Vary K (number of candidate budget levels).
    Measure CS width and wrong-selection rate as K grows.
    """
    rng = random.Random(seed)
    items = list(table.keys())
    results = []

    for k in k_values:
        # Choose K budgets evenly spaced from BUDGETS
        if k >= len(BUDGETS):
            chosen_budgets = BUDGETS
        else:
            step = (len(BUDGETS) - 1) / (k - 1) if k > 1 else 0
            chosen_budgets = [BUDGETS[int(round(i * step))] for i in range(k)]

        reference_b = chosen_budgets[-1]
        oracle_risks = {}
        for pi_b in chosen_budgets:
            diffs = [row[pi_b]["brier"] - row[reference_b]["brier"]
                     for row in table.values() if pi_b in row and reference_b in row]
            oracle_risks[pi_b] = statistics.mean(diffs) if diffs else float("nan")
        oracle_best = min((b for b in chosen_budgets if not math.isnan(oracle_risks.get(b, float("nan")))),
                          key=oracle_risks.get, default=reference_b)

        cs_widths = []
        wrong_sel = 0; n_t = 0
        for _ in range(n_perm):
            perm = items[:]
            rng.shuffle(perm)
            running = defaultdict(list)
            for qid in perm:
                row = table[qid]
                if rng.random() >= rho: continue
                for pi_b in chosen_budgets:
                    if pi_b in row and reference_b in row:
                        running[pi_b].append(
                            (row[pi_b]["brier"] - row[reference_b]["brier"]) / rho
                        )
            # CS width for each policy
            widths = []
            cs_sel = reference_b
            c_bonf = math.sqrt(2 * math.log(2 * k / alpha))
            for pi_b, diffs in running.items():
                n_d = len(diffs)
                if n_d < 2: continue
                m    = statistics.mean(diffs)
                var  = statistics.variance(diffs)
                half = c_bonf * math.sqrt(var / n_d)
                widths.append(2 * half)
                if m + half < 0:
                    cs_sel = pi_b
            if widths: cs_widths.append(statistics.mean(widths))
            if oracle_risks.get(cs_sel, float("inf")) > oracle_risks.get(oracle_best, 0) + 1e-6:
                wrong_sel += 1
            n_t += 1

        results.append({
            "k": k,
            "chosen_budgets": chosen_budgets,
            "mean_cs_width":     statistics.mean(cs_widths) if cs_widths else float("nan"),
            "wrong_selection_rate": wrong_sel / max(n_t, 1),
            "n_perm": n_perm,
        })

    return {"k_results": results, "n_items": len(table)}


# ---------------------------------------------------------------------------
# A18 — Audit Schedule
# ---------------------------------------------------------------------------

def a18_audit_schedule(
    table: dict,
    n_perm: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
    reference_b: int = 4096,
) -> dict:
    rng = random.Random(seed)
    items = list(table.keys())
    n = len(items)

    oracle_risks = {}
    for pi_b in BUDGETS:
        diffs = [row[pi_b]["brier"] - row[reference_b]["brier"]
                 for row in table.values() if pi_b in row and reference_b in row]
        oracle_risks[pi_b] = statistics.mean(diffs) if diffs else float("nan")
    oracle_best = min((b for b in BUDGETS if not math.isnan(oracle_risks.get(b, float("nan")))),
                      key=oracle_risks.get, default=reference_b)

    schedules = {
        "fixed_0.20":    lambda t, n: 0.20,
        "decay":         lambda t, n: max(0.05, 0.40 / math.sqrt(t + 1)),
        "warm_start":    lambda t, n: 0.40 if t < n * 0.25 else (0.20 if t < n * 0.5 else 0.10),
        "adaptive_0.10": lambda t, n: 0.10,
    }

    results = {}
    for sched_name, rho_fn in schedules.items():
        wrong_sel = 0; n_t = 0
        total_audit_fractions = []

        for _ in range(n_perm):
            perm = items[:]
            rng.shuffle(perm)
            running = defaultdict(list)
            audit_count = 0

            for t, qid in enumerate(perm):
                row = table[qid]
                rho = rho_fn(t, n)
                if rng.random() < rho:
                    audit_count += 1
                    for pi_b in BUDGETS:
                        if pi_b in row and reference_b in row:
                            running[pi_b].append(
                                (row[pi_b]["brier"] - row[reference_b]["brier"]) / rho
                            )

            c_bonf = math.sqrt(2 * math.log(2 * len(BUDGETS) / alpha))
            cs_sel = reference_b
            for pi_b, diffs in running.items():
                nd = len(diffs)
                if nd < 2: continue
                m   = statistics.mean(diffs)
                var = statistics.variance(diffs)
                if m + c_bonf * math.sqrt(var / nd) < 0:
                    cs_sel = pi_b; break

            if oracle_risks.get(cs_sel, float("inf")) > oracle_risks.get(oracle_best, 0) + 1e-6:
                wrong_sel += 1
            total_audit_fractions.append(audit_count / max(n, 1))
            n_t += 1

        results[sched_name] = {
            "wrong_selection_rate":   wrong_sel / max(n_t, 1),
            "mean_audit_fraction":    statistics.mean(total_audit_fractions),
            "n_perm": n_perm,
        }

    return {"schedule_results": results, "oracle_best": oracle_best, "n_items": n}


# ---------------------------------------------------------------------------
# A19 — Bootstrap / Random-Order Robustness
# ---------------------------------------------------------------------------

def a19_bootstrap_robustness(
    table: dict,
    n_boot: int = 1000,
    n_perm: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    rng = random.Random(seed)
    items = list(table.keys())

    # Key metrics: Brier and accuracy at each budget, oracle saving
    budget_boot_ci = {}
    for b in BUDGETS:
        recs = [row[b] for row in table.values() if b in row]
        if not recs: continue
        briers = [r["brier"] for r in recs]
        accs   = [int(r["correct"]) for r in recs]
        n_r    = len(briers)
        boot_b = []; boot_a = []
        for _ in range(n_boot):
            idx = [rng.randrange(n_r) for _ in range(n_r)]
            boot_b.append(statistics.mean(briers[i] for i in idx))
            boot_a.append(statistics.mean(accs[i]   for i in idx))
        boot_b.sort(); boot_a.sort()
        lo_i = int(alpha / 2 * n_boot)
        hi_i = int((1 - alpha / 2) * n_boot) - 1
        budget_boot_ci[b] = {
            "brier_point": statistics.mean(briers),
            "brier_lo":    boot_b[max(0, lo_i)],
            "brier_hi":    boot_b[min(n_boot-1, hi_i)],
            "acc_point":   statistics.mean(accs),
            "acc_lo":      boot_a[max(0, lo_i)],
            "acc_hi":      boot_a[min(n_boot-1, hi_i)],
            "n": n_r,
        }

    # Oracle saving CI
    oracle_buds = [
        min((b for b in BUDGETS if b in row), key=lambda b: row[b]["brier"])
        for row in table.values()
    ]
    n_o = len(oracle_buds)
    boot_saving = []
    for _ in range(n_boot):
        idx = [rng.randrange(n_o) for _ in range(n_o)]
        smp = [oracle_buds[i] for i in idx]
        boot_saving.append(1.0 - statistics.mean(smp) / B_MAX)
    boot_saving.sort()
    oracle_saving_ci = {
        "point": 1.0 - statistics.mean(oracle_buds) / B_MAX,
        "lo":    boot_saving[max(0, int(alpha/2*n_boot))],
        "hi":    boot_saving[min(n_boot-1, int((1-alpha/2)*n_boot)-1)],
        "n_items": n_o,
    }

    # Stream-order sensitivity: variance of prequential Brier across permutations
    final_losses = []
    for _ in range(min(n_perm, 500)):
        perm = items[:]
        rng.shuffle(perm)
        run_sum = defaultdict(float)
        run_cnt = defaultdict(int)
        total = 0.0
        for qid in perm:
            row = table[qid]
            preq_b = min(run_cnt, key=lambda b: run_sum[b] / max(run_cnt[b], 1)) if run_cnt else BUDGETS[-1]
            b_use  = preq_b if preq_b in row else max(b for b in BUDGETS if b in row)
            total += row[b_use]["brier"]
            for b, r in row.items():
                run_sum[b] += r["brier"]
                run_cnt[b] += 1
        final_losses.append(total / len(perm))

    return {
        "n_boot": n_boot,
        "n_perm": n_perm,
        "alpha":  alpha,
        "budget_bootstrap_ci": budget_boot_ci,
        "oracle_saving_ci":    oracle_saving_ci,
        "stream_order_sensitivity": {
            "mean_loss": statistics.mean(final_losses),
            "std_loss":  statistics.stdev(final_losses) if len(final_losses) > 1 else 0.0,
            "cv":        statistics.stdev(final_losses) / (statistics.mean(final_losses) + 1e-12) if len(final_losses) > 1 else 0.0,
            "n_perm": len(final_losses),
        },
    }


# ---------------------------------------------------------------------------
# A20 — Component Removal (VISTA ablation) — vectorized numpy implementation
# ---------------------------------------------------------------------------

def a20_component_removal(
    table: dict,
    rho: float = 0.20,
    alpha: float = 0.05,
    n_perm: int = 200,
    seed: int = 42,
    tau_conf: float = 0.85,
    tau_margin: float = 0.45,
) -> dict:
    """
    Full VISTA vs ablated variants.
    Vectorized: pre-computes item×budget matrices to avoid inner Python loops.

    VISTA components ablated:
      - Audit (randomized continuation)
      - IPW (inverse-probability weighting)
      - CS (anytime-valid confidence sequence monitoring)
      - Item-Adaptation (per-item threshold stopping)
      - Prequential (online policy update)
      - Feedback (label observation)
    """
    rng = random.Random(seed)
    qids   = list(table.keys())
    n      = len(qids)
    B_IDX  = {b: j for j, b in enumerate(BUDGETS)}
    nb     = len(BUDGETS)

    # Pre-compute matrices: shape (n_items, n_budgets)
    brier_mat   = np.full((n, nb), np.nan)
    maxp_mat    = np.full((n, nb), np.nan)
    marg_mat    = np.full((n, nb), np.nan)
    valid_mat   = np.zeros((n, nb), dtype=bool)

    for i, qid in enumerate(qids):
        for j, b in enumerate(BUDGETS):
            if b in table[qid]:
                r = table[qid][b]
                brier_mat[i, j] = r["brier"]
                maxp_mat[i, j]  = r["max_prob"]
                probs = r["option_probs"]
                s = sorted(probs, reverse=True)
                marg_mat[i, j]  = s[0] - s[1] if len(s) >= 2 else s[0]
                valid_mat[i, j] = True

    oracle_brier = np.nanmin(brier_mat, axis=1)    # per-item oracle Brier

    # Item-adaptive stopping assignment (vectorized):
    # For each item, first budget j where maxp >= tau_conf AND marg >= tau_margin
    stop_mask    = (maxp_mat >= tau_conf) & (marg_mat >= tau_margin) & valid_mat
    # argmax gives first True; if none, use last valid
    last_valid_j = (nb - 1 - np.argmax(valid_mat[:, ::-1], axis=1))
    item_adapt_j = np.where(
        stop_mask.any(axis=1),
        stop_mask.argmax(axis=1),
        last_valid_j,
    )
    item_adapt_brier = brier_mat[np.arange(n), item_adapt_j]

    # Fixed max-budget brier (VISTA_minus_Feedback / fallback)
    fixed_max_j  = last_valid_j
    fixed_max_brier = brier_mat[np.arange(n), fixed_max_j]

    # Static metrics for non-sequential variants
    def _static_variant(budget_j_per_item, label):
        briers = brier_mat[np.arange(n), budget_j_per_item]
        budgets_used = np.array(BUDGETS)[budget_j_per_item]
        return {
            "mean_brier":   float(np.nanmean(briers)),
            "mean_tokens":  float(np.nanmean(budgets_used)),
            "token_saving": float(1.0 - np.nanmean(budgets_used) / B_MAX),
            "mean_regret_vs_oracle": float(np.nanmean(briers - oracle_brier)),
        }

    # Sequential variants (prequential): need permutation loop, but vectorized per item
    def _prequential_regret(
        use_audit=True,
        use_ipw=True,
        use_item_adapt_override=True,
    ):
        regrets = []
        eff_rho = 1.0 if not use_audit else rho
        for _ in range(n_perm):
            perm = rng.sample(range(n), n)
            running_sum = np.zeros(nb)
            running_cnt = np.zeros(nb)
            total_loss  = 0.0
            oracle_loss = float(np.sum(oracle_brier))

            for i in perm:
                # Choose global policy from running mean
                with_data = running_cnt > 0
                if with_data.any():
                    means = np.where(with_data, running_sum / np.maximum(running_cnt, 1), np.inf)
                    preq_j = int(np.argmin(means))
                else:
                    preq_j = nb - 1  # start at max budget

                if use_item_adapt_override:
                    j_use = item_adapt_j[i]
                else:
                    j_use = preq_j if valid_mat[i, preq_j] else last_valid_j[i]

                total_loss += brier_mat[i, j_use]

                # Audit
                if rng.random() < eff_rho:
                    w = (1.0 / eff_rho) if use_ipw else 1.0
                    running_sum += np.where(valid_mat[i], brier_mat[i] * w, 0.0)
                    running_cnt += valid_mat[i].astype(float)

            regrets.append((total_loss - oracle_loss) / n)
        return {
            "mean_regret": float(statistics.mean(regrets)),
            "std_regret":  float(statistics.stdev(regrets)) if len(regrets) > 1 else 0.0,
        }

    # VISTA_Full: item-adaptive stopping + prequential update + audit + IPW
    full = _static_variant(item_adapt_j, "VISTA_Full")
    full["sequential"] = _prequential_regret(use_audit=True, use_ipw=True, use_item_adapt_override=True)

    # VISTA − Audit: no randomized continuation (rho=1 → always observe outcome)
    minus_audit = _static_variant(item_adapt_j, "VISTA_minus_Audit")
    minus_audit["sequential"] = _prequential_regret(use_audit=False, use_ipw=True, use_item_adapt_override=True)

    # VISTA − IPW: audit without weighting (biased update)
    minus_ipw = _static_variant(item_adapt_j, "VISTA_minus_IPW")
    minus_ipw["sequential"] = _prequential_regret(use_audit=True, use_ipw=False, use_item_adapt_override=True)

    # VISTA − CS: greedy selection (no CS monitoring) — static policy evaluation same as full,
    # but we report the wrong-selection risk from A01/A03 in the paper.
    # Here: represent as prequential with item-adapt but no CS guard.
    minus_cs = _static_variant(item_adapt_j, "VISTA_minus_CS")
    minus_cs["note"] = "CS removal effect quantified via A01/A03 wrong-selection rate"

    # VISTA − ItemAdapt: global prequential policy, no per-item threshold
    minus_item = _prequential_regret(use_audit=True, use_ipw=True, use_item_adapt_override=False)
    # Also compute static fixed-best
    best_j = int(np.nanargmin(np.nanmean(brier_mat, axis=0)))
    fixed_best = _static_variant(np.full(n, best_j, dtype=int), "best_fixed")
    minus_item_static = {
        "mean_brier":   float(np.nanmean(brier_mat[:, best_j])),
        "mean_tokens":  float(BUDGETS[best_j]),
        "token_saving": float(1.0 - BUDGETS[best_j] / B_MAX),
        "mean_regret_vs_oracle": float(np.nanmean(brier_mat[:, best_j] - oracle_brier)),
        "sequential": minus_item,
    }

    # VISTA − Prequential: frozen policy from budget[-1], no update
    frozen_j = nb - 1
    minus_preq = _static_variant(np.full(n, frozen_j, dtype=int), "VISTA_minus_Prequential")

    # VISTA − Feedback: no label observation → stays at max budget forever
    minus_fb = _static_variant(np.full(n, frozen_j, dtype=int), "VISTA_minus_Feedback")

    variants = {
        "VISTA_Full":              full,
        "VISTA_minus_Audit":       minus_audit,
        "VISTA_minus_IPW":         minus_ipw,
        "VISTA_minus_CS":          minus_cs,
        "VISTA_minus_ItemAdapt":   minus_item_static,
        "VISTA_minus_Prequential": minus_preq,
        "VISTA_minus_Feedback":    minus_fb,
        "Fixed_BestGlobal":        fixed_best,
        "Fixed_Bmax":              _static_variant(np.full(n, nb-1, dtype=int), "Fixed_Bmax"),
    }

    return {
        "variants": variants,
        "tau_conf":   tau_conf,
        "tau_margin": tau_margin,
        "rho":        rho,
        "n_perm":     n_perm,
        "n_items":    n,
        "oracle_mean_brier": float(np.nanmean(oracle_brier)),
    }


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def _meta(ablation_id: str, model_tag: str, ds_tag: str, **kwargs) -> dict:
    return {
        "_meta": {
            "ablation":   ablation_id,
            "model_tag":  model_tag,
            "ds_tag":     ds_tag,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
    }


def save_result(ablation_id: str, data: dict, suffix: str = ""):
    fname = f"a{ablation_id}{suffix}.json"
    path  = RESULTS_DIR / fname
    json.dump(data, open(path, "w"), indent=2)
    print(f"  → saved {path.relative_to(ROOT)}", flush=True)
    return path


# ---------------------------------------------------------------------------
# Master ablation runner
# ---------------------------------------------------------------------------

def run_campaign(
    ablation_ids: list[int],
    cell_filter: list[str] | None,
    n_perm: int,
    n_boot: int,
    verbose: bool = True,
):
    # Load all cells
    cells_to_run = ALL_CELLS
    if cell_filter:
        cells_to_run = [c for c in ALL_CELLS if f"{c[0]}_{c[1]}" in cell_filter]

    print(f"\nLoading {len(cells_to_run)} cells...", flush=True)
    tables = {}
    for (m, d, ds_name, model_name, gpu) in cells_to_run:
        key = f"{m}_{d}"
        try:
            tables[key] = load_cell(m, d)
            print(f"  {key}: {len(tables[key])} items", flush=True)
        except FileNotFoundError as e:
            print(f"  {key}: MISSING — {e}", flush=True)

    if not tables:
        print("No cells loaded. Aborting.", flush=True)
        return

    # Helper: tables keyed by model, then ds
    tables_by_model_ds: dict[tuple, dict] = {}
    for (m, d, *_) in cells_to_run:
        k = f"{m}_{d}"
        if k in tables:
            tables_by_model_ds[(m, d)] = tables[k]

    tables_m1 = {d: tables_by_model_ds[(m, d)] for (m, d) in tables_by_model_ds if m == "m1"}
    tables_m2 = {d: tables_by_model_ds[(m, d)] for (m, d) in tables_by_model_ds if m == "m2"}

    master = {}   # ablation_id → {cell_key → result}

    def _run_per_cell(aid: str, fn, cells_list=None):
        if aid not in [str(i) for i in ablation_ids]:
            return
        print(f"\n=== A{aid} ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in (cells_list or cells_to_run):
            key = f"{m}_{d}"
            if key not in tables:
                print(f"  [{key}] SKIP (not loaded)", flush=True)
                continue
            t0 = time.time()
            r = fn(tables[key])
            elapsed = time.time() - t0
            r.update(_meta(aid, m, d, ds_name=ds_name, model=model_name, elapsed_s=round(elapsed, 1)))
            res[key] = r
            print(f"  [{key}] {elapsed:.1f}s", flush=True)
        master[aid] = res
        save_result(aid, res)

    # A01
    if 1 in ablation_ids:
        print(f"\n=== A01 — IPW vs Naive ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a01_ipw_vs_naive(tables[key], n_perm=n_perm)
            r.update(_meta("01", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["01"] = res
        save_result("01", res)

    # A02
    if 2 in ablation_ids:
        print(f"\n=== A02 — Counterfactual Missingness ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a02_counterfactual_missingness(tables[key], n_perm=n_perm)
            r.update(_meta("02", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["02"] = res
        save_result("02", res)

    # A03
    if 3 in ablation_ids:
        print(f"\n=== A03 — CS vs Greedy ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a03_cs_vs_greedy(tables[key], n_perm=n_perm)
            r.update(_meta("03", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["03"] = res
        save_result("03", res)

    # A04
    if 4 in ablation_ids:
        print(f"\n=== A04 — Audit Rate Sensitivity ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a04_audit_rate_sensitivity(tables[key], n_perm=n_perm)
            r.update(_meta("04", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["04"] = res
        save_result("04", res)

    # A05
    if 5 in ablation_ids:
        print(f"\n=== A05 — Objective Divergence ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            r = a05_objective_divergence(tables[key])
            r.update(_meta("05", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] best_acc=B{r['best_budget_accuracy']} "
                  f"best_brier=B{r['best_budget_brier']} "
                  f"best_nll=B{r['best_budget_nll']} "
                  f"diverge={'YES' if r['acc_brier_diverge'] else 'NO'}", flush=True)
        master["05"] = res
        save_result("05", res)

    # A06
    if 6 in ablation_ids:
        print(f"\n=== A06 — Prefix Signal Ablation ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            try:
                t0 = time.time()
                r = a06_prefix_signal_proper(m, d)
                r.update(_meta("06", m, d, ds_name=ds_name))
                res[key] = r
                print(f"  [{key}] gate1_pass={r.get('gate1_pass')} {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"  [{key}] ERROR: {e}", flush=True)
                res[key] = {"error": str(e)}
        master["06"] = res
        save_result("06", res)

    # A07
    if 7 in ablation_ids:
        print(f"\n=== A07 — Signal Combination ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a07_signal_combination(tables[key])
            r.update(_meta("07", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["07"] = res
        save_result("07", res)

    # A08
    if 8 in ablation_ids:
        print(f"\n=== A08 — Global vs Item-Adaptive ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a08_global_vs_item_adaptive(tables[key])
            r.update(_meta("08", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] oracle_saving={r['oracle_token_saving']:.3f} {time.time()-t0:.1f}s", flush=True)
        master["08"] = res
        save_result("08", res)

    # A09
    if 9 in ablation_ids:
        print(f"\n=== A09 — Matched-Quality Policy Comparison ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a09_matched_quality(tables[key])
            r.update(_meta("09", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["09"] = res
        save_result("09", res)

    # A10
    if 10 in ablation_ids:
        print(f"\n=== A10 — Lambda Sensitivity ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a10_lambda_sensitivity(tables[key])
            r.update(_meta("10", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["10"] = res
        save_result("10", res)

    # A11
    if 11 in ablation_ids:
        print(f"\n=== A11 — Prequential vs Fixed Calibration ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a11_prequential_vs_fixed(tables[key], n_perm=n_perm)
            r.update(_meta("11", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["11"] = res
        save_result("11", res)

    # A12
    if 12 in ablation_ids:
        print(f"\n=== A12 — Distribution Shift ===", flush=True)
        shift_configs = [
            {"label": "ARC→MMLU-Pro",   "phase1": "ds2", "phase2": "ds1", "split": 0.5},
            {"label": "MMLU-Pro→MedMCQA", "phase1": "ds1", "phase2": "ds3", "split": 0.5},
            {"label": "MedMCQA→MedQA",  "phase1": "ds3", "phase2": "ds4", "split": 0.5},
            {"label": "General→Medical", "phase1": "ds1", "phase2": "ds3", "split": 0.5},
        ]
        # For M1
        if tables_m1:
            t0 = time.time()
            r = a12_distribution_shift(tables_m1, shift_configs, n_perm=min(n_perm, 100))
            r.update(_meta("12", "m1", "all", note="M1 distribution shift"))
            master.setdefault("12", {})["m1"] = r
            save_result("12", master["12"])
            print(f"  [m1_all] {time.time()-t0:.1f}s", flush=True)
        # For M2
        if tables_m2:
            t0 = time.time()
            r = a12_distribution_shift(tables_m2, shift_configs, n_perm=min(n_perm, 100))
            r.update(_meta("12", "m2", "all", note="M2 distribution shift"))
            master.setdefault("12", {})["m2"] = r
            save_result("12", master["12"])
            print(f"  [m2_all] {time.time()-t0:.1f}s", flush=True)

    # A13
    if 13 in ablation_ids:
        print(f"\n=== A13 — Cross-Model Transfer ===", flush=True)
        common_ds = [d for d in ["ds1", "ds2", "ds3", "ds4"]
                     if d in tables_m1 and d in tables_m2]
        t0 = time.time()
        r = a13_cross_model_transfer(tables_m1, tables_m2, common_ds)
        r.update(_meta("13", "m1+m2", "all"))
        master["13"] = r
        save_result("13", r)
        print(f"  [done] {time.time()-t0:.1f}s", flush=True)

    # A14
    if 14 in ablation_ids:
        print(f"\n=== A14 — Cross-Dataset Transfer ===", flush=True)
        # Per model
        res = {}
        for model_tag, t_model in [("m1", tables_m1), ("m2", tables_m2)]:
            if not t_model: continue
            t0 = time.time()
            r = a14_cross_dataset_transfer(t_model)
            r["model"] = model_tag
            r.update(_meta("14", model_tag, "all"))
            res[model_tag] = r
            print(f"  [{model_tag}] {time.time()-t0:.1f}s", flush=True)
        master["14"] = res
        save_result("14", res)

    # A15 — BLOCKED
    if 15 in ablation_ids:
        print(f"\n=== A15 — BLOCKED (GPU required) ===", flush=True)
        master["15"] = a15_blocked()
        save_result("15", master["15"])

    # A16
    if 16 in ablation_ids:
        print(f"\n=== A16 — Delayed Label Latency ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a16_delayed_label_latency(tables[key], n_perm=min(n_perm, 100))
            r.update(_meta("16", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["16"] = res
        save_result("16", res)

    # A17
    if 17 in ablation_ids:
        print(f"\n=== A17 — Policy Family Size ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run[:2]:   # representative cells
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a17_policy_family_size(tables[key], n_perm=min(n_perm, 100))
            r.update(_meta("17", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["17"] = res
        save_result("17", res)

    # A18
    if 18 in ablation_ids:
        print(f"\n=== A18 — Audit Schedule ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a18_audit_schedule(tables[key], n_perm=n_perm)
            r.update(_meta("18", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["18"] = res
        save_result("18", res)

    # A19
    if 19 in ablation_ids:
        print(f"\n=== A19 — Bootstrap / Random-Order Robustness ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a19_bootstrap_robustness(tables[key], n_boot=n_boot, n_perm=min(n_perm, 500))
            r.update(_meta("19", m, d, ds_name=ds_name))
            res[key] = r
            print(f"  [{key}] {time.time()-t0:.1f}s", flush=True)
        master["19"] = res
        save_result("19", res)

    # A20
    if 20 in ablation_ids:
        print(f"\n=== A20 — Component Removal ===", flush=True)
        res = {}
        for (m, d, ds_name, model_name, gpu) in cells_to_run:
            key = f"{m}_{d}"
            if key not in tables: continue
            t0 = time.time()
            r = a20_component_removal(tables[key], n_perm=n_perm)
            r.update(_meta("20", m, d, ds_name=ds_name))
            res[key] = r
            full_v = r['variants']['VISTA_Full']
            full_regret = full_v.get('sequential', {}).get('mean_regret',
                          full_v.get('mean_regret_vs_oracle', float('nan')))
            print(f"  [{key}] full_regret={full_regret:.4f} "
                  f"{time.time()-t0:.1f}s", flush=True)
        master["20"] = res
        save_result("20", res)

    # Save master index
    master_path = RESULTS_DIR / "campaign_master.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cells_run":  [f"{m}_{d}" for (m, d, *_) in cells_to_run if f"{m}_{d}" in tables],
        "ablations_completed": sorted(master.keys()),
        "n_perm": n_perm,
        "n_boot": n_boot,
    }
    json.dump(summary, open(master_path, "w"), indent=2)
    print(f"\nCampaign summary → {master_path.relative_to(ROOT)}", flush=True)
    return master


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="VISTA 20-Ablation Campaign")
    ap.add_argument("--ablations", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
                    help="Comma-separated ablation numbers to run (default: all)")
    ap.add_argument("--cells", default=None,
                    help="Comma-separated cell keys (e.g. m1_ds1,m2_ds1). Default: all.")
    ap.add_argument("--n-perm", type=int, default=200, help="Stream permutations (default 200)")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000)")
    ap.add_argument("--fast", action="store_true",
                    help="Use n_perm=50 for quick smoke test")
    args = ap.parse_args()

    ablation_ids = [int(x) for x in args.ablations.split(",")]
    cell_filter  = args.cells.split(",") if args.cells else None
    n_perm = 50 if args.fast else args.n_perm
    n_boot = 100 if args.fast else args.n_boot

    print("VISTA Ablation Campaign", flush=True)
    print(f"  Ablations: {ablation_ids}", flush=True)
    print(f"  Cells: {cell_filter or 'all'}", flush=True)
    print(f"  n_perm={n_perm}  n_boot={n_boot}", flush=True)

    t0 = time.time()
    run_campaign(ablation_ids, cell_filter, n_perm, n_boot)
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
