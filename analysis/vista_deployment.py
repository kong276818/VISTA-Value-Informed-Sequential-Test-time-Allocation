"""
VISTA full deployment simulation engine.

Global VISTA protocol
---------------------
Policy space: Fixed budget policies (one per grid point) + confidence/entropy/
margin threshold policies.  Reference = Fixed@b_ref (best fixed budget for cell).

1. Start with reference policy π_0 = Fixed@b_ref.
2. For each arriving item t:
   a. Apply current active policy π_t to select stopping budget b_i.
   b. Draw A_i ~ Bernoulli(rho).
   c. Actual compute = B_MAX if audited, else b_i.
   d. Deploy answer at b_i.
   e. Delayed label arrives.
   f. If audited: for every candidate policy π, compute its budget b^π_i from
      the stored trajectory; form IPW difference (l^π_i - l^ref_i)/rho;
      update the Welford running stats for π's CS.
   g. Certify cheapest candidate whose CS upper bound < 0 as new active policy.
3. Aggregate over permutations.

Compute accounting (exact):
  policy_compute  = sum_i b_i           (policy budget for each item)
  audit_extra     = sum_{audited i} (B_MAX - b_i)
  total_compute   = policy_compute + audit_extra
  net_saving      = 1 - mean_total_per_item / b_ref
  policy_saving   = 1 - mean_policy_per_item / b_ref  (audit-free policy saving)

CS implementation:
  Time-uniform empirical Bernstein (Howard et al. 2021), Bonferroni-corrected.
  Incremental (Welford) updates: O(1) per observation, O(1) CS bound computation.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Callable


BUDGETS  = [256, 512, 1024, 2048, 4096, 8192]
B_MAX    = 8192
BRIER_RANGE = 2.0


# ---------------------------------------------------------------------------
# Welford online running statistics (exact, no Fraction arithmetic)
# ---------------------------------------------------------------------------

class RunningStats:
    """
    Welford's online algorithm for mean and variance of a stream.
    O(1) update and O(1) query.
    """
    __slots__ = ("n", "_mean", "_M2")

    def __init__(self):
        self.n    = 0
        self._mean = 0.0
        self._M2   = 0.0

    def update(self, x: float) -> None:
        self.n    += 1
        delta      = x - self._mean
        self._mean += delta / self.n
        delta2     = x - self._mean
        self._M2  += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._M2 / (self.n - 1) if self.n > 1 else 0.0


# ---------------------------------------------------------------------------
# Empirical Bernstein CS (time-uniform, Bonferroni)
# ---------------------------------------------------------------------------

def _cs_upper_bound_welf(rs: RunningStats, c: float) -> float:
    """
    One-sided upper CS bound: mean + c * sqrt(var / n).
    c = sqrt(2 * log(2 * n_pi / alpha)) — precomputed per experiment.
    This is the formula used in run_ablations.py (ablation_a2_anytime_cs).
    It is the variance-adaptive Bonferroni-corrected confidence bound.
    The bound is NOT time-uniform in the strict Howard sense; it is a
    fixed-n normal approximation with union-bound correction over n_pi policies.
    """
    n = rs.n
    if n < 2:
        return float("inf")
    return rs.mean + c * math.sqrt(rs.var / n)


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

Policy = tuple[str, Callable[[dict], int]]


def _fixed_policy(b_fixed: int) -> Policy:
    def fn(bmap: dict, _b: int = b_fixed) -> int:
        return _b if _b in bmap else max(b for b in BUDGETS if b in bmap)
    return f"Fixed@{b_fixed}", fn


def _threshold_policy(signal: str, tau: float, direction: str) -> Policy:
    abbrev = signal[:4]
    label  = f"{abbrev}≥{tau}" if direction == "ge" else f"{abbrev}≤{tau}"

    def fn(bmap: dict, _s=signal, _t=tau, _d=direction) -> int:
        for b in BUDGETS:
            if b not in bmap:
                continue
            v = bmap[b][_s]
            if (_d == "ge" and v >= _t) or (_d == "le" and v <= _t):
                return b
        return max(b for b in BUDGETS if b in bmap)

    return label, fn


def build_policy_space(b_ref: int) -> list[Policy]:
    """Reference (Fixed@b_ref) at index 0; all other policies follow."""
    ref_label, ref_fn = _fixed_policy(b_ref)
    policies: list[Policy] = [(ref_label, ref_fn)]

    for b in BUDGETS:
        lbl, fn = _fixed_policy(b)
        if lbl != ref_label:
            policies.append((lbl, fn))

    for tau in [0.70, 0.80, 0.85, 0.90, 0.95]:
        policies.append(_threshold_policy("confidence", tau, "ge"))
    for tau in [0.10, 0.20, 0.30, 0.50]:
        policies.append(_threshold_policy("entropy", tau, "le"))
    for tau in [0.30, 0.50, 0.70]:
        policies.append(_threshold_policy("margin", tau, "ge"))

    return policies


# ---------------------------------------------------------------------------
# Precomputed per-policy statistics (offline; used for diagnostics and ranking)
# ---------------------------------------------------------------------------

def precompute_policy_stats(
    policies: list[Policy], item_table: dict, lambda_val: float, b_ref: int
) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for label, fn in policies:
        bud, bri, los = [], [], []
        for bmap in item_table.values():
            b = fn(bmap)
            if b not in bmap:
                b = max(bb for bb in BUDGETS if bb in bmap)
            bud.append(b)
            br = bmap[b]["brier"]
            bri.append(br)
            los.append(br + lambda_val * (b / B_MAX))
        n = len(bud)
        stats[label] = {
            "mean_budget": sum(bud) / n if n else 0.0,
            "mean_brier":  sum(bri) / n if n else float("nan"),
            "mean_loss":   sum(los) / n if n else float("nan"),
        }
    return stats


# ---------------------------------------------------------------------------
# Best fixed budget
# ---------------------------------------------------------------------------

def best_fixed_budget(item_table: dict) -> tuple[int, float]:
    best_b, best_br = None, float("inf")
    for b in BUDGETS:
        vals = [row[b]["brier"] for row in item_table.values() if b in row]
        if vals:
            m = sum(vals) / len(vals)
            if m < best_br:
                best_br = m
                best_b  = b
    return best_b, best_br


# ---------------------------------------------------------------------------
# Static policy evaluations (offline)
# ---------------------------------------------------------------------------

def evaluate_static_policy(item_table: dict, fn: Callable, b_ref: int) -> dict:
    briers, accs, nlls, tokens = [], [], [], []
    for bmap in item_table.values():
        b = fn(bmap)
        if b not in bmap:
            continue
        info = bmap[b]
        briers.append(info["brier"])
        accs.append(float(info["correct"]))
        nlls.append(info["nll"])
        tokens.append(b)
    n = len(briers)
    if n == 0:
        return {"n": 0}
    mean_tok = sum(tokens) / n
    return {
        "n":          n,
        "mean_brier": sum(briers) / n,
        "accuracy":   sum(accs)   / n,
        "mean_nll":   sum(nlls)   / n,
        "mean_tokens": mean_tok,
        "net_saving":  1.0 - mean_tok / b_ref,
    }


def all_static_baselines(item_table: dict, b_ref: int) -> dict:
    policies = build_policy_space(b_ref)
    out = {}
    for label, fn in policies:
        r = evaluate_static_policy(item_table, fn, b_ref)
        if r.get("n", 0) == 0:
            continue
        r["policy_type"] = "fixed" if label.startswith("Fixed@") else "threshold"
        out[label] = r
    return out


# ---------------------------------------------------------------------------
# One-permutation Global VISTA simulation (O(N·P) using Welford stats)
# ---------------------------------------------------------------------------

def _simulate_one_permutation(
    item_order:    list[str],
    item_table:    dict,
    policies:      list[Policy],
    policy_stats:  dict[str, dict],
    b_ref:         int,
    rho:           float,
    cs_c:          float,          # sqrt(2 * log(2 * n_pi / alpha)); precomputed
    rng:           random.Random,
    lambda_val:    float = 0.0,
) -> dict:
    ref_label, ref_fn = policies[0]
    policy_map   = {p[0]: p[1] for p in policies}
    mean_bud_map = {p[0]: policy_stats[p[0]]["mean_budget"] for p in policies}

    # Welford running stats per non-ref policy (tracking IPW diffs vs ref)
    rs_map: dict[str, RunningStats] = {p[0]: RunningStats() for p in policies[1:]}

    active_label = ref_label
    active_fn    = ref_fn

    briers_sum  = 0.0
    accs_sum    = 0.0
    nlls_sum    = 0.0
    policy_tok  = 0.0
    actual_tok  = 0.0
    n_audited   = 0
    n           = 0

    switch_times:  list[int] = []
    wrong_switches = 0
    prev_label     = ref_label

    ref_brier_oracle = policy_stats[ref_label]["mean_brier"]

    for t, qid in enumerate(item_order):
        bmap = item_table.get(qid)
        if bmap is None:
            continue
        n += 1

        # Step a
        b_chosen = active_fn(bmap)
        if b_chosen not in bmap:
            b_chosen = max(b for b in BUDGETS if b in bmap)

        # Step b
        audited = rng.random() < rho
        actual  = B_MAX if audited else b_chosen

        # Step d
        info = bmap[b_chosen]
        briers_sum += info["brier"]
        accs_sum   += float(info["correct"])
        nlls_sum   += info["nll"]
        policy_tok += b_chosen
        actual_tok += actual
        if audited:
            n_audited += 1

        # Step f: IPW update
        if audited:
            b_ref_i = ref_fn(bmap)
            if b_ref_i not in bmap:
                b_ref_i = max(b for b in BUDGETS if b in bmap)
            loss_ref = bmap[b_ref_i]["brier"] + lambda_val * (b_ref_i / B_MAX)

            for lbl, fn in policies[1:]:
                b_pi = fn(bmap)
                if b_pi not in bmap:
                    b_pi = max(b for b in BUDGETS if b in bmap)
                loss_pi = bmap[b_pi]["brier"] + lambda_val * (b_pi / B_MAX)
                rs_map[lbl].update((loss_pi - loss_ref) / rho)

        # Step g: certify cheapest policy with CS upper bound < 0
        new_label = ref_label
        best_mb   = mean_bud_map[ref_label]

        for lbl, _ in policies[1:]:
            rs = rs_map[lbl]
            if rs.n < 2:
                continue
            upper = _cs_upper_bound_welf(rs, cs_c)
            if upper < 0:
                mb = mean_bud_map[lbl]
                if mb < best_mb:
                    best_mb   = mb
                    new_label = lbl

        if new_label != prev_label:
            switch_times.append(t + 1)
            if policy_stats[new_label]["mean_brier"] > ref_brier_oracle + 1e-6:
                wrong_switches += 1

        prev_label   = new_label
        active_label = new_label
        active_fn    = policy_map[new_label]

    if n == 0:
        return {"n": 0}

    p_avg = policy_tok / n
    a_avg = actual_tok / n

    return {
        "n":              n,
        "n_audited":      n_audited,
        "mean_brier":     briers_sum / n,
        "accuracy":       accs_sum   / n,
        "mean_nll":       nlls_sum   / n,
        "policy_tokens":  p_avg,
        "audit_extra":    a_avg - p_avg,
        "actual_tokens":  a_avg,
        "net_saving":     1.0 - a_avg / b_ref,
        "policy_saving":  1.0 - p_avg / b_ref,
        "final_policy":   active_label,
        "n_switches":     len(switch_times),
        "wrong_switches": wrong_switches,
        "time_to_decide": switch_times[0] if switch_times else n,
        "b_ref":          b_ref,
    }


# ---------------------------------------------------------------------------
# Aggregate permutation results
# ---------------------------------------------------------------------------

def _s(vals: list) -> dict:
    clean = [v for v in vals if v is not None
             and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    nc = len(clean)
    if nc == 0:
        return {"mean": float("nan"), "std": 0.0,
                "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
    m  = sum(clean) / nc
    sd = math.sqrt(sum((x - m)**2 for x in clean) / max(nc - 1, 1))
    se = sd / math.sqrt(nc)
    med = sorted(clean)[nc // 2]
    return {"mean": m, "std": sd, "median": med,
            "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se, "n": nc}


def _aggregate_perms(perm_results: list[dict], b_ref: int, b_ref_brier: float) -> dict:
    ok = [r for r in perm_results if r.get("n", 0) > 0]
    if not ok:
        return {}

    def col(key):
        return [r[key] for r in ok if key in r]

    agg = {
        "mean_brier":    _s(col("mean_brier")),
        "accuracy":      _s(col("accuracy")),
        "mean_nll":      _s(col("mean_nll")),
        "policy_tokens": _s(col("policy_tokens")),
        "audit_extra":   _s(col("audit_extra")),
        "actual_tokens": _s(col("actual_tokens")),
        "net_saving":    _s(col("net_saving")),
        "policy_saving": _s(col("policy_saving")),
        "n_switches":    _s(col("n_switches")),
        "wrong_switches":_s(col("wrong_switches")),
        "time_to_decide":_s(col("time_to_decide")),
        "b_ref":         b_ref,
        "b_ref_brier":   b_ref_brier,
        "n_perms":       len(perm_results),
    }

    fp: dict[str, int] = defaultdict(int)
    for r in ok:
        fp[str(r.get("final_policy", f"Fixed@{b_ref}"))] += 1
    agg["final_policy_distribution"] = dict(fp)
    agg["modal_final_policy"] = max(fp, key=fp.get) if fp else f"Fixed@{b_ref}"

    return agg


# ---------------------------------------------------------------------------
# Full Global VISTA experiment
# ---------------------------------------------------------------------------

def run_global_vista(
    item_table:  dict,
    rho_list:    list[float] = (0.05, 0.10, 0.20, 0.40),
    lambda_list: list[float] = (0.0,),
    n_perms:     int         = 200,
    alpha:       float       = 0.05,
    seed:        int         = 42,
    b_max:       int         = B_MAX,
) -> dict:
    items  = list(item_table.keys())
    b_ref, b_ref_brier = best_fixed_budget(item_table)
    policies = build_policy_space(b_ref)

    rng = random.Random(seed)
    all_orders = []
    for _ in range(n_perms):
        order = items.copy()
        rng.shuffle(order)
        all_orders.append(order)

    results = {
        "b_ref":         b_ref,
        "b_ref_brier":   b_ref_brier,
        "n_perms":       n_perms,
        "n_policies":    len(policies),
        "policy_labels": [p[0] for p in policies],
    }

    n_pi = len(policies)
    # CS constant: c = sqrt(2 * log(2 * n_pi / alpha))
    # Implements ablation_a2 formula: certified if mean_d + c*sqrt(var_d/n) < 0
    cs_c = math.sqrt(2.0 * math.log(2.0 * n_pi / alpha))

    for rho in rho_list:
        results[rho] = {}
        for lam in lambda_list:
            p_stats  = precompute_policy_stats(policies, item_table, lam, b_ref)
            perm_rng = random.Random(seed ^ (hash((rho, lam)) & 0xFFFFFFFF))
            perm_res = []
            for order in all_orders:
                pr = _simulate_one_permutation(
                    order, item_table, policies, p_stats,
                    b_ref, rho, cs_c, perm_rng, lam)
                perm_res.append(pr)

            agg = _aggregate_perms(perm_res, b_ref, b_ref_brier)
            agg["rho"]          = rho
            agg["lambda"]       = lam
            agg["policy_stats"] = p_stats
            agg["cs_c"]         = cs_c
            results[rho][lam]   = agg

    return results


# ---------------------------------------------------------------------------
# Item-adaptive VISTA (threshold-only candidate set)
# ---------------------------------------------------------------------------

def run_item_adaptive_vista(
    item_table: dict,
    rho_list:   list[float] = (0.20,),
    n_perms:    int         = 200,
    alpha:      float       = 0.05,
    seed:       int         = 42,
    b_max:      int         = B_MAX,
) -> dict:
    b_ref, b_ref_brier = best_fixed_budget(item_table)

    thr: list[Policy] = [_fixed_policy(b_ref)]
    for tau in [0.70, 0.80, 0.85, 0.90, 0.95]:
        thr.append(_threshold_policy("confidence", tau, "ge"))
    for tau in [0.10, 0.20, 0.30, 0.50]:
        thr.append(_threshold_policy("entropy", tau, "le"))
    for tau in [0.30, 0.50, 0.70]:
        thr.append(_threshold_policy("margin", tau, "ge"))

    items = list(item_table.keys())
    rng   = random.Random(seed + 500)
    all_orders = []
    for _ in range(n_perms):
        order = items.copy()
        rng.shuffle(order)
        all_orders.append(order)

    n_pi_thr = len(thr)
    cs_c_thr = math.sqrt(2.0 * math.log(2.0 * n_pi_thr / alpha))
    p_stats = precompute_policy_stats(thr, item_table, 0.0, b_ref)
    results = {"b_ref": b_ref, "b_ref_brier": b_ref_brier}

    for rho in rho_list:
        perm_rng = random.Random(seed + 600 + int(rho * 1000))
        perm_res = []
        for order in all_orders:
            pr = _simulate_one_permutation(
                order, item_table, thr, p_stats,
                b_ref, rho, cs_c_thr, perm_rng, 0.0)
            perm_res.append(pr)
        agg = _aggregate_perms(perm_res, b_ref, b_ref_brier)
        agg["rho"] = rho
        results[rho] = agg

    return results


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_paired_ci(
    brier_a:  list[float],
    brier_b:  list[float],
    tokens_a: list[float],
    tokens_b: list[float],
    n_boot:   int   = 1000,
    seed:     int   = 42,
    alpha:    float = 0.05,
) -> dict:
    n   = len(brier_a)
    rng = random.Random(seed)
    bd, td = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bd.append(sum(brier_a[i] for i in idx) / n -
                  sum(brier_b[i] for i in idx) / n)
        td.append(sum(tokens_a[i] for i in idx) / n -
                  sum(tokens_b[i] for i in idx) / n)

    def pci(diffs):
        diffs.sort()
        lo = diffs[max(0, int(alpha / 2 * n_boot))]
        hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)]
        m  = diffs[n_boot // 2]
        return {"diff": m, "lo": lo, "hi": hi, "significant": not (lo <= 0 <= hi)}

    return {"brier_diff": pci(bd), "token_diff": pci(td), "n": n}
