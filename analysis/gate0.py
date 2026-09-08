"""
Gate 0 analysis: three oracle definitions and compute-saving thresholds.

Oracle taxonomy
---------------
Oracle A — Hindsight minimum-Brier oracle (hindsight_oracle_budget):
    Per-item: minimum budget b where Brier(b) == min_b Brier(b).
    Measures: quality upper bound and descriptive budget opportunity.
    Saving: 1 - mean_oracle / b_max  (relative to always-max baseline).
    NOT a matched-quality claim; quality improves beyond best_fixed.

Oracle B — Per-item threshold oracle (per_item_threshold_oracle):
    Per-item: minimum budget b where Brier(b) <= global_best_fixed_brier.
    Stronger per-item constraint; saving relative to best_fixed_budget.
    Can be negative when best_fixed is already the minimum budget.

Oracle C — Constrained matched-mean-quality oracle (constrained_mean_oracle):
    Minimize: mean_i budget_i
    Subject to: mean_i S_i,budget_i <= best_fixed_brier
    budget_i in {256,512,1024,2048,4096,8192}
    Primary Gate 0 opportunity metric. Solved via Lagrangian relaxation
    with binary search on the dual variable.
    Saving relative to best_fixed_budget; always >= 0 (best_fixed feasible).

Gate 0 PASS criterion (Oracle C):
    saving_vs_best_fixed >= 10%.
    Cells where best_fixed == min(budgets) trivially FAIL (saving == 0).
"""

from __future__ import annotations

import statistics
from analysis.metrics import brier
from analysis.data_loader import BUDGETS as DEFAULT_BUDGETS


# ---------------------------------------------------------------------------
# Oracle A helpers
# ---------------------------------------------------------------------------

def hindsight_oracle_budget(
    bmap: dict[int, dict],
    budgets: list[int],
) -> int:
    """
    Oracle A: minimum budget b where Brier(b) == min over all budgets.
    If the item is missing some budgets, operate only on present ones.
    """
    present = sorted(b for b in budgets if b in bmap)
    if not present:
        raise ValueError('Item has no budget data')
    briers = {b: brier(bmap[b]['option_probs'], bmap[b]['answer_index']) for b in present}
    best_brier = min(briers.values())
    tol = 1e-9
    for b in present:
        if abs(briers[b] - best_brier) <= tol:
            return b
    return present[-1]


def gate0_analysis(
    item_table: dict[str, dict[int, dict]],
    budgets: list[int] | None = None,
    b_max: int = 8192,
    saving_threshold: float = 0.10,
) -> dict:
    """
    Oracle A analysis: hindsight minimum-Brier oracle.

    Saving = 1 - mean_oracle / b_max  (relative to always-max baseline).
    This is a DESCRIPTIVE OPPORTUNITY figure, not matched-quality.

    Returns dict with oracle_budgets, mean_oracle, saving_vs_max,
    gate0_pass (saving_vs_max >= saving_threshold), n_items,
    budget_distribution.
    """
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    oracle_budgets: list[int] = []
    budget_dist: dict[int, int] = {b: 0 for b in budgets}

    for bmap in item_table.values():
        ob = hindsight_oracle_budget(bmap, budgets)
        oracle_budgets.append(ob)
        budget_dist[ob] = budget_dist.get(ob, 0) + 1

    if not oracle_budgets:
        return {'n_items': 0, 'gate0_pass': False}

    mean_ob   = statistics.mean(oracle_budgets)
    median_ob = statistics.median(oracle_budgets)
    p90_ob    = _percentile(oracle_budgets, 0.90)
    saving    = 1.0 - mean_ob / b_max

    return {
        'oracle_budgets':      oracle_budgets,
        'mean_oracle':         mean_ob,
        'median_oracle':       median_ob,
        'p90_oracle':          p90_ob,
        'saving_vs_max':       saving,
        'gate0_pass':          saving >= saving_threshold,
        'saving_threshold':    saving_threshold,
        'n_items':             len(oracle_budgets),
        'budget_distribution': dict(sorted(budget_dist.items())),
    }


# ---------------------------------------------------------------------------
# Oracle B helper
# ---------------------------------------------------------------------------

def per_item_threshold_oracle(
    item_table: dict[str, dict[int, dict]],
    budgets: list[int] | None = None,
    threshold: float | None = None,
) -> dict:
    """
    Oracle B: per-item minimum budget where individual Brier <= threshold.

    threshold defaults to best_fixed_mean_brier (min over b of mean Brier).
    Saving is relative to the best-fixed budget.
    Can be negative when best_fixed == min(budgets) and some items need
    higher budgets to meet the per-item threshold.
    """
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    # Compute best_fixed_brier if threshold not given
    if threshold is None:
        budget_means: dict[int, list[float]] = {b: [] for b in budgets}
        for bmap in item_table.values():
            for b in budgets:
                if b in bmap:
                    s = brier(bmap[b]['option_probs'], bmap[b]['answer_index'])
                    budget_means[b].append(s)
        mean_by_b = {b: sum(v) / len(v) for b, v in budget_means.items() if v}
        best_fixed_b = min(mean_by_b, key=mean_by_b.get)
        threshold = mean_by_b[best_fixed_b]
    else:
        mean_by_b = {}
        for b in budgets:
            vals = [brier(bmap[b]['option_probs'], bmap[b]['answer_index'])
                    for bmap in item_table.values() if b in bmap]
            if vals:
                mean_by_b[b] = sum(vals) / len(vals)
        best_fixed_b = min(mean_by_b, key=mean_by_b.get)

    oracle_budgets: list[int] = []
    cnt_below = 0

    for bmap in item_table.values():
        assigned = best_fixed_b
        for b in sorted(b for b in budgets if b in bmap):
            s = brier(bmap[b]['option_probs'], bmap[b]['answer_index'])
            if s <= threshold + 1e-9:
                assigned = b
                break
        oracle_budgets.append(assigned)
        if assigned < best_fixed_b:
            cnt_below += 1

    mean_ob = statistics.mean(oracle_budgets)
    saving = 1.0 - mean_ob / best_fixed_b

    return {
        'oracle_budgets':          oracle_budgets,
        'mean_oracle_budget':      mean_ob,
        'threshold':               threshold,
        'best_fixed_budget':       best_fixed_b,
        'saving_vs_best_fixed':    saving,
        'pct_items_below_bfixed':  cnt_below / len(oracle_budgets) * 100 if oracle_budgets else 0.0,
        'n_items':                 len(oracle_budgets),
    }


# ---------------------------------------------------------------------------
# Oracle C — Constrained Matched-Mean-Quality Oracle (Gate 0 primary)
# ---------------------------------------------------------------------------

def constrained_mean_oracle(
    item_table: dict[str, dict[int, dict]],
    budgets: list[int] | None = None,
    saving_threshold: float = 0.10,
) -> dict:
    """
    Oracle C: primary Gate 0 metric.

    Solve:
        minimize   mean_i budget_i
        subject to mean_i S_i,budget_i <= best_fixed_brier
                   budget_i in budgets  (discrete)

    where best_fixed_brier = min_b mean_i S_i,b.

    Solved via Lagrangian relaxation with binary search on the dual variable
    lambda. For given lambda, each item independently chooses
        b*_i(lambda) = argmin_b [b + lambda * S_i,b].
    Binary search finds the smallest lambda at which mean_i S_i,b*_i <= target.

    best_fixed_budget is always feasible, so saving_vs_best_fixed >= 0.
    Cells where best_fixed_budget == min(budgets) have saving == 0 (FAIL).

    Returns
    -------
    best_fixed_budget       : int
    best_fixed_mean_brier   : float
    oracle_mean_budget      : float
    oracle_mean_brier       : float
    saving_vs_best_fixed    : float in [0, 1)
    gate0_pass              : bool (saving >= saving_threshold)
    budget_distribution     : {b: count}
    n_items                 : int
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError('numpy required for constrained_mean_oracle')

    if budgets is None:
        budgets = DEFAULT_BUDGETS

    qids = list(item_table.keys())
    N = len(qids)
    if N == 0:
        return {'n_items': 0, 'gate0_pass': False}

    # Build S[i, j] = Brier(item i, budget j)
    S = []
    for qid in qids:
        bmap = item_table[qid]
        row = []
        for b in budgets:
            if b in bmap:
                row.append(brier(bmap[b]['option_probs'], bmap[b]['answer_index']))
            else:
                row.append(float('inf'))
        S.append(row)
    S = np.array(S)           # (N, K)
    B = np.array(budgets, dtype=float)  # (K,)

    # Best fixed budget (minimises population mean Brier)
    mean_by_b = S.mean(axis=0)  # (K,)
    best_b_idx = int(np.argmin(mean_by_b))
    best_fixed_b   = int(budgets[best_b_idx])
    best_fixed_brier = float(mean_by_b[best_b_idx])

    # If best_fixed is already minimum budget, saving is trivially 0
    if best_fixed_b == min(budgets):
        return {
            'best_fixed_budget':     best_fixed_b,
            'best_fixed_mean_brier': best_fixed_brier,
            'oracle_mean_budget':    float(best_fixed_b),
            'oracle_mean_brier':     best_fixed_brier,
            'saving_vs_best_fixed':  0.0,
            'gate0_pass':            False,
            'budget_distribution':   {best_fixed_b: N},
            'n_items':               N,
        }

    def lagrangian_solve(lam: float):
        obj = B[np.newaxis, :] + lam * S      # (N, K)
        cj  = np.argmin(obj, axis=1)           # (N,)
        return B[cj], S[np.arange(N), cj], cj

    # Binary search: find smallest lam where mean_brier <= best_fixed_brier
    lo, hi = 0.0, 1e8
    _, cs_hi, _ = lagrangian_solve(hi)
    if float(cs_hi.mean()) > best_fixed_brier + 1e-8:
        # Hindsight oracle cannot reach target (degenerate case)
        chosen_b = np.full(N, best_fixed_b, dtype=float)
        chosen_s = S[:, best_b_idx]
        budget_dist = {best_fixed_b: N}
    else:
        for _ in range(80):
            mid = (lo + hi) / 2
            _, cs_mid, _ = lagrangian_solve(mid)
            if float(cs_mid.mean()) <= best_fixed_brier + 1e-10:
                hi = mid
            else:
                lo = mid

        chosen_b, chosen_s, choices_j = lagrangian_solve(hi)

        # Clamp numerical violations
        if float(chosen_s.mean()) > best_fixed_brier + 1e-7:
            chosen_b = np.full(N, best_fixed_b, dtype=float)
            chosen_s = S[:, best_b_idx]

        budget_dist = {}
        for bv in budgets:
            cnt = int((chosen_b == bv).sum())
            if cnt > 0:
                budget_dist[int(bv)] = cnt

    mean_budget = float(chosen_b.mean())
    mean_brier_result = float(chosen_s.mean())
    saving = max(0.0, 1.0 - mean_budget / best_fixed_b)

    return {
        'best_fixed_budget':     best_fixed_b,
        'best_fixed_mean_brier': best_fixed_brier,
        'oracle_mean_budget':    mean_budget,
        'oracle_mean_brier':     mean_brier_result,
        'saving_vs_best_fixed':  saving,
        'gate0_pass':            saving >= saving_threshold,
        'budget_distribution':   budget_dist,
        'n_items':               N,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _percentile(data: list[float | int], q: float) -> float:
    s = sorted(data)
    n = len(s)
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac
