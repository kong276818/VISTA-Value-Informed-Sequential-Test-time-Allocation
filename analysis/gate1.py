"""
Gate 1: prefix signal predictability.

For each adjacent budget edge (b_lo → b_hi) and each signal, compute
  - Spearman correlation between signal at b_lo and Brier-improvement at b_hi
  - AUC of signal predicting "improvement" (Brier at b_hi < Brier at b_lo)

Signals (computed from b_lo distribution):
  - confidence:       max(probs)
  - entropy:          H(probs)
  - margin:           top1 - top2
  - answer_stability: 1 if argmax(b_lo)==argmax(b_prev), else 0  [NaN for first edge]
  - conf_slope:       confidence(b_lo) - confidence(b_prev)       [NaN for first edge]
  - prob_slope:       max_prob(b_lo) - max_prob(b_prev)           [= conf_slope; alias]

"Improvement" label: Brier(b_hi) < Brier(b_lo) - eps
G_{i,j} = Brier(b_lo) - Brier(b_hi)  [positive = improvement]
"""

from __future__ import annotations

import math
import statistics
from analysis.metrics import brier, confidence, entropy, margin


# Signals computable from a single budget level
SIGNALS_SINGLE: dict[str, callable] = {
    'confidence': confidence,
    'entropy':    entropy,
    'margin':     margin,
}

# Legacy alias for code that imports SIGNALS
SIGNALS = SIGNALS_SINGLE


def _spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return float('nan')
    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _rank(x: list[float]) -> list[float]:
    """Convert values to average ranks (1-based)."""
    indexed = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and x[indexed[j]] == x[indexed[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def _auc_binary(signal: list[float], label: list[bool]) -> float:
    """
    AUC of signal predicting binary label.
    signal: higher = predicts positive label.
    Returns AUC in [0,1]; 0.5 = random.
    """
    pairs = list(zip(signal, label))
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return float('nan')
    concordant = sum(sp > sn for sp in pos for sn in neg)
    tied       = sum(sp == sn for sp in pos for sn in neg)
    total      = len(pos) * len(neg)
    return (concordant + 0.5 * tied) / total


def gate1_edge(
    item_table: dict[str, dict[int, dict]],
    b_lo: int,
    b_hi: int,
    b_prev: int | None = None,
    eps: float = 1e-6,
) -> dict:
    """
    Compute Gate 1 metrics for one adjacent budget edge (b_lo → b_hi).

    b_prev: the budget level just before b_lo (None for the first edge).
            Used for answer_stability, conf_slope, prob_slope.

    Returns dict with:
      Per-signal sub-dicts:  {auc, spearman, n}
      Aggregate G stats:     frac_G_pos, frac_G_neg, mean_G, median_G
      n: number of valid items

    G_{i,j} = Brier(b_lo) - Brier(b_hi)  [positive = improvement]
    """
    signals_vals: dict[str, list[float]] = {k: [] for k in SIGNALS_SINGLE}
    # Cross-budget signals (require b_prev)
    stab_vals:       list[float] = []
    conf_slope_vals: list[float] = []
    prob_slope_vals: list[float] = []

    delta_briers: list[float] = []   # Brier(b_hi) - Brier(b_lo); negative = improvement
    labels: list[bool] = []           # True if Brier decreased (G > 0)

    for bmap in item_table.values():
        if b_lo not in bmap or b_hi not in bmap:
            continue
        r_lo = bmap[b_lo]
        r_hi = bmap[b_hi]
        b_lo_val = brier(r_lo['option_probs'], r_lo['answer_index'])
        b_hi_val = brier(r_hi['option_probs'], r_hi['answer_index'])
        delta = b_hi_val - b_lo_val

        for sname, sfn in SIGNALS_SINGLE.items():
            signals_vals[sname].append(sfn(r_lo['option_probs']))

        # Cross-budget signals
        if b_prev is not None and b_prev in bmap:
            r_prev = bmap[b_prev]
            argmax_lo   = r_lo['option_probs'].index(max(r_lo['option_probs']))
            argmax_prev = r_prev['option_probs'].index(max(r_prev['option_probs']))
            stab_vals.append(1.0 if argmax_lo == argmax_prev else 0.0)
            conf_lo   = confidence(r_lo['option_probs'])
            conf_pr   = confidence(r_prev['option_probs'])
            conf_slope_vals.append(conf_lo - conf_pr)
            prob_slope_vals.append(conf_lo - conf_pr)   # max_prob = confidence

        delta_briers.append(delta)
        labels.append(delta < -eps)

    n = len(delta_briers)
    G_vals = [-d for d in delta_briers]   # G = Brier(b_lo) - Brier(b_hi)

    result = {
        'n':          n,
        'frac_G_pos': sum(g > eps for g in G_vals) / n if n > 0 else float('nan'),
        'frac_G_neg': sum(g < -eps for g in G_vals) / n if n > 0 else float('nan'),
        'mean_G':     statistics.mean(G_vals) if G_vals else float('nan'),
        'median_G':   statistics.median(G_vals) if G_vals else float('nan'),
    }

    neg_delta = G_vals  # = [-d for d in delta_briers]

    for sname, svals in signals_vals.items():
        if sname == 'entropy':
            auc_signal = svals           # high entropy → predicts improvement
        else:
            auc_signal = [-s for s in svals]  # high conf/margin → less improvement

        result[sname] = {
            'auc':      _auc_binary(auc_signal, labels),
            'spearman': _spearman(svals, neg_delta),
            'n':        n,
        }

    # Cross-budget signals (b_prev-dependent)
    if stab_vals:
        ns = len(stab_vals)
        stab_labels = labels[:ns]
        result['answer_stability'] = {
            'auc':      _auc_binary([-s for s in stab_vals], stab_labels),
            'spearman': _spearman(stab_vals, neg_delta[:ns]),
            'n':        ns,
        }
        result['conf_slope'] = {
            'auc':      _auc_binary(conf_slope_vals, stab_labels),
            'spearman': _spearman(conf_slope_vals, neg_delta[:ns]),
            'n':        ns,
        }
        result['prob_slope'] = {
            'auc':      _auc_binary(prob_slope_vals, stab_labels),
            'spearman': _spearman(prob_slope_vals, neg_delta[:ns]),
            'n':        ns,
        }
    else:
        for sname in ('answer_stability', 'conf_slope', 'prob_slope'):
            result[sname] = {'auc': float('nan'), 'spearman': float('nan'), 'n': 0}

    return result


def gate1_full(
    item_table: dict[str, dict[int, dict]],
    budgets: list[int] | None = None,
) -> dict:
    """
    Compute Gate 1 metrics for all adjacent budget edges.

    Returns dict keyed by (b_lo, b_hi) tuple.
    Each value includes per-signal {auc, spearman} plus G aggregate stats.
    """
    from analysis.data_loader import BUDGETS as DEFAULT_BUDGETS
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    result = {}
    for i in range(len(budgets) - 1):
        b_lo, b_hi = budgets[i], budgets[i + 1]
        b_prev = budgets[i - 1] if i > 0 else None
        result[(b_lo, b_hi)] = gate1_edge(item_table, b_lo, b_hi, b_prev=b_prev)
    return result


def gate1_pass(gate1_result: dict) -> bool:
    """
    Gate 1 PASS criterion:  AUC >= 0.60  OR  |Spearman rho| >= 0.30
    for ANY signal on ANY edge.
    """
    all_signals = ('confidence', 'entropy', 'margin',
                   'answer_stability', 'conf_slope', 'prob_slope')
    for edge_data in gate1_result.values():
        for sig in all_signals:
            sd = edge_data.get(sig, {})
            auc = sd.get('auc', float('nan'))
            rho = sd.get('spearman', float('nan'))
            if (not _isnan(auc) and auc >= 0.60) or \
               (not _isnan(rho) and abs(rho) >= 0.30):
                return True
    return False


def _isnan(x: float) -> bool:
    import math
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return True
