"""
Core metric functions for VISTA evaluation.

Brier score convention (summed multiclass, NOT divided by C):
  brier(p, y) = sum_c (p_c - 1{c==y})^2

All functions operate on a single item's probability vector and ground truth.
Aggregation helpers operate on lists of per-item values.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


# ---------------------------------------------------------------------------
# Per-item metrics
# ---------------------------------------------------------------------------

def brier(probs: list[float], answer_index: int) -> float:
    """Summed multiclass Brier score for one item (not divided by C)."""
    return sum(
        (p - (1.0 if i == answer_index else 0.0)) ** 2
        for i, p in enumerate(probs)
    )


def nll(probs: list[float], answer_index: int) -> float:
    """Negative log-likelihood for the correct class."""
    p = max(probs[answer_index], 1e-12)
    return -math.log(p)


def is_correct(probs: list[float], answer_index: int) -> bool:
    return probs.index(max(probs)) == answer_index


def confidence(probs: list[float]) -> float:
    """Max probability (confidence of the argmax prediction)."""
    return max(probs)


def entropy(probs: list[float]) -> float:
    """Entropy of the distribution (nats)."""
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def margin(probs: list[float]) -> float:
    """Difference between top-2 probabilities."""
    s = sorted(probs, reverse=True)
    return s[0] - s[1] if len(s) >= 2 else s[0]


def is_conf_wrong(probs: list[float], answer_index: int, threshold: float = 0.90) -> bool:
    """True if model is confidently wrong: conf >= threshold AND prediction wrong."""
    if probs.index(max(probs)) == answer_index:
        return False
    return max(probs) >= threshold


# ---------------------------------------------------------------------------
# Aggregate metrics over a list of (probs, answer_index) items
# ---------------------------------------------------------------------------

def accuracy(records: list[dict]) -> float:
    """Mean 0/1 accuracy; records must have 'option_probs' and 'answer_index'."""
    if not records:
        return float('nan')
    return sum(is_correct(r['option_probs'], r['answer_index']) for r in records) / len(records)


def mean_brier(records: list[dict]) -> float:
    if not records:
        return float('nan')
    return sum(brier(r['option_probs'], r['answer_index']) for r in records) / len(records)


def mean_nll(records: list[dict]) -> float:
    if not records:
        return float('nan')
    return sum(nll(r['option_probs'], r['answer_index']) for r in records) / len(records)


def mean_confidence(records: list[dict]) -> float:
    if not records:
        return float('nan')
    return sum(confidence(r['option_probs']) for r in records) / len(records)


def conf_wrong_rate(records: list[dict], threshold: float = 0.90) -> float:
    if not records:
        return float('nan')
    return sum(
        is_conf_wrong(r['option_probs'], r['answer_index'], threshold)
        for r in records
    ) / len(records)


# ---------------------------------------------------------------------------
# Budget / token metrics
# ---------------------------------------------------------------------------

def token_metrics(budgets_used: list[int]) -> dict:
    """
    Summarise token usage distribution.
    budgets_used: one integer budget per item (the selected/assigned budget).
    """
    if not budgets_used:
        return {}
    return {
        'mean':   statistics.mean(budgets_used),
        'median': statistics.median(budgets_used),
        'p90':    _percentile(budgets_used, 0.90),
        'total':  sum(budgets_used),
    }


def token_saving_vs_fixed_max(budgets_used: list[int], b_max: int = 8192) -> float:
    """Fractional token saving relative to always-max baseline."""
    if not budgets_used:
        return float('nan')
    baseline = b_max * len(budgets_used)
    used = sum(budgets_used)
    return (baseline - used) / baseline


def _percentile(data: list[float | int], q: float) -> float:
    s = sorted(data)
    n = len(s)
    idx = q * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# Flip / stability metrics across adjacent budget levels
# ---------------------------------------------------------------------------

def flip_rate(
    item_table: dict[str, dict[int, dict]],
    b_low: int,
    b_high: int,
) -> float:
    """
    Fraction of items where argmax prediction changes between b_low and b_high.
    Only items that have both budgets are included.
    """
    n, flips = 0, 0
    for bmap in item_table.values():
        if b_low not in bmap or b_high not in bmap:
            continue
        n += 1
        pred_lo = bmap[b_low]['option_probs'].index(max(bmap[b_low]['option_probs']))
        pred_hi = bmap[b_high]['option_probs'].index(max(bmap[b_high]['option_probs']))
        if pred_lo != pred_hi:
            flips += 1
    return flips / n if n > 0 else float('nan')


def all_flip_rates(item_table: dict, budgets: list[int]) -> dict[tuple[int, int], float]:
    """Compute flip rate for every adjacent pair in budgets."""
    result = {}
    for i in range(len(budgets) - 1):
        b_lo, b_hi = budgets[i], budgets[i + 1]
        result[(b_lo, b_hi)] = flip_rate(item_table, b_lo, b_hi)
    return result


# ---------------------------------------------------------------------------
# Budget-level summary table
# ---------------------------------------------------------------------------

def budget_level_summary(item_table: dict, budgets: list[int]) -> list[dict]:
    """
    For each budget level, compute aggregate metrics over all items
    that have that budget.  Returns list of dicts sorted by budget.
    """
    rows = []
    for b in budgets:
        records = [bmap[b] for bmap in item_table.values() if b in bmap]
        if not records:
            continue
        rows.append({
            'budget':     b,
            'n':          len(records),
            'accuracy':   accuracy(records),
            'mean_brier': mean_brier(records),
            'mean_nll':   mean_nll(records),
            'mean_conf':  mean_confidence(records),
            'conf_wrong': conf_wrong_rate(records),
        })
    return rows
