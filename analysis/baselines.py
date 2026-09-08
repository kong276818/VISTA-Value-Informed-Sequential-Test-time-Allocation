"""
Stopping-policy baselines.

Policy interface: each policy receives the item_table and returns a dict
  {question_id: assigned_budget}
plus summary metrics via evaluate_policy().

Policy taxonomy
---------------
Fixed (1–7):
  F1  fixed_256   — always budget 256
  F2  fixed_512   — always budget 512
  F3  fixed_1024
  F4  fixed_2048
  F5  fixed_4096
  F6  fixed_8192  — always-max baseline
  F7  fixed_median  — always the median budget in the grid

Prefix-signal threshold (8–12):
  T8  conf_thresh  — stop at first b where confidence >= τ (default 0.90)
  T9  entropy_thresh — stop at first b where entropy <= τ (default 0.30)
  T10 margin_thresh — stop at first b where margin >= τ (default 0.50)
  T11 conf_low_thresh — stop at first b where confidence < τ (explore more) [reverse]
  T12 early_exit  — stop at B=1024 unless confidence < 0.50

External SOTA (13–15) — NOT_IMPLEMENTED:
  E13 s1_simple_scaling — S1-style budget forcing (external)
  E14 adaptive_compute  — Adaptive compute (external)
  E15 budget_aware_ft   — Fine-tuned budget-aware model (external)

VISTA (16–17):
  V16 vista_global      — global threshold calibrated on validation split
  V17 vista_item_adaptive — per-item adaptive policy

Oracles (18–19):
  O18 hindsight_brier   — oracle minimising Brier per item
  O19 hindsight_acc     — oracle maximising accuracy per item
"""

from __future__ import annotations

import statistics
from typing import Any
from analysis.metrics import (
    brier, nll, accuracy, mean_brier, mean_nll,
    confidence, entropy, margin, is_correct,
    token_metrics, token_saving_vs_fixed_max,
    conf_wrong_rate,
)
from analysis.data_loader import BUDGETS


NOT_IMPLEMENTED = 'NOT_IMPLEMENTED'


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _records_for_assignments(
    item_table: dict[str, dict[int, dict]],
    assignments: dict[str, int],
) -> list[dict]:
    """Build flat list of records given per-item budget assignments."""
    records = []
    for qid, b in assignments.items():
        bmap = item_table.get(qid, {})
        if b in bmap:
            records.append(bmap[b])
    return records


def _first_budget_meeting(
    bmap: dict[int, dict],
    budgets: list[int],
    criterion: callable,
) -> int:
    """Return the first budget (ascending) where criterion(bmap[b]) is True."""
    for b in sorted(b for b in budgets if b in bmap):
        if criterion(bmap[b]):
            return b
    # fallback: max available budget
    return max(b for b in budgets if b in bmap)


# ---------------------------------------------------------------------------
# Fixed policies
# ---------------------------------------------------------------------------

def _fixed_policy(item_table: dict, b: int) -> dict[str, int]:
    return {qid: b for qid, bmap in item_table.items() if b in bmap}


def fixed_256(item_table):   return _fixed_policy(item_table, 256)
def fixed_512(item_table):   return _fixed_policy(item_table, 512)
def fixed_1024(item_table):  return _fixed_policy(item_table, 1024)
def fixed_2048(item_table):  return _fixed_policy(item_table, 2048)
def fixed_4096(item_table):  return _fixed_policy(item_table, 4096)
def fixed_8192(item_table):  return _fixed_policy(item_table, 8192)

def fixed_median(item_table: dict) -> dict[str, int]:
    """Always use the median budget in the grid."""
    b = sorted(BUDGETS)[len(BUDGETS) // 2]
    return _fixed_policy(item_table, b)


# ---------------------------------------------------------------------------
# Threshold policies (applied to prefix signals at each budget level)
# ---------------------------------------------------------------------------

def conf_thresh_policy(
    item_table: dict, tau: float = 0.90
) -> dict[str, int]:
    """Stop at first budget where model confidence >= tau."""
    assignments = {}
    for qid, bmap in item_table.items():
        def crit(r): return confidence(r['option_probs']) >= tau
        assignments[qid] = _first_budget_meeting(bmap, BUDGETS, crit)
    return assignments


def entropy_thresh_policy(
    item_table: dict, tau: float = 0.30
) -> dict[str, int]:
    """Stop at first budget where entropy <= tau nats."""
    assignments = {}
    for qid, bmap in item_table.items():
        def crit(r): return entropy(r['option_probs']) <= tau
        assignments[qid] = _first_budget_meeting(bmap, BUDGETS, crit)
    return assignments


def margin_thresh_policy(
    item_table: dict, tau: float = 0.50
) -> dict[str, int]:
    """Stop at first budget where margin (top1-top2) >= tau."""
    assignments = {}
    for qid, bmap in item_table.items():
        def crit(r): return margin(r['option_probs']) >= tau
        assignments[qid] = _first_budget_meeting(bmap, BUDGETS, crit)
    return assignments


def conf_low_thresh_policy(
    item_table: dict, tau: float = 0.50
) -> dict[str, int]:
    """Continue to next budget if confidence is low (< tau); else stop. [exploration bias]"""
    assignments = {}
    for qid, bmap in item_table.items():
        # Stop at first budget where confidence >= tau (same as conf_thresh but tau=0.5)
        def crit(r): return confidence(r['option_probs']) >= tau
        assignments[qid] = _first_budget_meeting(bmap, BUDGETS, crit)
    return assignments


def early_exit_policy(
    item_table: dict, early_b: int = 1024, conf_floor: float = 0.50
) -> dict[str, int]:
    """
    Use early_b unless confidence < conf_floor there; if so, use max budget.
    """
    assignments = {}
    for qid, bmap in item_table.items():
        if early_b not in bmap:
            assignments[qid] = max(b for b in BUDGETS if b in bmap)
            continue
        if confidence(bmap[early_b]['option_probs']) >= conf_floor:
            assignments[qid] = early_b
        else:
            assignments[qid] = max(b for b in BUDGETS if b in bmap)
    return assignments


# ---------------------------------------------------------------------------
# Oracle policies
# ---------------------------------------------------------------------------

def oracle_brier(item_table: dict) -> dict[str, int]:
    """Hindsight oracle: minimum budget achieving minimum Brier per item."""
    from analysis.gate0 import hindsight_oracle_budget
    return {
        qid: hindsight_oracle_budget(bmap, BUDGETS)
        for qid, bmap in item_table.items()
    }


def oracle_accuracy(item_table: dict) -> dict[str, int]:
    """Hindsight oracle: minimum budget achieving correct prediction."""
    assignments = {}
    for qid, bmap in item_table.items():
        present = sorted(b for b in BUDGETS if b in bmap)
        assigned = present[-1]   # fallback: max
        for b in present:
            r = bmap[b]
            if is_correct(r['option_probs'], r['answer_index']):
                assigned = b
                break
        assignments[qid] = assigned
    return assignments


# ---------------------------------------------------------------------------
# VISTA policies (placeholder with calibration hook)
# ---------------------------------------------------------------------------

def vista_global(
    item_table: dict,
    tau_conf: float = 0.85,
    tau_margin: float = 0.45,
) -> dict[str, int]:
    """
    VISTA global policy: stop at first budget where BOTH
      confidence >= tau_conf AND margin >= tau_margin.
    Global thresholds calibrated on a held-out validation split (not implemented
    here — pass calibrated tau values as arguments).
    """
    assignments = {}
    for qid, bmap in item_table.items():
        def crit(r):
            p = r['option_probs']
            return confidence(p) >= tau_conf and margin(p) >= tau_margin
        assignments[qid] = _first_budget_meeting(bmap, BUDGETS, crit)
    return assignments


def vista_item_adaptive(
    item_table: dict,
    category_thresholds: dict[str, tuple[float, float]] | None = None,
    default_tau: tuple[float, float] = (0.85, 0.45),
) -> dict[str, int]:
    """
    VISTA item-adaptive: per-category or per-item thresholds.
    category_thresholds: {category: (tau_conf, tau_margin)}.
    Falls back to default_tau for unseen categories.
    """
    assignments = {}
    for qid, bmap in item_table.items():
        # Determine threshold from first available budget's category
        cat = ''
        for b in sorted(b for b in BUDGETS if b in bmap):
            cat = bmap[b].get('category', '')
            break
        tau_conf, tau_margin = (
            category_thresholds.get(cat, default_tau)
            if category_thresholds
            else default_tau
        )
        def crit(r, tc=tau_conf, tm=tau_margin):
            p = r['option_probs']
            return confidence(p) >= tc and margin(p) >= tm
        assignments[qid] = _first_budget_meeting(bmap, BUDGETS, crit)
    return assignments


# ---------------------------------------------------------------------------
# Evaluate a policy
# ---------------------------------------------------------------------------

def evaluate_policy(
    item_table: dict,
    assignments: dict[str, int],
    b_max: int = 8192,
    conf_wrong_threshold: float = 0.90,
) -> dict:
    """
    Given per-item budget assignments, compute all summary metrics.

    Returns dict with keys:
      accuracy, mean_brier, mean_nll, mean_conf, conf_wrong_rate,
      token_mean, token_median, token_p90, token_total, token_saving,
      n_items
    """
    records = _records_for_assignments(item_table, assignments)
    budgets_used = [assignments[qid] for qid in assignments if qid in item_table]

    tok = token_metrics(budgets_used)
    saving = token_saving_vs_fixed_max(budgets_used, b_max)

    return {
        'accuracy':        accuracy(records),
        'mean_brier':      mean_brier(records),
        'mean_nll':        mean_nll(records),
        'mean_conf':       mean_confidence_from_records(records),
        'conf_wrong_rate': conf_wrong_rate(records, conf_wrong_threshold),
        'token_mean':      tok.get('mean', float('nan')),
        'token_median':    tok.get('median', float('nan')),
        'token_p90':       tok.get('p90', float('nan')),
        'token_total':     tok.get('total', 0),
        'token_saving':    saving,
        'n_items':         len(records),
    }


def mean_confidence_from_records(records: list[dict]) -> float:
    if not records:
        return float('nan')
    return sum(confidence(r['option_probs']) for r in records) / len(records)


# ---------------------------------------------------------------------------
# Registry of all policies
# ---------------------------------------------------------------------------

POLICY_REGISTRY: dict[str, dict] = {
    'F1_fixed_256':         {'fn': fixed_256,          'label': 'Fixed B=256',          'group': 'fixed'},
    'F2_fixed_512':         {'fn': fixed_512,          'label': 'Fixed B=512',          'group': 'fixed'},
    'F3_fixed_1024':        {'fn': fixed_1024,         'label': 'Fixed B=1024',         'group': 'fixed'},
    'F4_fixed_2048':        {'fn': fixed_2048,         'label': 'Fixed B=2048',         'group': 'fixed'},
    'F5_fixed_4096':        {'fn': fixed_4096,         'label': 'Fixed B=4096',         'group': 'fixed'},
    'F6_fixed_8192':        {'fn': fixed_8192,         'label': 'Fixed B=8192 (max)',   'group': 'fixed'},
    'F7_fixed_median':      {'fn': fixed_median,       'label': 'Fixed B=median',       'group': 'fixed'},
    'T8_conf_thresh':       {'fn': conf_thresh_policy, 'label': 'Conf≥0.90',            'group': 'threshold'},
    'T9_entropy_thresh':    {'fn': entropy_thresh_policy, 'label': 'Entropy≤0.30',      'group': 'threshold'},
    'T10_margin_thresh':    {'fn': margin_thresh_policy,  'label': 'Margin≥0.50',       'group': 'threshold'},
    'T11_conf_low':         {'fn': conf_low_thresh_policy, 'label': 'Conf≥0.50',        'group': 'threshold'},
    'T12_early_exit':       {'fn': early_exit_policy,  'label': 'EarlyExit(1024,0.5)', 'group': 'threshold'},
    'E13_s1':               {'fn': NOT_IMPLEMENTED,    'label': 'S1 Simple Scaling',    'group': 'external'},
    'E14_adaptive':         {'fn': NOT_IMPLEMENTED,    'label': 'AdaptiveCompute',      'group': 'external'},
    'E15_budget_ft':        {'fn': NOT_IMPLEMENTED,    'label': 'BudgetAware-FT',       'group': 'external'},
    'V16_vista_global':     {'fn': vista_global,       'label': 'VISTA-Global',         'group': 'vista'},
    'V17_vista_adaptive':   {'fn': vista_item_adaptive,'label': 'VISTA-Adaptive',       'group': 'vista'},
    'O18_oracle_brier':     {'fn': oracle_brier,       'label': 'Oracle-Brier',         'group': 'oracle'},
    'O19_oracle_acc':       {'fn': oracle_accuracy,    'label': 'Oracle-Accuracy',      'group': 'oracle'},
}


def run_all_policies(
    item_table: dict,
    b_max: int = 8192,
    conf_wrong_threshold: float = 0.90,
) -> dict[str, dict]:
    """
    Run every implementable policy and return {policy_key: metrics_dict}.
    NOT_IMPLEMENTED policies are skipped.
    """
    results = {}
    for key, reg in POLICY_REGISTRY.items():
        fn = reg['fn']
        if fn is NOT_IMPLEMENTED:
            results[key] = {'status': NOT_IMPLEMENTED, 'label': reg['label'], 'group': reg['group']}
            continue
        assignments = fn(item_table)
        metrics = evaluate_policy(item_table, assignments, b_max, conf_wrong_threshold)
        results[key] = {**metrics, 'label': reg['label'], 'group': reg['group']}
    return results
