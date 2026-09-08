"""
Paired bootstrap confidence intervals.

Usage:
  ci = bootstrap_ci(metric_fn, records_a, records_b, seed=42, n_boot=1000)
  # ci['diff'] = point estimate of metric_a - metric_b
  # ci['lo'], ci['hi'] = 95% CI for the difference

For single-sample CIs (e.g., accuracy at a given budget):
  ci = bootstrap_ci_single(metric_fn, records, seed=42, n_boot=1000)
"""

from __future__ import annotations

import random
import math


BOOTSTRAP_SEED = 42
N_BOOT = 1000


def bootstrap_ci_single(
    metric_fn: callable,
    records: list[dict],
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
) -> dict:
    """
    Percentile bootstrap CI for a scalar metric computed over a list of records.

    metric_fn(records) -> float
    """
    rng = random.Random(seed)
    n = len(records)
    point = metric_fn(records)

    boot_vals = []
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        boot_vals.append(metric_fn(sample))

    boot_vals.sort()
    lo_idx = int(math.floor(alpha / 2 * n_boot))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_boot)) - 1
    return {
        'point': point,
        'lo':    boot_vals[max(0, lo_idx)],
        'hi':    boot_vals[min(n_boot - 1, hi_idx)],
        'n':     n,
        'n_boot': n_boot,
        'alpha': alpha,
    }


def bootstrap_ci_paired(
    metric_fn: callable,
    records_a: list[dict],
    records_b: list[dict],
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
) -> dict:
    """
    Paired bootstrap CI for difference metric_fn(A) - metric_fn(B).

    Assumes records_a[i] and records_b[i] correspond to the same item.
    metric_fn(records) -> float
    """
    if len(records_a) != len(records_b):
        raise ValueError(
            f'Paired bootstrap requires equal-length record lists '
            f'({len(records_a)} != {len(records_b)})'
        )
    rng = random.Random(seed)
    n = len(records_a)
    point = metric_fn(records_a) - metric_fn(records_b)

    boot_diffs = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_a = [records_a[i] for i in idxs]
        sample_b = [records_b[i] for i in idxs]
        boot_diffs.append(metric_fn(sample_a) - metric_fn(sample_b))

    boot_diffs.sort()
    lo_idx = int(math.floor(alpha / 2 * n_boot))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_boot)) - 1
    return {
        'diff': point,
        'lo':   boot_diffs[max(0, lo_idx)],
        'hi':   boot_diffs[min(n_boot - 1, hi_idx)],
        'n':    n,
        'n_boot': n_boot,
        'alpha': alpha,
        'significant': not (boot_diffs[max(0, lo_idx)] <= 0 <= boot_diffs[min(n_boot - 1, hi_idx)]),
    }


def paired_significance(
    metric_fn: callable,
    records_a: list[dict],
    records_b: list[dict],
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
) -> bool:
    """Convenience: return True if the paired CI for A-B excludes zero."""
    ci = bootstrap_ci_paired(metric_fn, records_a, records_b, seed, n_boot, alpha)
    return ci['significant']
