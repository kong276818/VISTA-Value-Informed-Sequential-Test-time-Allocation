"""
Load and structure VISTA checkpoint-table results.

Item table schema:
  item_table[question_id][budget] = {
      'option_probs': list[float],   # softmax-normalised per-option probs
      'raw_logprobs': list[float],   # raw log-probs from the model
      'answer_index': int,           # 0-based correct answer index
      'n_options':    int,
      'category':     str,
      'answer':       str,           # letter, e.g. 'A'
      'n_missing':    int,           # logprob positions not in top-200
  }

Dataset registry maps (model_tag, ds_tag) → file paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BUDGETS: list[int] = [256, 512, 1024, 2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Registry: (model_tag, ds_tag) → (results_path, items_path, gpu_tag, n_expected)
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[tuple[str, str], dict] = {
    ('m1', 'ds1'): {
        'results': 'data/results_m1_ds1_4090.json',
        'items':   'data/items_ds1.json',
        'gpu_tag': '4090',
        'ds_name': 'MMLU-Pro',
        'n_options': 10,
        'n_expected': 1000,
    },
    ('m1', 'ds2'): {
        'results': 'data/results_m1_ds2_4090.json',
        'items':   'data/items_ds2.json',
        'gpu_tag': '4090',
        'ds_name': 'ARC-Challenge',
        'n_options': 4,
        'n_expected': 979,          # 21 items excluded
    },
    ('m1', 'ds3'): {
        'results': 'data/results_m1_ds3_4090.json',
        'items':   'data/items_ds3.json',
        'gpu_tag': '4090',
        'ds_name': 'MedMCQA',
        'n_options': 4,
        'n_expected': 500,
    },
    ('m1', 'ds4'): {
        'results': 'data/results_m1_ds4_4090.json',
        'items':   'data/items_ds4.json',
        'gpu_tag': '4090',
        'ds_name': 'MedQA-USMLE',
        'n_options': 4,
        'n_expected': 500,
    },
    # M2 entries - populated after full corpus completes
    ('m2', 'ds1'): {
        'results': 'data/results_m2_ds1_3090a.json',
        'items':   'data/items_ds1.json',
        'gpu_tag': '3090a',
        'ds_name': 'MMLU-Pro',
        'n_options': 10,
        'n_expected': 1000,
    },
    ('m2', 'ds2'): {
        'results': 'data/results_m2_ds2_3090a.json',
        'items':   'data/items_ds2.json',
        'gpu_tag': '3090a',
        'ds_name': 'ARC-Challenge',
        'n_options': 4,
        'n_expected': 979,
    },
    ('m2', 'ds3'): {
        'results': 'data/results_m2_ds3_3090b.json',
        'items':   'data/items_ds3.json',
        'gpu_tag': '3090b',
        'ds_name': 'MedMCQA',
        'n_options': 4,
        'n_expected': 500,
    },
    ('m2', 'ds4'): {
        'results': 'data/results_m2_ds4_3090b.json',
        'items':   'data/items_ds4.json',
        'gpu_tag': '3090b',
        'ds_name': 'MedQA-USMLE',
        'n_options': 4,
        'n_expected': 500,
    },
}


def _coerce(v: Any, typ: type) -> Any:
    """Coerce a value that may have been JSON-serialised as a string."""
    if isinstance(v, typ):
        return v
    try:
        if typ is int:
            return int(v)
        if typ is float:
            return float(v)
        if typ is list:
            return json.loads(v) if isinstance(v, str) else list(v)
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return v


def load_item_table(
    results_path: str | Path,
    root: str | Path = '.',
) -> dict[str, dict[int, dict]]:
    """
    Load a run-d5 results file and return item_table.

    item_table[question_id][budget_int] = record_dict
    """
    path = Path(root) / results_path
    payload = json.loads(path.read_text())
    records = payload['results'] if isinstance(payload, dict) else payload

    item_table: dict[str, dict[int, dict]] = {}
    for r in records:
        qid = str(r['question_id'])
        b   = int(_coerce(r['budget'], int))
        item_table.setdefault(qid, {})[b] = {
            'option_probs': [float(x) for x in _coerce(r['option_probs'], list)],
            'raw_logprobs': [float(x) if x is not None else None for x in _coerce(r['raw_logprobs'], list)],
            'answer_index': int(_coerce(r['answer_index'], int)),
            'n_options':    int(_coerce(r['n_options'], int)),
            'category':     str(r.get('category', '')),
            'answer':       str(r.get('answer', '')),
            'n_missing':    int(_coerce(r.get('n_missing', 0), int)),
        }
    return item_table


def load_items_ground_truth(
    items_path: str | Path,
    root: str | Path = '.',
) -> dict[str, dict]:
    """
    Load items file and return {question_id: item_metadata}.
    """
    path = Path(root) / items_path
    payload = json.loads(path.read_text())
    items = payload['items'] if isinstance(payload, dict) else payload
    return {it['question_id']: it for it in items}


def load_cell(
    model_tag: str,
    ds_tag: str,
    root: str | Path = '.',
) -> tuple[dict[str, dict[int, dict]], dict, dict]:
    """
    Load a (model, dataset) cell.
    Returns (item_table, items_meta, registry_entry).
    Raises FileNotFoundError if the results file is absent (M2 not finished).
    """
    key = (model_tag, ds_tag)
    if key not in DATASET_REGISTRY:
        raise KeyError(f'Unknown cell ({model_tag}, {ds_tag})')
    entry = DATASET_REGISTRY[key]
    results_path = Path(root) / entry['results']
    if not results_path.exists():
        raise FileNotFoundError(
            f'Results file not found: {results_path} — '
            f'corpus for ({model_tag}, {ds_tag}) may not be complete.'
        )
    item_table = load_item_table(entry['results'], root=root)
    items_meta = load_items_ground_truth(entry['items'], root=root)
    return item_table, items_meta, entry


def available_cells(root: str | Path = '.') -> list[tuple[str, str]]:
    """Return (model_tag, ds_tag) pairs whose results files exist."""
    available = []
    for (m, d), entry in DATASET_REGISTRY.items():
        p = Path(root) / entry['results']
        if p.exists():
            available.append((m, d))
    return available


def validate_item_table(
    item_table: dict,
    entry: dict,
    verbose: bool = False,
) -> dict:
    """
    Sanity check: count items, budgets, missing logprobs, duplicate qids.
    Returns summary dict.
    """
    n_items     = len(item_table)
    n_expected  = entry.get('n_expected', -1)
    budget_counts = {b: 0 for b in BUDGETS}
    total_missing = 0
    incomplete = []   # items not present at all 6 budgets

    for qid, bmap in item_table.items():
        for b in BUDGETS:
            if b in bmap:
                budget_counts[b] += 1
                total_missing += bmap[b]['n_missing']
        if set(bmap.keys()) != set(BUDGETS):
            incomplete.append(qid)

    summary = {
        'n_items':        n_items,
        'n_expected':     n_expected,
        'n_match':        n_items == n_expected,
        'budget_counts':  budget_counts,
        'total_missing_logprobs': total_missing,
        'n_incomplete':   len(incomplete),
        'incomplete_qids': incomplete[:10],
    }
    if verbose:
        print(f"  n_items={n_items} (expected={n_expected}, match={summary['n_match']})")
        print(f"  budget_counts: {budget_counts}")
        print(f"  total_missing_logprobs={total_missing}")
        print(f"  incomplete items={len(incomplete)}")
    return summary
