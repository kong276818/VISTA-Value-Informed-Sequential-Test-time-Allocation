"""
Top-level evaluation runner.

Usage:
  python -m analysis.evaluate_all [--model m1] [--ds ds1 ds2 ds3 ds4] [--out-dir out/eval]

Produces for each (model, ds) cell:
  out_dir/<model>_<ds>/budget_audit.{tex,csv}
  out_dir/<model>_<ds>/policy_comparison.{tex,csv}
  out_dir/<model>_<ds>/gate0.json
  out_dir/<model>_<ds>/gate1.json
  out_dir/<model>_<ds>/pareto.png
  out_dir/<model>_<ds>/gate1_auc.png

Also writes:
  out_dir/gate0_summary.tex
  out_dir/budget_response.png   (multi-dataset for one model)
  out_dir/results_summary.json  (machine-readable)

Validation anchors (M1 only):
  DS1 Accuracy@8192 ≈ 0.738 ± 0.005
  DS1 Brier@8192   ≈ 0.4917 ± 0.005
  DS3 Accuracy optimum should be B=4096 (not B=1024 for Brier)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Validation anchors for M1 results (from paper.tex / known values)
# ---------------------------------------------------------------------------
M1_ANCHORS = {
    ('m1', 'ds1'): {
        'accuracy@8192':   (0.738, 0.005),   # (expected, tolerance)
        'mean_brier@8192': (0.4917, 0.005),
    },
    ('m1', 'ds2'): {
        'accuracy@8192':   (0.958, 0.005),   # paper.tex tab:budget-audit line 820
    },
    ('m1', 'ds3'): {
        # MedMCQA: accuracy optimum at B=4096, Brier optimum at B=1024
        'accuracy@4096_gt_8192': True,  # special check: acc@4096 >= acc@8192
        'brier_optimum_b':       1024,  # the budget achieving lowest Brier
    },
    ('m1', 'ds4'): {
        'accuracy@8192':   (0.782, 0.005),   # paper.tex tab:budget-audit line 836
    },
}

CONF_WRONG_THRESHOLD = 0.90


def validate_anchors(
    model_tag: str,
    ds_tag: str,
    budget_rows: list[dict],
) -> list[str]:
    """
    Check validation anchors. Returns list of WARN strings (empty = all pass).
    """
    key = (model_tag, ds_tag)
    anchors = M1_ANCHORS.get(key, {})
    warns = []

    rows_by_b = {r['budget']: r for r in budget_rows}

    for anchor_key, expected in anchors.items():
        if anchor_key.startswith('accuracy@') and '@' in anchor_key and '_gt_' not in anchor_key:
            b = int(anchor_key.split('@')[1])
            if b not in rows_by_b:
                warns.append(f'WARN [{key}] Budget {b} not found in results')
                continue
            actual = rows_by_b[b]['accuracy']
            center, tol = expected
            if abs(actual - center) > tol:
                warns.append(
                    f'WARN [{key}] {anchor_key}: expected {center:.4f}±{tol}, '
                    f'got {actual:.4f} (Δ={actual-center:+.4f})'
                )

        elif anchor_key.startswith('mean_brier@'):
            b = int(anchor_key.split('@')[1])
            if b not in rows_by_b:
                warns.append(f'WARN [{key}] Budget {b} not found in results')
                continue
            actual = rows_by_b[b]['mean_brier']
            center, tol = expected
            if abs(actual - center) > tol:
                warns.append(
                    f'WARN [{key}] {anchor_key}: expected {center:.4f}±{tol}, '
                    f'got {actual:.4f} (Δ={actual-center:+.4f})'
                )

        elif anchor_key == 'accuracy@4096_gt_8192':
            if 4096 in rows_by_b and 8192 in rows_by_b:
                acc_4096 = rows_by_b[4096]['accuracy']
                acc_8192 = rows_by_b[8192]['accuracy']
                if acc_4096 < acc_8192 - 0.001:
                    warns.append(
                        f'WARN [{key}] Expected acc@4096 >= acc@8192 '
                        f'(MedMCQA divergence), got {acc_4096:.4f} < {acc_8192:.4f}'
                    )

        elif anchor_key == 'brier_optimum_b':
            best_b = min(rows_by_b.items(), key=lambda kv: kv[1]['mean_brier'])[0]
            if best_b != expected:
                warns.append(
                    f'WARN [{key}] Expected Brier optimum at B={expected}, '
                    f'got B={best_b}'
                )

    return warns


def run_cell_evaluation(
    model_tag: str,
    ds_tag: str,
    root: str | Path = '.',
    out_dir: str | Path = 'out/eval',
    skip_figures: bool = False,
) -> dict:
    """
    Full evaluation for one (model, dataset) cell.
    Returns summary dict.
    """
    from analysis.data_loader import load_cell, validate_item_table, BUDGETS
    from analysis.metrics import budget_level_summary
    from analysis.gate0 import gate0_analysis
    from analysis.gate1 import gate1_full
    from analysis.baselines import run_all_policies
    from analysis.make_tables import (
        latex_budget_audit, csv_budget_audit,
        latex_policy_comparison, csv_policy_comparison,
        write_outputs,
    )
    from analysis.make_figures import pareto_frontier_figure, gate1_heatmap

    root = Path(root)
    cell_dir = Path(out_dir) / f'{model_tag}_{ds_tag}'
    cell_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== Evaluating ({model_tag}, {ds_tag}) ===')

    item_table, items_meta, entry = load_cell(model_tag, ds_tag, root=root)
    ds_name    = entry['ds_name']
    model_name = 'Qwen3-8B' if model_tag == 'm1' else 'DeepSeek-R1-Distill-Llama-8B'

    # Validate data integrity
    val = validate_item_table(item_table, entry, verbose=True)

    # Budget-level metrics
    budget_rows = budget_level_summary(item_table, BUDGETS)
    latex_ba = latex_budget_audit(budget_rows, ds_name, model_name)
    csv_ba   = csv_budget_audit(budget_rows)
    write_outputs(cell_dir, 'budget_audit', latex_ba, csv_ba)

    # Validation anchors
    warns = validate_anchors(model_tag, ds_tag, budget_rows)
    for w in warns:
        print(w)
    if not warns:
        print(f'  [OK] All validation anchors pass for ({model_tag}, {ds_tag})')

    # Gate 0
    g0 = gate0_analysis(item_table, BUDGETS)
    (cell_dir / 'gate0.json').write_text(json.dumps(g0, default=_json_default, indent=2))
    print(f'  Gate0: mean_oracle={g0["mean_oracle"]:.0f}  '
          f'saving={g0["saving_vs_max"]*100:.1f}%  '
          f'PASS={g0["gate0_pass"]}')

    # Gate 1
    g1 = gate1_full(item_table, BUDGETS)
    g1_serialisable = {f'{k[0]}-{k[1]}': v for k, v in g1.items()}
    (cell_dir / 'gate1.json').write_text(json.dumps(g1_serialisable, indent=2))

    # Print Gate 1 summary
    print('  Gate1 AUC (confidence, entropy, margin):')
    for edge, sig_data in sorted(g1.items()):
        vals = [f'{sig_data.get(s, {}).get("auc", float("nan")):.3f}'
                for s in ['confidence', 'entropy', 'margin']]
        n = sig_data.get('n', 0)
        print(f'    {edge[0]:4d}→{edge[1]:4d}: conf={vals[0]} ent={vals[1]} mar={vals[2]}  n={n}')

    # Policy comparison
    policy_results = run_all_policies(item_table, conf_wrong_threshold=CONF_WRONG_THRESHOLD)
    latex_pc = latex_policy_comparison(policy_results, ds_name, model_name)
    csv_pc   = csv_policy_comparison(policy_results)
    write_outputs(cell_dir, 'policy_comparison', latex_pc, csv_pc)

    # Print key policy results
    print('  Key policies:')
    for key in ['F6_fixed_8192', 'F3_fixed_1024', 'T8_conf_thresh',
                'V16_vista_global', 'O18_oracle_brier']:
        m = policy_results.get(key, {})
        if m.get('status') == 'NOT_IMPLEMENTED':
            continue
        print(f'    {key}: acc={m.get("accuracy", float("nan")):.4f}  '
              f'brier={m.get("mean_brier", float("nan")):.4f}  '
              f'saving={m.get("token_saving", float("nan"))*100:.1f}%')

    # Figures
    if not skip_figures:
        try:
            pareto_frontier_figure(
                policy_results, ds_name,
                out_path=cell_dir / 'pareto.png',
            )
            gate1_heatmap(
                g1, ds_name,
                out_path=cell_dir / 'gate1_auc.png',
            )
        except Exception as e:
            print(f'  [WARN] Figure generation failed: {e}')

    return {
        'model_tag':     model_tag,
        'ds_tag':        ds_tag,
        'ds_name':       ds_name,
        'n_items':       val['n_items'],
        'n_complete':    val['n_items'] - val['n_incomplete'],
        'gate0':         {k: v for k, v in g0.items() if k != 'oracle_budgets'},
        'validation_warns': warns,
        'budget_rows':   budget_rows,
        'policy_results': {k: {kk: vv for kk, vv in v.items() if kk != 'status'}
                           for k, v in policy_results.items()
                           if v.get('status') != 'NOT_IMPLEMENTED'},
    }


def _json_default(obj):
    if hasattr(obj, '__float__'):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return float(obj)
    raise TypeError(f'Not serializable: {type(obj)}')


def main():
    parser = argparse.ArgumentParser(description='VISTA evaluation pipeline')
    parser.add_argument('--model', nargs='+', default=['m1'],
                        help='Model tags to evaluate (default: m1)')
    parser.add_argument('--ds',    nargs='+', default=['ds1', 'ds2', 'ds3', 'ds4'],
                        help='Dataset tags to evaluate')
    parser.add_argument('--out-dir', default='out/eval',
                        help='Output directory (default: out/eval)')
    parser.add_argument('--root', default='.',
                        help='Repository root (default: .)')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Skip figure generation (faster dry-run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Alias for --skip-figures')
    args = parser.parse_args()

    skip_figures = args.skip_figures or args.dry_run
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from analysis.data_loader import available_cells, DATASET_REGISTRY, BUDGETS

    all_results = {}
    gate0_all   = {}
    budget_summaries_by_ds: dict[str, list[dict]] = {}

    for model_tag in args.model:
        for ds_tag in args.ds:
            key = (model_tag, ds_tag)
            if key not in DATASET_REGISTRY:
                print(f'SKIP unknown cell ({model_tag}, {ds_tag})')
                continue

            try:
                summary = run_cell_evaluation(
                    model_tag, ds_tag,
                    root=args.root,
                    out_dir=out_dir,
                    skip_figures=skip_figures,
                )
                all_results[f'{model_tag}_{ds_tag}'] = summary
                gate0_all[key] = summary['gate0']
                ds_name = summary['ds_name']
                budget_summaries_by_ds[ds_name] = summary['budget_rows']

            except FileNotFoundError as e:
                print(f'SKIP ({model_tag}, {ds_tag}): {e}')

    # Multi-dataset budget response figure
    if budget_summaries_by_ds and not skip_figures:
        from analysis.make_figures import budget_response_figure
        try:
            budget_response_figure(
                budget_summaries_by_ds,
                out_path=out_dir / 'budget_response.png',
            )
            print(f'\nWrote {out_dir}/budget_response.png')
        except Exception as e:
            print(f'[WARN] Budget response figure failed: {e}')

    # Gate 0 summary table
    if gate0_all:
        from analysis.make_tables import latex_gate0_summary
        tex = latex_gate0_summary(gate0_all)
        (out_dir / 'gate0_summary.tex').write_text(tex)
        print(f'Wrote {out_dir}/gate0_summary.tex')

    # Machine-readable summary
    (out_dir / 'results_summary.json').write_text(
        json.dumps(all_results, default=_json_default, indent=2)
    )
    print(f'Wrote {out_dir}/results_summary.json')

    # Final status
    n_warn = sum(len(r.get('validation_warns', [])) for r in all_results.values())
    n_cells = len(all_results)
    print(f'\nDone. {n_cells} cell(s) evaluated, {n_warn} validation warning(s).')
    if n_warn > 0:
        print('[ACTION REQUIRED] Review validation warnings above before reporting results.')


if __name__ == '__main__':
    main()
