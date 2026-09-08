"""
Generate LaTeX and CSV output tables from evaluation results.

Table types:
  - budget_audit:    rows=budgets, cols=metrics, one table per (model, ds)
  - policy_comparison: rows=policies, cols=metrics, one table per (model, ds)
  - gate0_summary:   one row per (model, ds)
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Budget audit table (accuracy, Brier, NLL per budget level)
# ---------------------------------------------------------------------------

def latex_budget_audit(
    rows: list[dict],
    ds_name: str,
    model_name: str = 'Qwen3-8B',
    label: str | None = None,
) -> str:
    """
    rows: output of metrics.budget_level_summary()
    Each row has: budget, n, accuracy, mean_brier, mean_nll
    """
    if label is None:
        label = f'tab:budget-audit-{ds_name.lower().replace("-", "")}'

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        rf'\caption{{Budget vs.\ performance ({model_name}, {ds_name})}}',
        rf'\label{{{label}}}',
        r'\begin{tabular}{rrrrrr}',
        r'\toprule',
        r'Budget & $n$ & Accuracy$\uparrow$ & Brier$\downarrow$ & NLL$\downarrow$ \\',
        r'\midrule',
    ]
    for r in sorted(rows, key=lambda x: x['budget']):
        b     = r['budget']
        n     = r['n']
        acc   = r['accuracy']
        brier = r['mean_brier']
        nll   = r['mean_nll']
        lines.append(
            rf'{b:>5} & {n:>4} & {acc:.4f} & {brier:.4f} & {nll:.4f} \\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def csv_budget_audit(rows: list[dict]) -> str:
    """Return CSV string for budget audit rows."""
    buf = io.StringIO()
    fieldnames = ['budget', 'n', 'accuracy', 'mean_brier', 'mean_nll', 'mean_conf', 'conf_wrong']
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for r in sorted(rows, key=lambda x: x['budget']):
        writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Policy comparison table
# ---------------------------------------------------------------------------

def latex_policy_comparison(
    policy_results: dict[str, dict],
    ds_name: str,
    model_name: str = 'Qwen3-8B',
    label: str | None = None,
) -> str:
    """
    policy_results: output of baselines.run_all_policies()
    """
    if label is None:
        label = f'tab:policy-{ds_name.lower().replace("-", "")}'

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        rf'\caption{{Policy comparison ({model_name}, {ds_name})}}',
        rf'\label{{{label}}}',
        r'\begin{tabular}{llrrrrrr}',
        r'\toprule',
        r'ID & Policy & Acc$\uparrow$ & Brier$\downarrow$ & '
        r'Mean$B$ & Saving\% & CW\%\\',
        r'\midrule',
    ]

    group_order = ['fixed', 'threshold', 'external', 'vista', 'oracle']
    by_group: dict[str, list] = {g: [] for g in group_order}
    for key, m in policy_results.items():
        g = m.get('group', 'fixed')
        by_group.setdefault(g, []).append((key, m))

    for group in group_order:
        items = by_group.get(group, [])
        if not items:
            continue
        lines.append(rf'\multicolumn{{7}}{{l}}{{\textit{{{group}}}}} \\')
        for key, m in sorted(items):
            if m.get('status') == 'NOT_IMPLEMENTED':
                label_txt = m['label']
                lines.append(
                    rf'  {key[:3]} & {label_txt} & --- & --- & --- & --- & --- \\'
                )
                continue
            acc    = m.get('accuracy', float('nan'))
            br     = m.get('mean_brier', float('nan'))
            mean_b = m.get('token_mean', float('nan'))
            saving = m.get('token_saving', float('nan')) * 100
            cw     = m.get('conf_wrong_rate', float('nan')) * 100
            label_txt = m['label']
            lines.append(
                rf'  {key[:3]} & {label_txt} & {acc:.4f} & {br:.4f} '
                rf'& {mean_b:.0f} & {saving:.1f} & {cw:.1f} \\'
            )

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def csv_policy_comparison(policy_results: dict[str, dict]) -> str:
    buf = io.StringIO()
    fieldnames = [
        'policy_key', 'label', 'group',
        'accuracy', 'mean_brier', 'mean_nll',
        'token_mean', 'token_median', 'token_p90', 'token_saving',
        'conf_wrong_rate', 'n_items',
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for key, m in sorted(policy_results.items()):
        if m.get('status') == 'NOT_IMPLEMENTED':
            writer.writerow({'policy_key': key, 'label': m['label'], 'group': m['group']})
            continue
        row = {'policy_key': key, **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}}
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gate 0 summary table
# ---------------------------------------------------------------------------

def latex_gate0_summary(gate0_results: dict[tuple, dict]) -> str:
    """
    gate0_results: {(model_tag, ds_tag): gate0_analysis_result}
    """
    DS_NAMES = {
        'ds1': 'MMLU-Pro', 'ds2': 'ARC-Challenge',
        'ds3': 'MedMCQA',  'ds4': 'MedQA-USMLE',
    }
    MODEL_NAMES = {'m1': 'Qwen3-8B', 'm2': 'DeepSeek-R1-Distill-Llama-8B'}

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        r'\caption{Gate 0: matched-quality oracle analysis}',
        r'\label{tab:gate0}',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Model & Dataset & Mean$B^*$ & Saving\% & Gate~0 \\',
        r'\midrule',
    ]
    for (m, d), r in sorted(gate0_results.items()):
        model_name = MODEL_NAMES.get(m, m)
        ds_name    = DS_NAMES.get(d, d)
        mean_ob  = r.get('mean_oracle', float('nan'))
        saving   = r.get('saving_vs_max', float('nan')) * 100
        pass_str = r'$\checkmark$' if r.get('gate0_pass') else r'$\times$'
        lines.append(
            rf'  {model_name} & {ds_name} & {mean_ob:.0f} & {saving:.1f} & {pass_str} \\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Gate 1 summary table (AUC per signal × edge × dataset)
# ---------------------------------------------------------------------------

def latex_gate1_summary(
    gate1_results: dict[tuple[int, int], dict],
    ds_name: str,
    model_name: str = 'Qwen3-8B',
) -> str:
    signal_names = ['confidence', 'entropy', 'margin']
    edges = sorted(gate1_results.keys())

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        rf'\caption{{Gate 1: AUC of prefix signals predicting Brier improvement ({model_name}, {ds_name})}}',
        r'\label{tab:gate1}',
    ]
    col_spec = 'l' + 'r' * (len(signal_names) * 2)
    lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')

    header = 'Edge'
    for s in signal_names:
        header += rf' & \multicolumn{{2}}{{c}}{{{s}}}'
    lines.append(header + r' \\')
    subheader = ''
    for _ in signal_names:
        subheader += r' & AUC & $\rho$'
    lines.append(subheader + r' \\')
    lines.append(r'\midrule')

    for edge in edges:
        b_lo, b_hi = edge
        row = f'{b_lo}→{b_hi}'
        sig_data = gate1_results[edge]
        for s in signal_names:
            sd = sig_data.get(s, {})
            auc = sd.get('auc', float('nan'))
            rho = sd.get('spearman', float('nan'))
            row += rf' & {auc:.3f} & {rho:.3f}'
        lines.append(row + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------

def write_outputs(
    out_dir: str | Path,
    stem: str,
    latex_str: str,
    csv_str: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{stem}.tex').write_text(latex_str)
    (out_dir / f'{stem}.csv').write_text(csv_str)
