"""
Figures for VISTA evaluation.

Figure types:
  - budget_response:  accuracy and Brier vs budget (multi-dataset, log2 x-axis)
  - pareto_frontier:  accuracy vs token-saving across policies
  - gate1_heatmap:    AUC heatmap for Gate 1 signals
"""

from __future__ import annotations

from pathlib import Path


# Shared style constants
DS_STYLE = {
    'MMLU-Pro':      {'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
    'ARC-Challenge': {'color': '#2ca02c', 'marker': 's', 'ls': '--'},
    'MedMCQA':       {'color': '#d62728', 'marker': '^', 'ls': '-.'},
    'MedQA-USMLE':   {'color': '#ff7f0e', 'marker': 'D', 'ls': ':'},
}

BUDGET_ORDER = [256, 512, 1024, 2048, 4096, 8192]


def budget_response_figure(
    dataset_summaries: dict[str, list[dict]],
    out_path: str | Path = 'fig_budget_response.png',
    dpi: int = 150,
) -> None:
    """
    dataset_summaries: {ds_name: budget_level_summary rows}
    Each row: {budget, accuracy, mean_brier, ...}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import math

    fig, (ax_acc, ax_brier) = plt.subplots(1, 2, figsize=(10, 4))

    for ds_name, rows in dataset_summaries.items():
        style = DS_STYLE.get(ds_name, {})
        color  = style.get('color', None)
        marker = style.get('marker', 'o')
        ls     = style.get('ls', '-')

        rows_sorted = sorted(rows, key=lambda r: r['budget'])
        xs    = [math.log2(r['budget']) for r in rows_sorted]
        accs  = [r['accuracy']   for r in rows_sorted]
        briers = [r['mean_brier'] for r in rows_sorted]

        ax_acc.plot(xs, accs,   marker=marker, color=color, ls=ls, label=ds_name, lw=1.5)
        ax_brier.plot(xs, briers, marker=marker, color=color, ls=ls, label=ds_name, lw=1.5)

    budget_ticks = [math.log2(b) for b in BUDGET_ORDER]
    for ax in (ax_acc, ax_brier):
        ax.set_xticks(budget_ticks)
        ax.set_xticklabels([str(b) for b in BUDGET_ORDER], fontsize=8)
        ax.set_xlabel('Budget (tokens)', fontsize=10)
        ax.grid(True, alpha=0.3)

    ax_acc.set_ylabel('Accuracy', fontsize=10)
    ax_acc.set_title('(a) Accuracy vs Budget', fontsize=11)
    ax_acc.legend(fontsize=8, loc='lower right')

    ax_brier.set_ylabel('Summed Brier (mean over items)', fontsize=10)
    ax_brier.set_title('(b) Summed Brier vs Budget', fontsize=11)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def pareto_frontier_figure(
    policy_results: dict[str, dict],
    ds_name: str,
    out_path: str | Path | None = None,
    dpi: int = 150,
) -> None:
    """
    Accuracy vs token-saving scatter across all implementable policies.
    Highlights VISTA policies; oracles shown as dashed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = f'fig_pareto_{ds_name.lower().replace("-", "")}.png'

    fig, ax = plt.subplots(figsize=(7, 5))

    group_colors = {
        'fixed':     '#aec7e8',
        'threshold': '#ffbb78',
        'vista':     '#d62728',
        'oracle':    '#9467bd',
        'external':  '#c7c7c7',
    }

    for key, m in policy_results.items():
        if m.get('status') == 'NOT_IMPLEMENTED':
            continue
        acc     = m.get('accuracy', float('nan'))
        saving  = m.get('token_saving', float('nan')) * 100
        group   = m.get('group', 'fixed')
        label   = m.get('label', key)
        color   = group_colors.get(group, 'gray')

        ls = '--' if group == 'oracle' else '-'
        zorder = 5 if group == 'vista' else 2

        ax.scatter([saving], [acc], color=color, zorder=zorder, s=60,
                   marker='*' if group == 'vista' else 'o')
        ax.annotate(
            label, (saving, acc),
            fontsize=6, textcoords='offset points', xytext=(4, 2),
        )

    ax.set_xlabel('Token saving vs fixed-max (%)', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title(f'Accuracy vs Token Saving — {ds_name}', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def gate1_heatmap(
    gate1_results: dict[tuple[int, int], dict],
    ds_name: str,
    metric: str = 'auc',
    out_path: str | Path | None = None,
    dpi: int = 150,
) -> None:
    """
    Heatmap of signal AUC or Spearman across edges (rows) × signals (cols).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if out_path is None:
        out_path = f'fig_gate1_{ds_name.lower().replace("-", "")}_{metric}.png'

    signal_names = ['confidence', 'entropy', 'margin']
    edges = sorted(gate1_results.keys())
    edge_labels = [f'{b_lo}→{b_hi}' for b_lo, b_hi in edges]

    data = []
    for edge in edges:
        row = []
        for s in signal_names:
            val = gate1_results[edge].get(s, {}).get(metric, float('nan'))
            row.append(val)
        data.append(row)

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(5, max(3, len(edges) * 0.6)))
    im = ax.imshow(data_arr, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    fig.colorbar(im, ax=ax, label=metric.upper())

    ax.set_xticks(range(len(signal_names)))
    ax.set_xticklabels(signal_names, fontsize=9)
    ax.set_yticks(range(len(edge_labels)))
    ax.set_yticklabels(edge_labels, fontsize=9)
    ax.set_title(f'Gate 1 {metric.upper()} — {ds_name}', fontsize=11)

    for i in range(len(edges)):
        for j in range(len(signal_names)):
            v = data_arr[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if v < 0.4 or v > 0.8 else 'black')

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
