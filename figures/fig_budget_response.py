"""
Figure: Budget–response curves for M1 and M2.
2×2 subplots: row=model, col=metric (Accuracy, Brier).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from vista_data import BUDGETS, M1_DATA, M2_DATA
from vista_style import (apply_style, style_ax, get_dash_kwargs,
                          COLORS, MARKERS, DASHES, DATASETS, FIG_WIDTH)

apply_style()

# ── data ────────────────────────────────────────────────────────────────────

xs = list(range(len(BUDGETS)))   # [0,1,2,3,4,5]
budget_labels = [str(b) for b in BUDGETS]

# Gather range for shared y-axes
all_acc = []
all_brier = []
for ds in DATASETS:
    all_acc.extend(M1_DATA[ds]['acc'])
    all_acc.extend(M2_DATA[ds]['acc'])
    all_brier.extend(M1_DATA[ds]['brier'])
    all_brier.extend(M2_DATA[ds]['brier'])

acc_pad = 0.05 * (max(all_acc) - min(all_acc))
brier_pad = 0.05 * (max(all_brier) - min(all_brier))
acc_ylim   = (min(all_acc)   - acc_pad,   max(all_acc)   + acc_pad)
brier_ylim = (min(all_brier) - brier_pad, max(all_brier) + brier_pad)


# ── figure layout ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 5.8))

# Subplot references
ax_m1_acc,   ax_m1_brier   = axes[0]
ax_m2_acc,   ax_m2_brier   = axes[1]

model_data = [
    ("M1 (Qwen3-8B)", M1_DATA),
    ("M2 (DeepSeek-R1-Distill-Llama-8B)", M2_DATA),
]

for row_idx, (model_label, model_d) in enumerate(model_data):
    ax_acc   = axes[row_idx][0]
    ax_brier = axes[row_idx][1]

    for di, ds in enumerate(DATASETS):
        c = COLORS[di]
        m = MARKERS[di]
        dk = get_dash_kwargs(di)

        # Accuracy (in %)
        acc_vals = [v * 100 for v in model_d[ds]['acc']]
        ax_acc.plot(
            xs, acc_vals,
            marker=m, color=c,
            markerfacecolor='white', markeredgecolor=c, markeredgewidth=1.5,
            label=ds, **dk
        )

        # Brier
        brier_vals = model_d[ds]['brier']
        ax_brier.plot(
            xs, brier_vals,
            marker=m, color=c,
            markerfacecolor='white', markeredgecolor=c, markeredgewidth=1.5,
            label=ds, **dk
        )

    # Shared y-axis limits
    ax_acc.set_ylim(acc_ylim[0] * 100, acc_ylim[1] * 100)
    ax_brier.set_ylim(brier_ylim)

    # x-axis
    for ax in [ax_acc, ax_brier]:
        ax.set_xticks(xs)
        ax.set_xticklabels(budget_labels)
        style_ax(ax)

    # y-axis labels (left column only for each type)
    ax_acc.set_ylabel('Accuracy (%)')
    ax_brier.set_ylabel('Brier ↓')

    # x-axis label only on bottom row
    if row_idx == 1:
        ax_acc.set_xlabel('Reasoning budget (tokens)')
        ax_brier.set_xlabel('Reasoning budget (tokens)')

# ── last-point labels with de-collision ─────────────────────────────────────

def add_end_labels(ax, model_d, metric, scale=1.0, fmt=None):
    """
    Place labels at x=5 (last budget) with leader lines.
    De-collides by nudging overlapping labels apart.
    """
    entries = []
    for di, ds in enumerate(DATASETS):
        vals = model_d[ds][metric]
        y_val = vals[-1] * scale
        if fmt is None:
            if metric == 'acc':
                lbl = f"{y_val:.1f}%"
            else:
                lbl = f"{y_val:.3f}"
        else:
            lbl = fmt(y_val)
        entries.append({'y_data': y_val, 'label': lbl, 'color': COLORS[di]})

    # Sort by y descending for easier collision resolution
    entries.sort(key=lambda e: e['y_data'])

    # Compute min gap in data units: ~8pt
    fig_height_in = fig.get_figheight()
    ax_height_in  = ax.get_position().height * fig_height_in
    ylo, yhi = ax.get_ylim()
    pts_per_data = ax_height_in * 72.0 / (yhi - ylo)   # pts per data unit
    min_gap = 8.0 / pts_per_data   # 8pt in data units

    # Iteratively spread labels
    adjusted = [e['y_data'] for e in entries]
    MAX_ITER = 200
    for _ in range(MAX_ITER):
        moved = False
        for i in range(1, len(adjusted)):
            if adjusted[i] - adjusted[i-1] < min_gap:
                mid = (adjusted[i] + adjusted[i-1]) / 2
                adjusted[i-1] = mid - min_gap / 2
                adjusted[i]   = mid + min_gap / 2
                moved = True
        if not moved:
            break

    # Place text and leader lines
    for i, e in enumerate(entries):
        ax.annotate(
            e['label'],
            xy=(5, e['y_data']),
            xytext=(5.15, adjusted[i]),
            fontsize=6.5,
            va='center', ha='left',
            color=e['color'],
            arrowprops=dict(
                arrowstyle='-',
                color=e['color'],
                lw=0.6,
                connectionstyle='arc3,rad=0',
            ),
        )

# Need to call tight_layout first to finalize axis geometry, then add labels
fig.tight_layout(rect=[0.06, 0.08, 1.0, 0.97])

for row_idx, (model_label, model_d) in enumerate(model_data):
    ax_acc   = axes[row_idx][0]
    ax_brier = axes[row_idx][1]
    add_end_labels(ax_acc,   model_d, 'acc',   scale=100)
    add_end_labels(ax_brier, model_d, 'brier', scale=1.0)

# ── row labels ───────────────────────────────────────────────────────────────

for row_idx, (model_label, _) in enumerate(model_data):
    # Position row label at left margin, centered vertically in the row
    row_axes = axes[row_idx]
    y0 = row_axes[0].get_position().y0
    y1 = row_axes[0].get_position().y1
    yc = (y0 + y1) / 2
    fig.text(
        0.01, yc, model_label,
        ha='center', va='center',
        rotation=90,
        fontsize=8,
        fontweight='bold',
    )

# ── single legend at bottom ───────────────────────────────────────────────────

# Collect handles from one subplot (all have same datasets)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=4,
    bbox_to_anchor=(0.5, 0.0),
    fontsize=7,
    frameon=False,
    handlelength=2.5,
)

# ── save ─────────────────────────────────────────────────────────────────────

out_dir = os.path.dirname(__file__)
pdf_path = os.path.join(out_dir, 'fig_budget_response.pdf')
png_path = os.path.join(out_dir, 'fig_budget_response.png')

fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
