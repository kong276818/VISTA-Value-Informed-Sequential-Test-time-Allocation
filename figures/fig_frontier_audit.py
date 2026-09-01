"""
Figure: Quality–Compute Frontier and rho-sensitivity.
1×2: panel (a) M1/DS1 frontier, panel (b) audit-rate sensitivity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from vista_data import ORACLE_C, RHO_SENSITIVITY, POLICY_COMPARISON
from vista_style import apply_style, style_ax, COLORS, FIG_WIDTH

apply_style()

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.2))

# ============================================================================
# Panel (a): Quality–Compute Frontier (M1/DS1 = MMLU-Pro)
# ============================================================================

# --- data points (from Table 8 + Oracle C Table 5) ---

points = [
    {
        'name':     'Best fixed (b=8192)',
        'tokens':   8192,
        'brier':    0.492,
        'marker':   'o',
        'color':    '#3a7fd5',
        'ms':       6,
        'is_threshold': False,
    },
    {
        'name':     'Confidence stopping',
        'tokens':   1453,
        'brier':    0.664,
        'marker':   'v',
        'color':    '#d8720f',
        'ms':       6,
        'is_threshold': True,
    },
    {
        'name':     'Entropy stopping',
        'tokens':   1624,
        'brier':    0.643,
        'marker':   'v',
        'color':    '#006837',
        'ms':       6,
        'is_threshold': True,
    },
    {
        'name':     'Stability stopping',
        'tokens':   1655,
        'brier':    0.608,
        'marker':   'v',
        'color':    '#7e2c86',
        'ms':       6,
        'is_threshold': True,
    },
    {
        'name':     r'VISTA ($\lambda$=0)',
        'tokens':   8191,
        'brier':    0.492,
        'marker':   's',
        'color':    '#3a7fd5',
        'ms':       6,
        'is_threshold': False,
    },
    {
        'name':     r'VISTA ($\lambda$=0.01)',
        'tokens':   7931,
        'brier':    0.498,
        'marker':   's',
        'color':    '#d8720f',
        'ms':       6,
        'is_threshold': False,
    },
    {
        'name':     'Oracle C',
        'tokens':   461,
        'brier':    0.4911,
        'marker':   '*',
        'color':    '#7e2c86',
        'ms':       12,
        'is_threshold': False,
    },
]

# Plot all points
for p in points:
    ax_a.scatter(
        p['tokens'], p['brier'],
        marker=p['marker'],
        s=p['ms']**2,
        color=p['color'],
        facecolors='white',
        edgecolors=p['color'],
        linewidths=1.5,
        zorder=5,
    )

# --- threshold cluster: ellipse bracket ---
thresh_pts = [p for p in points if p['is_threshold']]
tx = [p['tokens'] for p in thresh_pts]
ty = [p['brier']  for p in thresh_pts]
cx = np.mean(tx)
cy = np.mean(ty)
width_log = np.log10(max(tx)) - np.log10(min(tx))
# Ellipse in data coordinates (log scale x handled via transform)
# Use a FancyBboxPatch in axes coordinates
ax_a.set_xscale('log')  # set log before computing transforms

# Draw ellipse around threshold cluster in log-x space
from matplotlib.patches import Ellipse
# in log space: center at log10(cx), width covers log10(max)-log10(min)+margin
# We need to draw in data units after setting log scale
ell_x = cx
ell_y = cy
ell_w = (max(tx) - min(tx)) * 2.5  # data units in log space, approximate
ell_h = (max(ty) - min(ty)) * 4.0

ell = Ellipse(
    (ell_x, ell_y),
    width=ell_w,
    height=ell_h,
    fill=False,
    edgecolor='#888888',
    linewidth=0.8,
    linestyle='--',
    zorder=3,
)
ax_a.add_patch(ell)

# Annotation for threshold cluster — text kept below y=0.73
ax_a.annotate(
    'Threshold stopping',
    xy=(cx, max(ty) + 0.010),
    xytext=(600, 0.718),
    xycoords='data',
    textcoords='data',
    fontsize=6.5,
    color='#555555',
    arrowprops=dict(
        arrowstyle='->',
        color='#888888',
        lw=0.8,
    ),
    ha='center',
)

# --- horizontal dashed line at best-fixed Brier ---
best_brier = 0.492
ax_a.axhline(best_brier, color='#3a7fd5', linewidth=0.8, linestyle='--', zorder=2)
ax_a.text(
    420, best_brier - 0.013,
    'best-fixed Brier',
    fontsize=6, color='#3a7fd5', ha='left', va='top',
)

# --- individual labels (all in data coordinates to avoid crowding) ---
# Right-side cluster (x≈8000) stacked vertically at x≈4800;
# Oracle C annotated close to its point.
label_specs = [
    # (xy_data, xytext_data, label_text, ha, va)
    ((8192, 0.492),  (4800, 0.476),  'Best fixed\n(b=8192)',   'center', 'top'),
    ((8191, 0.492),  (4800, 0.520),  r'VISTA ($\lambda$=0)',    'center', 'center'),
    ((7931, 0.498),  (4800, 0.498),  r'VISTA ($\lambda$=0.01)', 'center', 'center'),
    ((461,  0.4911), (700,  0.507),  'Oracle C',                'left',   'center'),
]
for (xy_d, xytext_d, lbl, ha2, va2) in label_specs:
    ax_a.annotate(
        lbl,
        xy=xy_d,
        xytext=xytext_d,
        xycoords='data',
        textcoords='data',
        fontsize=6.5,
        ha=ha2, va=va2,
        color='#333333',
        arrowprops=dict(
            arrowstyle='-',
            color='#aaaaaa',
            lw=0.6,
            shrinkA=2, shrinkB=3,
        ),
    )

# --- axes ---
ax_a.set_xlabel('Mean tokens (log)')
ax_a.set_ylabel('Brier ↓')
ax_a.set_title('(a) Quality–Compute Frontier\n(M1 / MMLU-Pro)', fontsize=8)

# y range: show all points with margin
all_brier_vals = [p['brier'] for p in points]
y_lo = min(all_brier_vals) - 0.04
y_hi = max(all_brier_vals) + 0.12
ax_a.set_ylim(y_lo, y_hi)

style_ax(ax_a)
# Keep log scale x with explicit ticks to avoid crowded auto-labels
ax_a.set_xscale('log')
ax_a.set_xlim(380, 12000)
ax_a.set_xticks([500, 1000, 2000, 4000, 8000])
ax_a.set_xticklabels(['500', '1k', '2k', '4k', '8k'])
ax_a.tick_params(axis='x', which='minor', length=2, width=0.4, colors='#aaaaaa')

# ============================================================================
# Panel (b): rho sensitivity (Table 11)
# ============================================================================

rho_configs = [
    ('M1/MMLU-Pro',      {'color': COLORS[0], 'marker': 'o', 'linestyle': '-'}),
    ('M1/ARC-Challenge', {'color': COLORS[1], 'marker': 's', 'dashes': (6, 2)}),
    ('M1/MedMCQA',       {'color': COLORS[2], 'marker': '^', 'linestyle': ':'}),
    ('M2/MMLU-Pro',      {'color': COLORS[3], 'marker': 'D', 'dashes': (6, 2, 2, 2)}),
]

near_zero_artists = []

for cell_name, style_kw in rho_configs:
    cell = RHO_SENSITIVITY[cell_name]
    rho        = cell['rho']
    net_saving = cell['net_saving']
    ci_lo      = cell['ci_lo']
    ci_hi      = cell['ci_hi']
    b_ref      = cell['b_ref']

    label = f"{cell_name} (b_ref={b_ref})"

    # Build line style
    ls_kw = {}
    if 'linestyle' in style_kw:
        ls_kw['linestyle'] = style_kw['linestyle']
    elif 'dashes' in style_kw:
        ls_kw['dashes'] = style_kw['dashes']

    line, = ax_b.plot(
        rho, net_saving,
        marker=style_kw['marker'],
        color=style_kw['color'],
        markerfacecolor='white',
        markeredgecolor=style_kw['color'],
        markeredgewidth=1.5,
        label=label,
        **ls_kw,
    )

    # 95% CI shading
    lo = [net_saving[i] - abs(net_saving[i] - ci_lo[i]) for i in range(len(rho))]
    hi = [net_saving[i] + abs(ci_hi[i] - net_saving[i]) for i in range(len(rho))]
    ax_b.fill_between(
        rho, lo, hi,
        color=style_kw['color'],
        alpha=0.12,
        zorder=1,
    )

    # Collect near-zero lines for annotation
    if b_ref == 8192:
        near_zero_artists.append((rho, net_saving, style_kw['color']))

# y=0 reference line
ax_b.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=2)

# Annotation for b_ref=B_MAX cluster near y=0
# Pick midpoint of rho range for annotation x
ann_x = 0.12
ann_y = 0.003
ax_b.annotate(
    'b_ref = B_max:\naudit cost negligible',
    xy=(0.10, 0.001),
    xytext=(0.22, 0.15),
    fontsize=6.5,
    color='#333333',
    arrowprops=dict(
        arrowstyle='->',
        color='#888888',
        lw=0.8,
    ),
    ha='left',
)

# axes
ax_b.set_xlabel(r'Audit rate $\rho$')
ax_b.set_ylabel(r'Net saving ($\lambda$=0)')
ax_b.set_title(r'(b) $\rho$ Sensitivity', fontsize=8)
ax_b.set_xscale('log')
ax_b.tick_params(axis='x', which='minor', length=2, width=0.4, colors='#aaaaaa')

style_ax(ax_b)
ax_b.set_xscale('log')  # reapply after style_ax

ax_b.legend(
    loc='lower left',
    fontsize=6,
    frameon=False,
)

# ── save ─────────────────────────────────────────────────────────────────────

plt.tight_layout()

out_dir = os.path.dirname(__file__)
pdf_path = os.path.join(out_dir, 'fig_frontier_audit.pdf')
png_path = os.path.join(out_dir, 'fig_frontier_audit.png')

fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
