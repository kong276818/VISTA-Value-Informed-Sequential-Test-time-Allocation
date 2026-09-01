"""
Common matplotlib style for MVT-CS figures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Path to DejaVu Sans TTF (confirmed present)
_DEJAVU_SANS_TTF = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
_DEJAVU_SANS_BOLD_TTF = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# Register the font with matplotlib if not already present
def _register_dejavu():
    """Register DejaVu Sans TTF with matplotlib font manager."""
    # Add the system font directory so matplotlib finds the TTF
    fm.fontManager.addfont(_DEJAVU_SANS_TTF)
    if _DEJAVU_SANS_BOLD_TTF:
        try:
            fm.fontManager.addfont(_DEJAVU_SANS_BOLD_TTF)
        except Exception:
            pass

_register_dejavu()

# Color palette: [MMLU-Pro, ARC-Challenge, MedMCQA, MedQA-USMLE]
COLORS = ['#3a7fd5', '#d8720f', '#006837', '#7e2c86']

# Markers for 4 datasets
MARKERS = ['o', 's', '^', 'D']

# Dash patterns for 4 datasets: (None,None) = solid
DASHES = [(None, None), (6, 2), (2, 2), (6, 2, 2, 2)]

# Dataset names in order
DATASETS = ['MMLU-Pro', 'ARC-Challenge', 'MedMCQA', 'MedQA-USMLE']

# Figure width
FIG_WIDTH = 7.1


def apply_style():
    """Set global rcParams for paper-quality figures."""
    plt.rcParams.update({
        # Font — DejaVu Sans TTF (prevents Type 3)
        'font.family':       'DejaVu Sans',
        'font.size':         8,
        'axes.titlesize':    9,
        'axes.labelsize':    8,
        'xtick.labelsize':   7,
        'ytick.labelsize':   7,
        'legend.fontsize':   7,

        # PDF/PS: embed as Type 42 (TrueType), not Type 3
        'pdf.fonttype':      42,
        'ps.fonttype':       42,

        # Save options
        'savefig.dpi':       400,
        'savefig.bbox':      'tight',

        # Axes appearance
        'axes.spines.top':   False,
        'axes.spines.right': False,

        # Grid
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'grid.color':        '#cccccc',
        'grid.linewidth':    0.5,
        'grid.linestyle':    '-',

        # Ticks
        'xtick.color':       '#888888',
        'ytick.color':       '#888888',
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,

        # Lines
        'lines.linewidth':   1.4,
        'lines.markersize':  5,

        # Legend
        'legend.frameon':    False,
    })


def style_ax(ax):
    """Apply spine/grid style to a single axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#cccccc', linewidth=0.5, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', colors='#888888', length=3, width=0.6)


def get_dash_kwargs(idx):
    """Return linestyle kwargs for dash pattern at index idx."""
    d = DASHES[idx]
    if d[0] is None:
        return {'linestyle': '-'}
    else:
        return {'dashes': d}
