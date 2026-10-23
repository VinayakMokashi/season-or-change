"""Shared figure style: one colour per method, fixed across every figure."""
import matplotlib as mpl

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
PAPER = '#e34948'  # published numbers, used as reference marks

# Temporal methods take the first categorical slots in a fixed order; MoCo the
# fourth. Reference baselines (random init, ImageNet) are neutral greys.
COLORS = {
    'SeCo': '#2a78d6',
    'TeCo': '#eb6834',
    'CACo': '#1baf7a',
    'MoCo': '#eda100',
    'ImageNet': '#52514e',
    'Random': '#c3c2b7',
}


def apply():
    mpl.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
        'font.size': 9,
        'text.color': INK,
        'axes.labelcolor': INK_2,
        'axes.edgecolor': AXIS,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK_2,
        'ytick.labelcolor': INK_2,
        'legend.frameon': False,
        'lines.linewidth': 2,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
