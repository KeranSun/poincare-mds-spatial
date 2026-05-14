"""
Nature Methods figure style — shared constants and utilities.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# ============================================================
# Style configuration (Nature Methods)
# ============================================================
def setup_style():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    matplotlib.rcParams['font.size'] = 7
    matplotlib.rcParams['axes.linewidth'] = 0.6
    matplotlib.rcParams['xtick.major.width'] = 0.6
    matplotlib.rcParams['ytick.major.width'] = 0.6
    matplotlib.rcParams['xtick.major.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 3
    matplotlib.rcParams['xtick.major.pad'] = 2
    matplotlib.rcParams['ytick.major.pad'] = 2
    matplotlib.rcParams['axes.labelpad'] = 3
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['legend.fontsize'] = 6
    matplotlib.rcParams['figure.dpi'] = 150

# ============================================================
# Color palettes
# ============================================================
TOL_BRIGHT = {
    'blue':    '#4477AA',
    'cyan':    '#66CCEE',
    'green':   '#228833',
    'yellow':  '#CCBB44',
    'red':     '#EE6677',
    'purple':  '#AA3377',
    'grey':    '#BBBBBB',
}

METHOD_COLORS = {
    'Poincaré MDS': '#332288',
    'PHATE':        '#44AA99',
    'Euclidean MDS': '#CC6677',
    't-SNE':        '#DDCC77',
}

CT_COLORS = {
    'Epithelial':  '#E64B35',
    'Fibroblast':  '#00A087',
    'T_cell':      '#FF7F00',
    'Macrophage':  '#8491B4',
    'Endothelial': '#91D1C2',
    'CAF_m':       '#4DBBD5',
}

ZONE_COLORS = {
    'inner':  '#CC6677',
    'middle': '#999933',
    'outer':  '#44AA99',
}

STAGE_COLORS = {
    'Progenitor':     '#332288',
    'Intermediate':   '#44AA99',
    'Mature_SATB2':   '#CC6677',
    'Mature_TBR1':    '#999933',
}

LAYER_COLORS = {
    'Granule':     '#332288',
    'Purkinje':    '#44AA99',
    'Molecular':   '#CC6677',
    'WhiteMatter': '#DDCC77',
}

CLUSTER_CMAP = plt.cm.Set2

# ============================================================
# Utility functions
# ============================================================
def label_panel(ax, label, x=-0.15, y=1.08):
    """Nature style: bold lowercase, outside top-left."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left')

def draw_poincare_disk(ax, lw=0.5, color='0.5'):
    """Draw the unit circle boundary of the Poincaré disk."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '-', color=color, lw=lw)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')

def save_figure(fig, name, fig_dir):
    """Save figure as both PDF and PNG at 300 dpi."""
    os.makedirs(fig_dir, exist_ok=True)
    pdf_path = os.path.join(fig_dir, f'{name}.pdf')
    png_path = os.path.join(fig_dir, f'{name}.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    plt.close(fig)


def add_regression_with_ci(ax, x, y, color='#CC6677', ci=95, label=None):
    """Add linear regression line with CI band."""
    from scipy.stats import pearsonr
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # Prediction interval
    n = len(x)
    x_mean = x.mean()
    se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    t_val = stats.t.ppf(1 - (1 - ci/100) / 2, n - 2)
    ci_band = t_val * se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))

    ax.plot(x_line, y_line, '-', color=color, lw=1.5, label=label)
    ax.fill_between(x_line, y_line - ci_band, y_line + ci_band,
                     color=color, alpha=0.15, zorder=0)


def format_pvalue(p, threshold=0.001):
    """Format p-value for annotation using matplotlib mathtext."""
    if p == 0 or p < 1e-300:
        return r'$p < 10^{-300}$'
    if p < threshold:
        exp = int(np.floor(np.log10(max(p, 1e-300))))
        return r'$p < 10^{' + str(exp) + '}$'
    return f'p = {p:.3f}'


def violin_boxplot(ax, data, positions, colors, alpha=0.4):
    """Violin + boxplot + jitter overlay (Nature Methods style)."""
    for i, (d, pos, col) in enumerate(zip(data, positions, colors)):
        # Violin
        parts = ax.violinplot([d], positions=[pos], showmeans=False,
                               showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_alpha(alpha)
            pc.set_edgecolor('0.3')
            pc.set_linewidth(0.5)

        # Boxplot overlay
        bp = ax.boxplot([d], positions=[pos], widths=0.15, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color='black', lw=0.8),
                        whiskerprops=dict(lw=0.5, color='0.3'),
                        capprops=dict(lw=0.5, color='0.3'))
        for patch in bp['boxes']:
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            patch.set_edgecolor('0.3')
            patch.set_linewidth(0.5)

        # Jitter points
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(d))
        ax.scatter(np.full_like(d, pos) + jitter, d, s=3, c='0.4',
                   alpha=0.3, zorder=3, edgecolors='none')


def add_stat_annotation(ax, x1, x2, y, pval, text_y=None):
    """Add significance bar with p-value."""
    if text_y is None:
        text_y = y + 0.02
    ax.plot([x1, x1, x2, x2], [y, y+0.01, y+0.01, y], lw=0.6, color='0.3')
    p_str = format_pvalue(pval)
    ax.text((x1 + x2) / 2, text_y, p_str, ha='center', va='bottom', fontsize=5)


def draw_degree_markers(ax, r=1.0, fontsize=5):
    """Add degree markers (0, 90, 180, 270) on Poincaré disk boundary."""
    for angle, label in [(0, '0°'), (90, '90°'), (180, '180°'), (270, '270°')]:
        rad = np.radians(angle)
        x = r * 1.08 * np.cos(rad)
        y = r * 1.08 * np.sin(rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                color='0.5', fontstyle='italic')
