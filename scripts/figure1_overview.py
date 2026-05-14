"""
Figure 1: Method Overview (Nature Methods style)
- 1a: Poincaré disk embedding (alpha=0.6, Set2, degree markers)
- 1b: Spatial coordinates
- 1c: Radial distribution (density curve + boxplot)
- 1d: Cell types on disk
- 1e: Distance fidelity (regression + 95% CI + Pearson r)
- 1f: Algorithm schematic (3-step horizontal flow)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scanpy as sc
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr, gaussian_kde
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figure_style import (setup_style, label_panel, draw_poincare_disk, save_figure,
                          CT_COLORS, CLUSTER_CMAP, draw_degree_markers,
                          add_regression_with_ci, format_pvalue)

setup_style()

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'spatial_data', '21_00731_hyperbolic_v3.h5ad')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

SIGNATURES = {
    'Epithelial':  ['EPCAM', 'KRT18', 'KRT19', 'KRT8'],
    'Fibroblast':  ['COL1A1', 'COL1A2', 'DCN', 'LUM'],
    'T_cell':      ['CD3D', 'CD3E', 'CD2'],
    'Macrophage':  ['CD68', 'C1QA', 'C1QB', 'C1QC'],
    'Endothelial': ['VWF', 'CDH5', 'ENG'],
    'CAF_m':       ['FAP', 'POSTN', 'ACTA2', 'MMP2'],
}


def compute_module_scores(adata, signatures):
    scores = {}
    for name, genes in signatures.items():
        avail = [g for g in genes if g in adata.var_names]
        if not avail:
            scores[name] = np.zeros(adata.n_obs)
            continue
        idx = [list(adata.var_names).index(g) for g in avail]
        expr = adata.X[:, idx]
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        raw = expr.mean(axis=1)
        scores[name] = (raw - raw.mean()) / (raw.std() + 1e-8)
    return scores


def plot_distance_fidelity(ax, D_pca, D_hyp, r_pearson):
    n = D_pca.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    x = D_pca[mask]
    y = D_hyp[mask]

    rng = np.random.RandomState(42)
    n_plot = min(50000, len(x))
    idx = rng.choice(len(x), n_plot, replace=False)
    ax.scatter(x[idx], y[idx], s=0.3, alpha=0.05, c='#332288', rasterized=True)

    add_regression_with_ci(ax, x, y, color='#CC6677')

    r, p = r_pearson
    n_pairs = len(x)
    r_clipped = np.clip(r, -0.9999, 0.9999)
    z = np.arctanh(r_clipped)
    se = 1 / np.sqrt(n_pairs - 3)
    r_lo = np.tanh(z - 1.96 * se)
    r_hi = np.tanh(z + 1.96 * se)

    p_str = format_pvalue(p)
    text = f'Pearson r = {r:.3f} [{r_lo:.3f}, {r_hi:.3f}]\n{p_str}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=5.5,
            va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', edgecolor='0.7', alpha=0.9))

    ax.set_xlabel('PCA Euclidean distance', fontsize=7)
    ax.set_ylabel('Poincaré geodesic distance', fontsize=7)


def plot_algorithm_schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    C_BLUE = '#332288'
    C_GREEN = '#228833'
    C_RED = '#CC6677'

    # Three boxes in a horizontal row
    box_w, box_h = 2.8, 2.6
    y_center = 2.0
    positions = [(0.3, y_center - box_h/2), (3.6, y_center - box_h/2), (6.9, y_center - box_h/2)]
    colors = [C_BLUE, C_GREEN, C_RED]
    titles = ['Step 1', 'Step 2', 'Step 3']
    subtitles = ['k-NN Graph', 'Torgerson Init', 'Riemannian Adam']
    descriptions = [
        'k=30 NN in PCA space\n'
        'Shortest-path distances\n'
        'via Dijkstra algorithm',
        'Classical MDS:\n'
        'B = -0.5 H D^2 H\n'
        'Top-2 eigenvectors',
        'Minimize stress on\n'
        'Poincaré ball manifold\n'
        '+ repulsion loss',
    ]

    theta = np.linspace(0, 2 * np.pi, 200)

    for i, (x0, y0) in enumerate(positions):
        # Box
        box = FancyBboxPatch((x0, y0), box_w, box_h, boxstyle="round,pad=0.15",
                             facecolor='white', edgecolor=colors[i], linewidth=1.0)
        ax.add_patch(box)

        # Mini disk at top of box
        cx = x0 + box_w / 2
        cy = y0 + box_h - 0.55
        r_disk = 0.35
        ax.plot(cx + r_disk * np.cos(theta), cy + r_disk * np.sin(theta),
                lw=0.4, color='0.7')

        rng = np.random.RandomState(42 + i * 7)
        n_pts = 8
        angles = rng.uniform(0, 2 * np.pi, n_pts)
        if i == 0:
            radii = rng.uniform(0.1, 0.25, n_pts)
        elif i == 1:
            radii = rng.uniform(0.05, 0.15, n_pts)
        else:
            radii = np.concatenate([rng.uniform(0.05, 0.15, 3),
                                    rng.uniform(0.15, 0.3, 5)])
        px = cx + radii * np.cos(angles)
        py = cy + radii * np.sin(angles)
        ax.scatter(px, py, s=8, c=colors[i], zorder=3, edgecolors='white', linewidths=0.2)

        # Text
        ax.text(cx, y0 + box_h - 1.1, titles[i], fontsize=7, fontweight='bold',
                color=colors[i], ha='center', va='center')
        ax.text(cx, y0 + box_h - 1.45, subtitles[i], fontsize=6.5, fontweight='bold',
                color='#222222', ha='center', va='center')
        ax.text(cx, y0 + 0.55, descriptions[i], fontsize=5.5, color='#444444',
                ha='center', va='center', linespacing=1.4)

    # Arrows between boxes
    for i in range(2):
        x_start = positions[i][0] + box_w + 0.15
        x_end = positions[i + 1][0] - 0.15
        y_arrow = y_center
        ax.annotate('', xy=(x_end, y_arrow), xytext=(x_start, y_arrow),
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))


def main():
    print("=== Figure 1: Method Overview (Nature Methods) ===")

    print("  Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    emb = adata.obsm['X_poincare']
    spatial = adata.obsm['spatial']
    pca_coords = adata.obsm['X_pca']
    norms = np.linalg.norm(emb, axis=1)
    clusters = adata.obs['cluster'].values
    unique_clusters = sorted(clusters.unique())

    scores = compute_module_scores(adata, SIGNATURES)

    n_sub = min(2000, len(emb))
    idx = np.random.RandomState(42).choice(len(emb), n_sub, replace=False)
    D_hyp = squareform(pdist(emb[idx]))
    D_pca = squareform(pdist(pca_coords[idx]))
    r_pearson = pearsonr(D_hyp.ravel(), D_pca.ravel())

    # Equal-sized 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.subplots_adjust(wspace=0.45, hspace=0.50)

    # Panel a: Poincaré disk
    ax = axes[0, 0]
    draw_poincare_disk(ax)
    for i, cl in enumerate(unique_clusters):
        mask = clusters == cl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[CLUSTER_CMAP(i / len(unique_clusters))], s=12, alpha=0.6,
                   label=f'C{cl}', rasterized=True, edgecolors='white', linewidths=0.2)
    ax.legend(fontsize=4.5, ncol=2, loc='upper right', markerscale=1.5,
              handletextpad=0.2, columnspacing=0.4, borderpad=0.15)
    ax.set_xlabel('Poincaré $x_1$', fontsize=7)
    ax.set_ylabel('Poincaré $x_2$', fontsize=7)
    draw_degree_markers(ax)
    ax.text(0, -1.18, 'Angular position has no analytical significance; only radius encodes hierarchy',
            ha='center', va='top', fontsize=4, color='0.5', fontstyle='italic')
    label_panel(ax, 'a')

    # Panel b: Spatial coordinates
    ax = axes[0, 1]
    for i, cl in enumerate(unique_clusters):
        mask = clusters == cl
        ax.scatter(spatial[mask, 0], spatial[mask, 1],
                   c=[CLUSTER_CMAP(i / len(unique_clusters))], s=12, alpha=0.6,
                   rasterized=True, edgecolors='white', linewidths=0.2)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('Spatial $x$', fontsize=7)
    ax.set_ylabel('Spatial $y$', fontsize=7)
    label_panel(ax, 'b')

    # Panel c: Radial distribution (clean density + boxplot)
    ax = axes[0, 2]
    kde = gaussian_kde(norms)
    x_range = np.linspace(norms.min(), norms.max(), 200)
    ax.fill_between(x_range, kde(x_range), alpha=0.15, color='#332288')
    ax.plot(x_range, kde(x_range), '-', color='#332288', lw=1.2, label='All spots')

    # Boxplot at bottom
    bp = ax.boxplot([norms], positions=[0], widths=0.25, vert=False, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='black', lw=0.8),
                    whiskerprops=dict(lw=0.5, color='0.3'),
                    capprops=dict(lw=0.5, color='0.3'))
    for patch in bp['boxes']:
        patch.set_facecolor('#332288')
        patch.set_alpha(0.3)
        patch.set_edgecolor('0.3')

    median = np.percentile(norms, 50)
    q25 = np.percentile(norms, 25)
    q75 = np.percentile(norms, 75)
    ax.axvline(median, color='#CC6677', ls='--', lw=0.8, zorder=0)
    ax.axvline(q25, color='#44AA99', ls=':', lw=0.6, zorder=0)
    ax.axvline(q75, color='#44AA99', ls=':', lw=0.6, zorder=0)

    ymax = ax.get_ylim()[1]
    ax.text(median, ymax * 0.92, f'Median {median:.2f}', ha='center', va='top',
            fontsize=5, color='#CC6677')
    ax.text(q25, ymax * 0.82, f'Q1 {q25:.2f}', ha='center', va='top',
            fontsize=4.5, color='#44AA99')
    ax.text(q75, ymax * 0.82, f'Q3 {q75:.2f}', ha='center', va='top',
            fontsize=4.5, color='#44AA99')

    ax.set_xlabel('Poincaré radius', fontsize=7)
    ax.set_ylabel('Density', fontsize=7)
    ax.legend(fontsize=5, loc='upper right')
    label_panel(ax, 'c')

    # Panel d: Cell types on disk
    ax = axes[1, 0]
    draw_poincare_disk(ax)
    for ct, color in CT_COLORS.items():
        s = scores[ct]
        thresh = np.percentile(s, 70)
        mask = s > thresh
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=12, alpha=0.6,
                   label=ct.replace('_', ' '), rasterized=True, edgecolors='white', linewidths=0.2)
    ax.legend(fontsize=4.5, loc='upper right', markerscale=1.5,
              handletextpad=0.2, borderpad=0.15)
    ax.set_xlabel('Poincaré $x_1$', fontsize=7)
    ax.set_ylabel('Poincaré $x_2$', fontsize=7)
    label_panel(ax, 'd')

    # Panel e: Distance fidelity
    ax = axes[1, 1]
    plot_distance_fidelity(ax, D_pca, D_hyp, r_pearson)
    label_panel(ax, 'e')

    # Panel f: Algorithm schematic
    ax = axes[1, 2]
    plot_algorithm_schematic(ax)
    label_panel(ax, 'f')

    save_figure(fig, 'Figure1_overview', FIG_DIR)
    print("=== Done ===")


if __name__ == '__main__':
    main()
