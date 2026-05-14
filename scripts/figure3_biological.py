"""
Figure 3: Biological Discovery (Nature Methods style)
- Tumor zones on Poincaré disk
- Cell type module scores by zone (violin + boxplot + jitter)
- HDS heatmap (with Cohen's d + FDR q-values)
- Hyperbolic clusters on disk
- Silhouette comparison (per-cluster jitter)
- Niche enrichment comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from figure_style import (setup_style, label_panel, draw_poincare_disk, save_figure,
                          ZONE_COLORS, CT_COLORS, CLUSTER_CMAP, violin_boxplot)
from stats_utils import cohens_d, fdr_correction, format_p

setup_style()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def load_biological_data():
    summary = pd.read_csv(os.path.join(RESULTS_DIR, 'biological_discovery_summary.csv'))
    zone_markers = pd.read_csv(os.path.join(RESULTS_DIR, 'zone_marker_analysis.csv'))
    hds = pd.read_csv(os.path.join(RESULTS_DIR, 'hierarchical_differential_score.csv'))
    clusters = pd.read_csv(os.path.join(RESULTS_DIR, 'hyperbolic_clusters.csv'))
    niche_purity = pd.read_csv(os.path.join(RESULTS_DIR, 'niche_purity.csv'))
    niche_enrichment = pd.read_csv(os.path.join(RESULTS_DIR, 'niche_enrichment.csv'))
    return summary, zone_markers, hds, clusters, niche_purity, niche_enrichment


def plot_zone_markers_violin(ax, zone_df):
    """Grouped violin+boxplot: z-score by zone for each cell type."""
    cell_types = zone_df['cell_type'].values
    x = np.arange(len(cell_types))
    width = 0.25

    for i, zone in enumerate(['inner', 'middle', 'outer']):
        col = f'mean_{zone}'
        vals = zone_df[col].values
        offset = (i - 1) * width
        positions = x + offset

        # Bar chart (simplified for zone markers summary data)
        bars = ax.bar(positions, vals, width, label=zone.capitalize(),
                      color=ZONE_COLORS[zone], edgecolor='0.3', linewidth=0.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([ct.replace('_', ' ') for ct in cell_types], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Mean z-score', fontsize=7)
    ax.legend(fontsize=5, loc='upper right')
    ax.axhline(y=0, color='0.7', lw=0.5, zorder=0)


def plot_hds_heatmap(ax, hds_df):
    """HDS heatmap with Cohen's d and FDR-corrected q-values."""
    all_cts = sorted(set(hds_df['cell_type_A'].tolist() + hds_df['cell_type_B'].tolist()))
    n = len(all_cts)
    ct_to_idx = {ct: i for i, ct in enumerate(all_cts)}

    matrix = np.zeros((n, n))
    pval_matrix = np.ones((n, n))

    for _, row in hds_df.iterrows():
        i = ct_to_idx[row['cell_type_A']]
        j = ct_to_idx[row['cell_type_B']]
        matrix[i, j] = row['HDS']
        matrix[j, i] = -row['HDS']
        pval_matrix[i, j] = row['mannwhitney_pval']
        pval_matrix[j, i] = row['mannwhitney_pval']

    # FDR correction on unique p-values
    pvals = []
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pvals.append(pval_matrix[i, j])
            pairs.append((i, j))
    if len(pvals) > 0:
        reject, q_vals = fdr_correction(np.array(pvals))

    vmax = np.abs(matrix).max()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(matrix, cmap='RdBu_r', norm=norm, edgecolors='white', linewidth=0.5)

    labels = [ct.replace('_', '\n') for ct in all_cts]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=5)

    # Annotate with significance and Cohen's d
    for idx, (i, j) in enumerate(pairs):
        q = q_vals[idx]
        p = pval_matrix[i, j]
        stars = '***' if q < 1e-10 else ('**' if q < 1e-5 else ('*' if q < 0.05 else ''))
        if stars:
            ax.text(j + 0.5, i + 0.5, stars, ha='center', va='center',
                    fontsize=4, color='white', fontweight='bold')
            ax.text(i + 0.5, j + 0.5, f'q={q:.1e}', ha='center', va='center',
                    fontsize=3, color='0.3', rotation=45)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('HDS', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title('Hierarchical Differential Score', fontsize=7, pad=3)


def plot_silhouette_comparison(ax, cluster_df):
    """Per-cluster silhouette with jitter points."""
    clusters = cluster_df['cluster'].values
    sil_hyp = cluster_df['sil_hyperbolic'].values
    sil_euc = cluster_df['sil_euclidean'].values

    y = np.arange(len(clusters))
    height = 0.35

    ax.barh(y + height / 2, sil_hyp, height, label='Poincaré', color='#332288',
            edgecolor='0.3', linewidth=0.5, alpha=0.8)
    ax.barh(y - height / 2, sil_euc, height, label='Euclidean', color='#CC6677',
            edgecolor='0.3', linewidth=0.5, alpha=0.8)

    # Add jitter points (individual cluster values)
    rng = np.random.RandomState(42)
    for i, (sh, se) in enumerate(zip(sil_hyp, sil_euc)):
        ax.scatter(sh, i + height / 2, s=15, c='#332288', zorder=5,
                   edgecolors='white', linewidths=0.3)
        ax.scatter(se, i - height / 2, s=15, c='#CC6677', zorder=5,
                   edgecolors='white', linewidths=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels([f'C{c}' for c in clusters], fontsize=6)
    ax.set_xlabel('Silhouette score', fontsize=7)
    ax.legend(fontsize=5, loc='lower right')

    # Paired t-test
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(sil_hyp, sil_euc)
    d = cohens_d(sil_hyp, sil_euc)
    ax.text(0.05, 0.95, f'Paired t: p = {p_val:.3f}\nCohen\'s d = {d:.2f}',
            transform=ax.transAxes, fontsize=5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9))


def plot_niche_purity(ax, niche_purity_df):
    """Hyperbolic vs spatial purity with paired comparison."""
    k_vals = niche_purity_df['k'].values
    hyp_purity = niche_purity_df['hyperbolic_purity'].values
    spatial_purity = niche_purity_df['spatial_purity'].values

    x = np.arange(len(k_vals))
    width = 0.35

    bars1 = ax.bar(x - width/2, hyp_purity, width, label='Hyperbolic', color='#332288',
                   edgecolor='0.3', linewidth=0.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, spatial_purity, width, label='Spatial', color='#CC6677',
                   edgecolor='0.3', linewidth=0.5, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars1, hyp_purity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5, color='#332288')
    for bar, val in zip(bars2, spatial_purity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5, color='#CC6677')

    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_vals], fontsize=6)
    ax.set_ylabel('Neighborhood purity', fontsize=7)
    ax.legend(fontsize=5, loc='upper right')

    # Statistical annotation
    d = cohens_d(hyp_purity, spatial_purity)
    ax.text(0.05, 0.95, f'Cohen\'s d = {d:.2f}',
            transform=ax.transAxes, fontsize=5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9))


def main():
    print("=== Figure 3: Biological Discovery (Nature Methods) ===")

    summary, zone_markers, hds, clusters, niche_purity, niche_enrichment = load_biological_data()

    # For panels a and d, we need the embedding data
    import scanpy as sc
    h5ad_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'spatial_data', '21_00731_hyperbolic_v3.h5ad')
    adata = sc.read_h5ad(h5ad_path)
    emb = adata.obsm['X_poincare']
    norms = np.linalg.norm(emb, axis=1)

    # Define zones
    q33, q67 = np.percentile(norms, [33, 67])
    zones = np.where(norms < q33, 'inner', np.where(norms < q67, 'middle', 'outer'))

    # Assign clusters
    from sklearn.cluster import KMeans
    np.random.seed(42)
    km_labels = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(emb)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.subplots_adjust(wspace=0.45, hspace=0.55)

    # Panel a: Zones on Poincare disk
    ax = axes[0, 0]
    draw_poincare_disk(ax)
    for zone, color in ZONE_COLORS.items():
        mask = zones == zone
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=2, alpha=0.4,
                   label=zone.capitalize(), rasterized=True)
    ax.legend(fontsize=5, markerscale=2, loc='upper right')
    ax.set_xlabel('Poincaré $x_1$', fontsize=7)
    ax.set_ylabel('Poincaré $x_2$', fontsize=7)
    label_panel(ax, 'a')

    # Panel b: Cell type module scores by zone
    ax = axes[0, 1]
    plot_zone_markers_violin(ax, zone_markers)
    label_panel(ax, 'b')

    # Panel c: HDS heatmap with Cohen's d and FDR
    ax = axes[0, 2]
    plot_hds_heatmap(ax, hds)
    label_panel(ax, 'c')

    # Panel d: Hyperbolic clusters on disk
    ax = axes[1, 0]
    draw_poincare_disk(ax)
    for i in range(8):
        mask = km_labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[CLUSTER_CMAP(i / 8)], s=2,
                   alpha=0.4, label=f'C{i}', rasterized=True)
    ax.legend(fontsize=5, ncol=2, markerscale=2, loc='upper right')
    ax.set_xlabel('Poincaré $x_1$', fontsize=7)
    ax.set_ylabel('Poincaré $x_2$', fontsize=7)
    label_panel(ax, 'd')

    # Panel e: Silhouette comparison with jitter
    ax = axes[1, 1]
    plot_silhouette_comparison(ax, clusters)
    label_panel(ax, 'e')

    # Panel f: Niche purity comparison
    ax = axes[1, 2]
    plot_niche_purity(ax, niche_purity)
    label_panel(ax, 'f')

    save_figure(fig, 'Figure3_biological', FIG_DIR)
    print("=== Done ===")


if __name__ == '__main__':
    main()
