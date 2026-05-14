"""
Task 3: Gastric Cancer Biological Discovery

Three sub-analyses on existing Poincaré MDS embedding:
3a: Tumor zone segmentation by radius
3b: Hyperbolic-only clusters
3c: Hierarchical Differential Score (HDS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Nature Methods style
TOL_BRIGHT = ['#332288', '#44AA99', '#CC6677', '#999933', '#AA4499', '#88CCEE', '#DDDDDD']


def load_data():
    """Load gastric cancer data with Poincaré embedding."""
    import scanpy as sc
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spatial_data')

    # Try v3 first, fall back to processed
    for fname in ['21_00731_hyperbolic_v3.h5ad', '21_00731_hyperbolic_full.h5ad', '21_00731_processed.h5ad']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            adata = sc.read_h5ad(path)
            print(f"  Loaded: {fname} ({adata.n_obs} spots)")
            break
    else:
        raise FileNotFoundError("No h5ad file found in spatial_data/")

    return adata


def compute_module_scores(adata):
    """Compute cell type module scores (z-score normalized)."""
    signatures = {
        'Epithelial': ['EPCAM', 'KRT18', 'KRT19', 'KRT8'],
        'Fibroblast': ['COL1A1', 'COL1A2', 'DCN', 'LUM'],
        'T_cell': ['CD3D', 'CD3E', 'CD2'],
        'Macrophage': ['CD68', 'C1QA', 'C1QB', 'C1QC'],
        'Endothelial': ['VWF', 'CDH5', 'ENG'],
        'CAF_m': ['FAP', 'POSTN', 'ACTA2', 'MMP2'],
    }

    scores = {}
    for ct, genes in signatures.items():
        available = [g for g in genes if g in adata.var_names]
        if available:
            expr = adata[:, available].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            raw = expr.mean(axis=1)
            # Z-score normalize to remove bias from different expression scales
            scores[ct] = (raw - raw.mean()) / (raw.std() + 1e-8)
        else:
            scores[ct] = np.zeros(adata.n_obs)

    return scores


# ============================================================
# 3a: Tumor Zone Segmentation
# ============================================================

def define_tumor_zones(norms):
    """Divide spots into inner/middle/outer by Poincaré radius quantiles."""
    q33, q67 = np.percentile(norms, [33, 67])
    zones = np.where(norms < q33, 'inner',
            np.where(norms < q67, 'middle', 'outer'))
    return zones


def zone_marker_analysis(scores, zones):
    """Kruskal-Wallis test for each marker across zones."""
    results = []
    zone_names = ['inner', 'middle', 'outer']
    for ct, score in scores.items():
        groups = [score[zones == z] for z in zone_names if np.sum(zones == z) > 0]
        if len(groups) >= 2:
            stat, pval = kruskal(*groups)
            means = {z: score[zones == z].mean() for z in zone_names if np.sum(zones == z) > 0}
            results.append({
                'cell_type': ct,
                'kruskal_stat': stat,
                'kruskal_pval': pval,
                **{f'mean_{z}': means.get(z, np.nan) for z in zone_names},
            })
    return pd.DataFrame(results)


# ============================================================
# 3b: Hyperbolic-only Clusters
# ============================================================

def find_hyperbolic_clusters(emb_poincare, pca_coords, n_clusters=10):
    """Find clusters that are tight in hyperbolic but loose in Euclidean space."""
    km_hyp = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(emb_poincare)
    km_euc = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pca_coords[:, :2])

    sil_hyp = silhouette_score(emb_poincare, km_hyp.labels_)
    sil_euc = silhouette_score(pca_coords[:, :2], km_euc.labels_)
    ari = adjusted_rand_score(km_hyp.labels_, km_euc.labels_)

    # Per-cluster silhouette
    from sklearn.metrics import silhouette_samples
    sil_samples_hyp = silhouette_samples(emb_poincare, km_hyp.labels_)
    sil_samples_euc = silhouette_samples(pca_coords[:, :2], km_euc.labels_)

    per_cluster = []
    for c in range(n_clusters):
        mask = km_hyp.labels_ == c
        per_cluster.append({
            'cluster': c,
            'size': mask.sum(),
            'sil_hyperbolic': sil_samples_hyp[mask].mean(),
            'sil_euclidean': sil_samples_euc[mask].mean(),
            'sil_diff': sil_samples_hyp[mask].mean() - sil_samples_euc[mask].mean(),
        })

    return {
        'sil_hyperbolic_global': sil_hyp,
        'sil_euclidean_global': sil_euc,
        'ari': ari,
        'per_cluster': pd.DataFrame(per_cluster),
        'labels_hyp': km_hyp.labels_,
        'labels_euc': km_euc.labels_,
    }


# ============================================================
# 3c: Hierarchical Differential Score
# ============================================================

def compute_hds(norms, scores):
    """Compute HDS for all cell type pairs.

    HDS = mean_radius(A) - mean_radius(B)
    Positive HDS = A is more peripheral (specialized).
    """
    # Assign dominant cell type per spot (z-score normalized)
    ct_names = list(scores.keys())
    score_matrix = np.column_stack([scores[ct] for ct in ct_names])
    dominant = np.argmax(score_matrix, axis=1)

    results = []
    for i, ct_a in enumerate(ct_names):
        for j, ct_b in enumerate(ct_names):
            if i >= j:
                continue
            mask_a = dominant == i
            mask_b = dominant == j
            if mask_a.sum() < 5 or mask_b.sum() < 5:
                continue
            stat, pval = mannwhitneyu(norms[mask_a], norms[mask_b])
            hds = norms[mask_a].mean() - norms[mask_b].mean()
            results.append({
                'cell_type_A': ct_a,
                'cell_type_B': ct_b,
                'HDS': hds,
                'mean_radius_A': norms[mask_a].mean(),
                'mean_radius_B': norms[mask_b].mean(),
                'mannwhitney_pval': pval,
                'n_A': mask_a.sum(),
                'n_B': mask_b.sum(),
            })
    return pd.DataFrame(results)


# ============================================================
# Visualization
# ============================================================

def plot_biological_discovery(emb, norms, zones, cluster_results, scores, output_path):
    """6-panel Nature Methods figure."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    zone_colors = {'inner': '#CC6677', 'middle': '#999933', 'outer': '#44AA99'}

    # a: Tumor zones on spatial coords
    ax = axes[0, 0]
    for z, c in zone_colors.items():
        mask = zones == z
        ax.scatter(emb[mask, 0], emb[mask, 1], c=c, s=8, alpha=0.6, label=z)
    ax.set_title('Zones on Poincaré disk', fontsize=10)
    ax.legend(fontsize=7)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    # b: Marker scores by zone
    ax = axes[0, 1]
    ct_list = ['CAF_m', 'Fibroblast', 'T_cell', 'Macrophage', 'Epithelial', 'Endothelial']
    zone_names = ['inner', 'middle', 'outer']
    x = np.arange(len(ct_list))
    width = 0.25
    for iz, z in enumerate(zone_names):
        means = [scores[ct][zones == z].mean() for ct in ct_list]
        ax.bar(x + iz * width, means, width, label=z, color=zone_colors[z])
    ax.set_xticks(x + width)
    ax.set_xticklabels(ct_list, fontsize=8, rotation=30)
    ax.set_ylabel('Mean module score')
    ax.set_title('Cell type by zone', fontsize=10)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # c: Zone proportions
    ax = axes[0, 2]
    zone_counts = [np.sum(zones == z) for z in zone_names]
    ax.pie(zone_counts, labels=zone_names, colors=[zone_colors[z] for z in zone_names],
           autopct='%1.0f%%', textprops={'fontsize': 8})
    ax.set_title('Zone proportions', fontsize=10)

    # d: Hyperbolic clusters on disk
    ax = axes[1, 0]
    labels_hyp = cluster_results['labels_hyp']
    n_clust = len(np.unique(labels_hyp))
    cmap = plt.cm.Set3(np.linspace(0, 1, n_clust))
    for c in range(n_clust):
        mask = labels_hyp == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap[c]], s=8, alpha=0.6)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Hyperbolic clusters', fontsize=10)
    ax.axis('off')

    # e: Silhouette comparison
    ax = axes[1, 1]
    per_clust = cluster_results['per_cluster']
    ax.barh(per_clust['cluster'], per_clust['sil_hyperbolic'],
            color='#332288', alpha=0.7, label='Hyperbolic', height=0.4, align='edge')
    ax.barh(per_clust['cluster'] + 0.4, per_clust['sil_euclidean'],
            color='#CC6677', alpha=0.7, label='Euclidean', height=0.4, align='edge')
    ax.set_xlabel('Silhouette score')
    ax.set_ylabel('Cluster')
    ax.set_title('Cluster quality comparison', fontsize=10)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # f: Radius distribution by zone
    ax = axes[1, 2]
    for z in zone_names:
        ax.hist(norms[zones == z], bins=30, alpha=0.5, color=zone_colors[z], label=z, density=True)
    ax.set_xlabel('Poincaré radius')
    ax.set_ylabel('Density')
    ax.set_title('Radius distribution by zone', fontsize=10)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=== Task 3: Gastric Cancer Biological Discovery ===\n")

    # Load data
    print("Loading data...")
    adata = load_data()

    # Get embedding and norms
    if 'X_poincare' in adata.obsm:
        emb = adata.obsm['X_poincare']
    else:
        print("  No X_poincare found, running PoincareMDS...")
        from poincare_mds import PoincareMDS
        model = PoincareMDS(curvature=0.5, n_epochs=2000, random_state=42)
        X_pca = adata.obsm['X_pca'][:, :10] if 'X_pca' in adata.obsm else adata.X
        emb = model.fit_transform(X_pca, verbose=False)

    norms = np.linalg.norm(emb, axis=1)
    print(f"  Embedding: {emb.shape}, radius range: [{norms.min():.3f}, {norms.max():.3f}]")

    # Compute scores
    print("Computing module scores...")
    scores = compute_module_scores(adata)

    # 3a: Tumor zones
    print("\n--- 3a: Tumor Zone Segmentation ---")
    zones = define_tumor_zones(norms)
    zone_df = zone_marker_analysis(scores, zones)
    print(zone_df.to_string(index=False))
    zone_df.to_csv(os.path.join(output_dir, 'zone_marker_analysis.csv'), index=False)

    # 3b: Hyperbolic clusters
    print("\n--- 3b: Hyperbolic-only Clusters ---")
    pca_coords = adata.obsm['X_pca'] if 'X_pca' in adata.obsm else emb
    cluster_results = find_hyperbolic_clusters(emb, pca_coords, n_clusters=8)
    print(f"  Silhouette (hyperbolic): {cluster_results['sil_hyperbolic_global']:.3f}")
    print(f"  Silhouette (Euclidean):  {cluster_results['sil_euclidean_global']:.3f}")
    print(f"  ARI between clusterings: {cluster_results['ari']:.3f}")
    cluster_results['per_cluster'].to_csv(
        os.path.join(output_dir, 'hyperbolic_clusters.csv'), index=False
    )

    # 3c: Hierarchical Differential Score
    print("\n--- 3c: Hierarchical Differential Score ---")
    hds_df = compute_hds(norms, scores)
    hds_df = hds_df.sort_values('HDS', key=abs, ascending=False)
    print(hds_df.head(10).to_string(index=False))
    hds_df.to_csv(os.path.join(output_dir, 'hierarchical_differential_score.csv'), index=False)

    # Plot
    print("\nGenerating figure...")
    fig_path = os.path.join(fig_dir, 'Figure_biological_discovery.pdf')
    plot_biological_discovery(emb, norms, zones, cluster_results, scores, fig_path)

    # Summary CSV
    summary = {
        'n_spots': adata.n_obs,
        'mean_radius': norms.mean(),
        'max_radius': norms.max(),
        'zone_inner_pct': (zones == 'inner').mean(),
        'zone_middle_pct': (zones == 'middle').mean(),
        'zone_outer_pct': (zones == 'outer').mean(),
        'sil_hyperbolic': cluster_results['sil_hyperbolic_global'],
        'sil_euclidean': cluster_results['sil_euclidean_global'],
        'ari_clusters': cluster_results['ari'],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'biological_discovery_summary.csv'), index=False)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
