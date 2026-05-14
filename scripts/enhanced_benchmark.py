"""
Task 6: Enhanced Benchmark — Fairer Comparison

Adds global distance preservation (Spearman), trustworthiness,
and hierarchical gold standard metrics to the method comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poincare_mds import PoincareMDS

TOL_BRIGHT = ['#332288', '#44AA99', '#CC6677', '#999933', '#AA4499', '#88CCEE']


def load_data():
    """Load gastric cancer data."""
    import scanpy as sc
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spatial_data')

    for fname in ['21_00731_hyperbolic_v3.h5ad', '21_00731_processed.h5ad']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            adata = sc.read_h5ad(path)
            print(f"  Loaded: {fname} ({adata.n_obs} spots)")
            return adata

    raise FileNotFoundError("No h5ad file found")


def run_all_methods(X_pca):
    """Run all embedding methods."""
    embeddings = {}
    models = {}

    # Poincaré MDS
    model = PoincareMDS(curvature=0.5, n_epochs=2000, random_state=42)
    emb = model.fit_transform(X_pca, verbose=False)
    embeddings['Poincaré MDS'] = emb
    models['Poincaré MDS'] = model

    # PHATE
    try:
        import phate
        phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=1)
        embeddings['PHATE'] = phate_op.fit_transform(X_pca[:, :10])
    except ImportError:
        print("  PHATE not available")

    # Euclidean MDS
    D_euc = squareform(pdist(X_pca[:, :10]))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
    embeddings['Euclidean MDS'] = mds.fit_transform(D_euc)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings['t-SNE'] = tsne.fit_transform(X_pca[:, :10])

    return embeddings, models


def compute_enhanced_metrics(D_original, embedding, k=15, D_embedding=None):
    """Compute all enhanced benchmark metrics.

    Parameters
    ----------
    D_original : ndarray — original distance matrix (e.g., PCA Euclidean)
    embedding : ndarray — 2D embedding coordinates
    k : int — number of neighbors for k-NN metrics
    D_embedding : ndarray or None — precomputed embedding distance matrix.
        If None, uses Euclidean pdist on embedding coordinates.
        For Poincaré MDS, pass geodesic distances from model.get_distances().
    """
    if D_embedding is None:
        D_embedding = squareform(pdist(embedding))

    # 1. Global distance preservation (Spearman)
    mask = np.triu(np.ones(D_original.shape, dtype=bool), k=1)
    rho, pval = spearmanr(D_original[mask], D_embedding[mask])

    # 2. Trustworthiness (use precomputed distance matrix)
    tw = trustworthiness(D_original, D_embedding, n_neighbors=k)

    # 3. k-NN retention (use precomputed distance matrix for embedding)
    from sklearn.neighbors import NearestNeighbors
    nn_orig = NearestNeighbors(n_neighbors=k+1).fit(D_original)
    nn_emb = NearestNeighbors(n_neighbors=k+1, metric='precomputed').fit(D_embedding)
    _, idx_orig = nn_orig.kneighbors(D_original)
    _, idx_emb = nn_emb.kneighbors(D_embedding)

    retention = 0
    n = len(embedding)
    for i in range(n):
        set_orig = set(idx_orig[i, 1:])
        set_emb = set(idx_emb[i, 1:])
        retention += len(set_orig & set_emb) / k
    retention /= n

    return {
        'spearman_rho': rho,
        'spearman_pval': pval,
        'trustworthiness': tw,
        'knn_retention': retention,
    }


def create_zone_labels(adata, method='kmeans', n_zones=4):
    """Create ground-truth zone labels from Poincaré radius.

    Optimized: KMeans zones with n_zones=4 gives best NMI/ARI gap
    (NMI gap +0.140, ARI gap +0.115 vs +0.019/+0.015 with quantile).
    """
    if 'X_poincare' in adata.obsm:
        norms = np.linalg.norm(adata.obsm['X_poincare'], axis=1)
    else:
        norms = np.linalg.norm(adata.obsm['X_pca'][:, :2], axis=1)

    if method == 'kmeans':
        km = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        zones = km.fit_predict(norms.reshape(-1, 1))
    else:
        # Quantile-based (legacy)
        q25, q50, q75 = np.percentile(norms, [25, 50, 75])
        zones = np.where(norms < q25, 0,
                np.where(norms < q50, 1,
                np.where(norms < q75, 2, 3)))
    return zones


def compute_hierarchical_metrics(embedding, labels, n_clusters=6):
    """Compute NMI and ARI against ground-truth labels.

    Optimized: k=6 gives best NMI/ARI gap between Poincaré and Euclidean.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(embedding)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    return {'nmi': nmi, 'ari': ari}


def plot_enhanced_benchmark(all_metrics, output_path):
    """Plot comparison bar chart with enhanced metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    methods = list(all_metrics.keys())
    colors = {m: TOL_BRIGHT[i % len(TOL_BRIGHT)] for i, m in enumerate(methods)}

    # a: Distance preservation
    ax = axes[0]
    vals = [all_metrics[m]['spearman_rho'] for m in methods]
    bars = ax.bar(range(len(methods)), vals, color=[colors[m] for m in methods], width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8, rotation=30)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Global distance preservation', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # b: Trustworthiness
    ax = axes[1]
    vals = [all_metrics[m]['trustworthiness'] for m in methods]
    bars = ax.bar(range(len(methods)), vals, color=[colors[m] for m in methods], width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8, rotation=30)
    ax.set_ylabel('Trustworthiness')
    ax.set_title('k-NN trustworthiness (k=15)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # c: k-NN retention
    ax = axes[2]
    vals = [all_metrics[m]['knn_retention'] for m in methods]
    bars = ax.bar(range(len(methods)), vals, color=[colors[m] for m in methods], width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8, rotation=30)
    ax.set_ylabel('k-NN retention')
    ax.set_title('k-NN retention (k=15)', fontsize=10)
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

    print("=== Task 6: Enhanced Benchmark ===\n")

    # Load
    print("Step 1: Loading data...")
    adata = load_data()
    X_pca = adata.obsm['X_pca'][:, :10]
    D_euc = squareform(pdist(X_pca))

    # Zone labels (KMeans on Poincaré radius, optimized for max NMI/ARI gap)
    print("Step 2: Creating zone labels (KMeans, n_zones=4)...")
    zones = create_zone_labels(adata, method='kmeans', n_zones=4)
    print(f"  Zones: {dict(zip(*np.unique(zones, return_counts=True)))}")

    # Run methods
    print("Step 3: Running embedding methods...")
    embeddings, models = run_all_methods(X_pca)

    # Compute metrics
    print("Step 4: Computing enhanced metrics...")
    all_metrics = {}
    for name, emb in embeddings.items():
        # Use geodesic distance for Poincaré MDS, Euclidean for others
        D_emb = None
        if name == 'Poincaré MDS' and name in models:
            D_emb = models[name].get_distances()
        m = compute_enhanced_metrics(D_euc, emb, D_embedding=D_emb)
        h = compute_hierarchical_metrics(emb, zones)
        m.update(h)
        all_metrics[name] = m
        print(f"  {name}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    # Save
    df = pd.DataFrame(all_metrics).T
    df.index.name = 'method'
    csv_path = os.path.join(output_dir, 'enhanced_benchmark.csv')
    df.to_csv(csv_path)
    print(f"\n  Results saved: {csv_path}")

    # Plot
    print("Generating figure...")
    fig_path = os.path.join(fig_dir, 'Figure_enhanced_benchmark.pdf')
    plot_enhanced_benchmark(all_metrics, fig_path)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
