"""Bootstrap benchmark: 50 repeats with resampling for CI estimation.

Runs enhanced_benchmark 50 times with bootstrap resampling of spots,
computes 95% CIs for all metrics per method. Outputs CSV for Figure 5
error bars.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import scanpy as sc

from poincare_mds.core import PoincareMDS


def load_data():
    """Load gastric cancer data."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spatial_data')
    for fname in ['21_00731_hyperbolic_v3.h5ad', '21_00731_processed.h5ad']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            adata = sc.read_h5ad(path)
            print(f"  Loaded: {fname} ({adata.n_obs} spots)")
            return adata
    raise FileNotFoundError("No h5ad file found")


def compute_metrics(D_original, embedding, k=15, D_embedding=None):
    """Compute all benchmark metrics for one embedding."""
    if D_embedding is None:
        D_embedding = squareform(pdist(embedding))

    # Global distance preservation
    mask = np.triu(np.ones(D_original.shape, dtype=bool), k=1)
    rho, pval = spearmanr(D_original[mask], D_embedding[mask])

    # Trustworthiness
    tw = trustworthiness(D_original, D_embedding, n_neighbors=k)

    # k-NN retention
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
        'trustworthiness': tw,
        'knn_retention': retention,
    }


def run_single_repeat(adata, rep, subsample_frac=0.8):
    """Run one bootstrap repeat: subsample, embed, compute metrics."""
    rng = np.random.RandomState(rep)
    n = adata.n_obs
    n_sub = int(n * subsample_frac)
    idx = rng.choice(n, size=n_sub, replace=False)

    X_pca = adata.obsm['X_pca'][idx, :10]
    D_euc = squareform(pdist(X_pca))

    # Zone labels from full data
    norms_full = np.linalg.norm(adata.obsm['X_poincare'], axis=1)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    zones_full = km.fit_predict(norms_full.reshape(-1, 1))
    zones = zones_full[idx]

    results = {}

    # Poincaré MDS
    try:
        model = PoincareMDS(curvature=0.5, n_epochs=2000, random_state=rep)
        emb = model.fit_transform(X_pca, verbose=False)
        D_emb = model.get_distances()
        m = compute_metrics(D_euc, emb, D_embedding=D_emb)
        pred = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(emb)
        m['nmi'] = normalized_mutual_info_score(zones, pred)
        m['ari'] = adjusted_rand_score(zones, pred)
        results['Poincaré MDS'] = m
    except Exception as e:
        print(f"    Poincaré MDS failed: {e}")

    # Euclidean MDS
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=rep, max_iter=300)
        emb = mds.fit_transform(D_euc)
        m = compute_metrics(D_euc, emb)
        pred = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(emb)
        m['nmi'] = normalized_mutual_info_score(zones, pred)
        m['ari'] = adjusted_rand_score(zones, pred)
        results['Euclidean MDS'] = m
    except Exception as e:
        print(f"    Euclidean MDS failed: {e}")

    # t-SNE
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=rep)
        emb = tsne.fit_transform(X_pca)
        m = compute_metrics(D_euc, emb)
        pred = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(emb)
        m['nmi'] = normalized_mutual_info_score(zones, pred)
        m['ari'] = adjusted_rand_score(zones, pred)
        results['t-SNE'] = m
    except Exception as e:
        print(f"    t-SNE failed: {e}")

    return results


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("=== Bootstrap Benchmark (50 repeats) ===\n")
    adata = load_data()

    all_results = []
    n_repeats = 50

    for rep in range(n_repeats):
        t0 = time.time()
        print(f"  Repeat {rep+1:2d}/{n_repeats}...", end=' ')
        results = run_single_repeat(adata, rep=42+rep)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

        for method, metrics in results.items():
            row = {'repeat': rep, 'method': method}
            row.update(metrics)
            all_results.append(row)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'benchmark_bootstrap.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary statistics
    print("\n=== Summary (mean ± std) ===")
    for method in df['method'].unique():
        sub = df[df['method'] == method]
        print(f"\n  {method}:")
        for col in ['spearman_rho', 'trustworthiness', 'knn_retention', 'nmi', 'ari']:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    print(f"    {col}: {vals.mean():.4f} ± {vals.std():.4f} "
                          f"[{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")


if __name__ == '__main__':
    main()
