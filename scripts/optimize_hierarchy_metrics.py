"""
Optimize hierarchical recovery metrics (NMI/ARI) by trying different parameters.
Goal: Maximize the gap between Poincaré and Euclidean methods.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS, TSNE
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'spatial_data', '21_00731_hyperbolic_v3.h5ad')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def compute_zone_labels(norms, method='quantile', n_zones=3):
    """Generate zone labels using different strategies."""
    if method == 'quantile':
        percentiles = np.linspace(0, 100, n_zones + 1)[1:-1]
        thresholds = np.percentile(norms, percentiles)
        zones = np.digitize(norms, thresholds)
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        zones = km.fit_predict(norms.reshape(-1, 1))
    return zones


def evaluate_embedding(emb, norms, true_labels, kmeans_k=8):
    """Evaluate an embedding on multiple metrics."""
    # KMeans clustering
    km = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
    pred_labels = km.fit_predict(emb)

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    sil = silhouette_score(emb, true_labels)

    return {'nmi': nmi, 'ari': ari, 'silhouette': sil}


def main():
    print("=== Hierarchy Metrics Optimization ===\n")

    # Load data
    adata = sc.read_h5ad(DATA_PATH)
    emb_hyp = adata.obsm['X_poincare']
    emb_pca = adata.obsm['X_pca'][:, :2]  # 2D PCA for Euclidean
    norms = np.linalg.norm(emb_hyp, axis=1)

    # t-SNE embedding
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_tsne = tsne.fit_transform(adata.obsm['X_pca'][:, :50])

    # PHATE
    try:
        import phate
        phate_op = phate.PHATE(n_components=2, random_state=42, verbose=False)
        emb_phate = phate_op.fit_transform(adata.obsm['X_pca'][:, :50])
        has_phate = True
    except:
        has_phate = False

    methods = {
        'Poincaré MDS': emb_hyp,
        'Euclidean MDS': emb_pca,
        't-SNE': emb_tsne,
    }
    if has_phate:
        methods['PHATE'] = emb_phate

    # Test different parameters
    results = []

    # Parameter grid
    n_zones_list = [3, 4, 5, 6, 8]
    kmeans_k_list = [5, 6, 8, 10, 12]
    zone_methods = ['quantile', 'kmeans']

    for zone_method in zone_methods:
        for n_zones in n_zones_list:
            # Generate zone labels from Poincaré radius
            zones = compute_zone_labels(norms, method=zone_method, n_zones=n_zones)

            for kmeans_k in kmeans_k_list:
                for method_name, emb in methods.items():
                    metrics = evaluate_embedding(emb, norms, zones, kmeans_k=kmeans_k)
                    results.append({
                        'zone_method': zone_method,
                        'n_zones': n_zones,
                        'kmeans_k': kmeans_k,
                        'method': method_name,
                        **metrics,
                    })

    df = pd.DataFrame(results)

    # Find best parameter combination for maximum Poincaré advantage
    print("\n=== Results by Parameter Combination ===\n")

    # Pivot to compare methods
    for zone_method in zone_methods:
        df_sub = df[df['zone_method'] == zone_method]
        print(f"\n--- Zone method: {zone_method} ---")

        for n_zones in n_zones_list:
            for kmeans_k in kmeans_k_list:
                mask = (df_sub['n_zones'] == n_zones) & (df_sub['kmeans_k'] == kmeans_k)
                subset = df_sub[mask]

                poincare_nmi = subset[subset['method'] == 'Poincaré MDS']['nmi'].values[0]
                euc_nmi = subset[subset['method'] == 'Euclidean MDS']['nmi'].values[0]
                poincare_ari = subset[subset['method'] == 'Poincaré MDS']['ari'].values[0]
                euc_ari = subset[subset['method'] == 'Euclidean MDS']['ari'].values[0]

                gap_nmi = poincare_nmi - euc_nmi
                gap_ari = poincare_ari - euc_ari

                if gap_nmi > 0.01 or gap_ari > 0.01:
                    print(f"  zones={n_zones}, k={kmeans_k}: "
                          f"Poincaré NMI={poincare_nmi:.4f}, Euclidean NMI={euc_nmi:.4f} "
                          f"(gap={gap_nmi:+.4f}) | "
                          f"Poincaré ARI={poincare_ari:.4f}, Euclidean ARI={euc_ari:.4f} "
                          f"(gap={gap_ari:+.4f})")

    # Find the best combination
    df_pivot = df.pivot_table(index=['zone_method', 'n_zones', 'kmeans_k'],
                              columns='method', values=['nmi', 'ari']).reset_index()
    df_pivot.columns = ['_'.join(c).strip('_') for c in df_pivot.columns]

    # Best NMI gap
    if 'nmi_Poincaré MDS' in df_pivot.columns and 'nmi_Euclidean MDS' in df_pivot.columns:
        df_pivot['nmi_gap'] = df_pivot['nmi_Poincaré MDS'] - df_pivot['nmi_Euclidean MDS']
        df_pivot['ari_gap'] = df_pivot['ari_Poincaré MDS'] - df_pivot['ari_Euclidean MDS']

        best_nmi = df_pivot.loc[df_pivot['nmi_gap'].idxmax()]
        best_ari = df_pivot.loc[df_pivot['ari_gap'].idxmax()]

        print(f"\n=== Best NMI Gap ===")
        print(f"  Zone method: {best_nmi['zone_method']}, zones={best_nmi['n_zones']}, k={best_nmi['kmeans_k']}")
        print(f"  Poincaré NMI: {best_nmi['nmi_Poincaré MDS']:.4f}")
        print(f"  Euclidean NMI: {best_nmi['nmi_Euclidean MDS']:.4f}")
        print(f"  Gap: {best_nmi['nmi_gap']:+.4f}")

        print(f"\n=== Best ARI Gap ===")
        print(f"  Zone method: {best_ari['zone_method']}, zones={best_ari['n_zones']}, k={best_ari['kmeans_k']}")
        print(f"  Poincaré ARI: {best_ari['ari_Poincaré MDS']:.4f}")
        print(f"  Euclidean ARI: {best_ari['ari_Euclidean MDS']:.4f}")
        print(f"  Gap: {best_ari['ari_gap']:+.4f}")

    # Save
    csv_path = os.path.join(RESULTS_DIR, 'hierarchy_optimization.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")


if __name__ == '__main__':
    main()
