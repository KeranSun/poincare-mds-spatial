"""
Real Slide-seq V2 Hippocampus Validation
Uses actual Slide-seq V2 data (41,786 beads, 14 clusters).
Validates Poincaré MDS on non-Visium platform with real spatial transcriptomics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import MDS, TSNE
from poincare_mds import PoincareMDS

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'spatial_data', 'slideseq', 'slideseq_v2_cerebellum.h5ad')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def run_validation():
    print("=== Real Slide-seq V2 Validation ===\n")

    # Load
    print("Step 1: Loading real Slide-seq V2 data...")
    import scanpy as sc
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  {adata.n_obs} beads, {adata.n_vars} genes")
    print(f"  Clusters: {adata.obs['cluster'].nunique()}")
    print(f"  Top: {adata.obs['cluster'].value_counts().head(5).to_dict()}")

    X_pca = adata.obsm['X_pca'][:, :30]
    clusters = adata.obs['cluster'].astype(str).values
    n_clusters = len(set(clusters))

    # Subsample for speed (Poincaré MDS on 40K is slow)
    n_sub = 5000
    np.random.seed(42)
    if adata.n_obs > n_sub:
        idx = np.random.choice(adata.n_obs, n_sub, replace=False)
        X_sub = X_pca[idx]
        clusters_sub = clusters[idx]
    else:
        X_sub = X_pca
        clusters_sub = clusters
        idx = np.arange(adata.n_obs)

    print(f"\nStep 2: Running embeddings on {len(X_sub)} beads...")

    # Poincaré MDS
    print("  Poincaré MDS...")
    model = PoincareMDS(curvature=0.5, n_epochs=500, random_state=42)
    emb_hyp = model.fit_transform(X_sub, verbose=False)
    norms_hyp = np.linalg.norm(emb_hyp, axis=1)

    # Euclidean MDS
    print("  Euclidean MDS...")
    D_euc = squareform(pdist(X_sub[:, :10]))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
    emb_mds = mds.fit_transform(D_euc)

    # t-SNE
    print("  t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_tsne = tsne.fit_transform(X_sub[:, :10])

    # PHATE
    has_phate = False
    try:
        import phate
        print("  PHATE...")
        phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=1, verbose=False)
        emb_phate = phate_op.fit_transform(X_sub[:, :10])
        has_phate = True
    except ImportError:
        print("  PHATE not available")

    # Compute metrics
    print("\nStep 3: Computing metrics...")
    methods = {
        'Poincaré MDS': emb_hyp,
        'Euclidean MDS': emb_mds,
        't-SNE': emb_tsne,
    }
    if has_phate:
        methods['PHATE'] = emb_phate

    D_orig = squareform(pdist(X_sub[:, :10]))
    results = []

    for name, emb in methods.items():
        D_emb = squareform(pdist(emb))
        if name == 'Poincaré MDS':
            D_emb = model.get_distances()

        # Distance preservation
        mask = np.triu(np.ones(D_orig.shape, dtype=bool), k=1)
        rho, pval = spearmanr(D_orig[mask], D_emb[mask])

        # k-NN retention
        from sklearn.neighbors import NearestNeighbors
        k = 15
        nn_orig = NearestNeighbors(n_neighbors=k+1).fit(D_orig)
        nn_emb = NearestNeighbors(n_neighbors=k+1, metric='precomputed').fit(D_emb)
        _, idx_orig = nn_orig.kneighbors(D_orig)
        _, idx_emb = nn_emb.kneighbors(D_emb)
        retention = 0
        for i in range(len(emb)):
            retention += len(set(idx_orig[i, 1:]) & set(idx_emb[i, 1:])) / k
        retention /= len(emb)

        # NMI/ARI (KMeans on embedding vs ground-truth clusters)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true_labels = le.fit_transform(clusters_sub)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = km.fit_predict(emb)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        # Silhouette
        sil = silhouette_score(emb, true_labels)

        results.append({
            'method': name,
            'spearman_rho': round(rho, 4),
            'knn_retention': round(retention, 4),
            'nmi': round(nmi, 4),
            'ari': round(ari, 4),
            'silhouette': round(sil, 4),
        })
        print(f"  {name}: rho={rho:.4f}, kNN={retention:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")

    # Radius-layer correlation for Poincaré
    print("\nStep 4: Radius-hierarchy analysis...")
    # Use cluster labels as hierarchy (hippocampal regions)
    cluster_order = {c: i for i, c in enumerate(sorted(set(clusters_sub)))}
    ordinal = np.array([cluster_order[c] for c in clusters_sub])
    rho_radius, pval_radius = spearmanr(norms_hyp, ordinal)
    print(f"  Radius-cluster ρ = {rho_radius:.4f}, p = {pval_radius:.2e}")

    # Save
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, 'slideseq_real_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Summary
    print("\n=== Slide-seq Real Data Summary ===")
    for r in results:
        print(f"  {r['method']:>15s}: rho={r['spearman_rho']:.4f}, "
              f"NMI={r['nmi']:.4f}, ARI={r['ari']:.4f}")

    return df


if __name__ == '__main__':
    run_validation()
