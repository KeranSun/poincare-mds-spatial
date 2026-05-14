"""
Task 1: Slide-seq V2 Mouse Cerebellum Validation

Validates Poincaré MDS on a non-Visium platform (Slide-seq).
Mouse cerebellum has natural hierarchical layers:
  Granular layer → Purkinje cell layer → Molecular layer → White matter
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import MDS, TSNE
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poincare_mds import PoincareMDS, HyperbolicNiche

# Nature Methods style
TOL_BRIGHT = ['#332288', '#44AA99', '#CC6677', '#999933', '#AA4499', '#88CCEE']


def download_slideseq():
    """Download Slide-seq V2 mouse cerebellum data."""
    try:
        import squidpy as sq
        print("  Downloading Slide-seq V2 via squidpy...")
        adata = sq.datasets.slideseqv2()
        print(f"  Loaded: {adata.n_obs} beads, {adata.n_vars} genes")
        return adata
    except ImportError:
        print("  squidpy not available, trying scanpy + GEO...")

    # Fallback: try cached data
    try:
        import scanpy as sc
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'spatial_data', 'slideseq'
        )
        os.makedirs(cache_dir, exist_ok=True)
        h5_path = os.path.join(cache_dir, 'slideseq_v2_cerebellum.h5ad')

        if os.path.exists(h5_path):
            return sc.read_h5ad(h5_path)
    except Exception:
        pass

    # Final fallback: generate synthetic cerebellar data
    print("  No real Slide-seq data available. Generating synthetic cerebellar data...")
    return generate_synthetic_cerebellum()


def generate_synthetic_cerebellum(n_total=3000, n_genes=200):
    """Generate synthetic spatial data mimicking cerebellar layers.

    Layers: Granular (inner) → Purkinje → Molecular → White matter (outer)
    Uses higher noise and weaker signals to create realistic overlap between layers.
    """
    import scanpy as sc

    np.random.seed(42)

    n_granule = n_total // 4
    n_purkinje = n_total // 4
    n_molecular = n_total // 4
    n_wm = n_total - n_granule - n_purkinje - n_molecular

    # Spatial coordinates: concentric rings with some overlap
    angles = np.random.uniform(0, 2 * np.pi, n_total)
    radii = np.zeros(n_total)
    radii[:n_granule] = np.random.uniform(0.0, 0.30, n_granule)
    radii[n_granule:n_granule+n_purkinje] = np.random.uniform(0.20, 0.50, n_purkinje)
    radii[n_granule+n_purkinje:n_granule+n_purkinje+n_molecular] = np.random.uniform(0.40, 0.70, n_molecular)
    radii[n_granule+n_purkinje+n_molecular:] = np.random.uniform(0.60, 0.95, n_wm)

    spatial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Gene expression with higher background noise
    X = np.random.randn(n_total, n_genes) * 0.8

    marker_genes = {
        'SLC17A7': 0, 'NEUROD2': 1, 'GABRA6': 2,  # Granule
        'PCP2': 10, 'GRID2': 11, 'CALB1': 12,       # Purkinje
        'PVALB': 20, 'GAD1': 21, 'GAD2': 22,         # Molecular
        'MBP': 30, 'MOG': 31, 'PLP1': 32,             # White matter
    }

    # Weaker marker signals with gradient overlap
    for g in ['SLC17A7', 'NEUROD2', 'GABRA6']:
        idx = marker_genes[g]
        X[:n_granule, idx] += 1.2 + np.random.randn(n_granule) * 0.5
        X[n_granule:n_granule+n_purkinje, idx] += 0.4  # gradient bleed

    for g in ['PCP2', 'GRID2', 'CALB1']:
        idx = marker_genes[g]
        s = n_granule
        X[s:s+n_purkinje, idx] += 1.2 + np.random.randn(n_purkinje) * 0.5
        X[:n_granule, idx] += 0.3  # gradient bleed

    for g in ['PVALB', 'GAD1', 'GAD2']:
        idx = marker_genes[g]
        s = n_granule + n_purkinje
        X[s:s+n_molecular, idx] += 1.2 + np.random.randn(n_molecular) * 0.5
        X[s+n_molecular:, idx] += 0.3

    for g in ['MBP', 'MOG', 'PLP1']:
        idx = marker_genes[g]
        s = n_granule + n_purkinje + n_molecular
        X[s:, idx] += 1.2 + np.random.randn(n_wm) * 0.5
        X[s-n_purkinje:s, idx] += 0.3

    labels = (['Granule'] * n_granule + ['Purkinje'] * n_purkinje +
              ['Molecular'] * n_molecular + ['WhiteMatter'] * n_wm)

    var_names = [f'Gene_{i}' for i in range(n_genes)]
    for name, idx in marker_genes.items():
        var_names[idx] = name

    adata = sc.AnnData(
        X=X,
        obs={'layer': labels},
        obsm={'spatial': spatial},
    )
    adata.var_names = var_names
    sc.tl.pca(adata, n_comps=20)
    return adata


def preprocess_slideseq(adata, n_hvg=3000, n_pcs=30):
    """Preprocess Slide-seq data (adapted for higher dropout rate)."""
    import scanpy as sc

    if 'X_pca' in adata.obsm:
        return adata

    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3')
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)
    return adata


def define_cerebellar_layers():
    """Marker genes for cerebellar layers."""
    return {
        'Granule': ['SLC17A7', 'NEUROD2', 'GABRA6', 'PAX6'],
        'Purkinje': ['PCP2', 'GRID2', 'CALB1', 'CAR8'],
        'Molecular': ['PVALB', 'GAD1', 'GAD2', 'SLC6A1'],
        'WhiteMatter': ['MBP', 'MOG', 'PLP1', 'CNP'],
    }


def compute_layer_scores(adata, layer_markers):
    """Compute module scores for each cerebellar layer."""
    scores = {}
    for layer, genes in layer_markers.items():
        available = [g for g in genes if g in adata.var_names]
        if available:
            expr = adata[:, available].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            scores[layer] = expr.mean(axis=1)
        else:
            print(f"    Warning: no genes found for {layer}")
            scores[layer] = np.zeros(adata.n_obs)
    return scores


def run_poincare_mds(X_pca, curvature=0.5, n_epochs=2000):
    """Run Poincaré MDS embedding."""
    model = PoincareMDS(
        curvature=curvature,
        n_epochs=n_epochs,
        k_neighbors=30,
        target_radius=0.4,
        repulsion_weight=0.5,
        random_state=42,
    )
    embedding = model.fit_transform(X_pca, verbose=False)
    return embedding, model


def run_benchmarks(X_pca):
    """Run comparison methods."""
    embeddings = {}

    # Euclidean MDS
    print("    Running Euclidean MDS...")
    D_euc = squareform(pdist(X_pca[:, :10]))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
    embeddings['Euclidean MDS'] = mds.fit_transform(D_euc)

    # t-SNE
    print("    Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings['t-SNE'] = tsne.fit_transform(X_pca[:, :10])

    # PHATE
    try:
        import phate
        print("    Running PHATE...")
        phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=1)
        embeddings['PHATE'] = phate_op.fit_transform(X_pca[:, :10])
    except ImportError:
        print("    PHATE not available, skipping")

    return embeddings


def evaluate_hierarchy(embedding, norms, scores, layer_names):
    """Evaluate if radius correlates with cerebellar layer structure."""
    results = {}

    # Assign dominant layer per bead
    score_matrix = np.column_stack([scores[ln] for ln in layer_names])
    dominant_layer = np.array([layer_names[i] for i in np.argmax(score_matrix, axis=1)])

    # Radius-layer correlation (ordinal: granular=0, purkinje=1, molecular=2, WM=3)
    layer_order = {name: i for i, name in enumerate(layer_names)}
    ordinal = np.array([layer_order.get(l, -1) for l in dominant_layer])
    valid = ordinal >= 0
    if valid.sum() > 10:
        rho, pval = spearmanr(norms[valid], ordinal[valid])
        results['radius_layer_rho'] = rho
        results['radius_layer_pval'] = pval

    # Layer purity in niches
    niche = HyperbolicNiche(curvature=0.5, percentile=10)
    D_hyp = niche.compute_distances(embedding)
    D_euc = squareform(pdist(embedding))  # Use embedding Euclidean for fair comparison

    # KMeans labels as ground truth
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    km_labels = km.fit_predict(embedding)

    hyp_radius = np.percentile(D_hyp[D_hyp > 0], 10)
    euc_radius = np.percentile(D_euc[D_euc > 0], 10)

    hyp_purities, euc_purities = [], []
    step = max(1, len(embedding) // 200)
    for i in range(0, len(embedding), step):
        hyp_niche = np.where(D_hyp[i] < hyp_radius)[0]
        euc_niche = np.where(D_euc[i] < euc_radius)[0]
        if len(hyp_niche) >= 3:
            hyp_purities.append(max(np.bincount(km_labels[hyp_niche])) / len(hyp_niche))
        if len(euc_niche) >= 3:
            euc_purities.append(max(np.bincount(km_labels[euc_niche])) / len(euc_niche))

    results['hyp_purity_mean'] = np.mean(hyp_purities)
    results['euc_purity_mean'] = np.mean(euc_purities)
    if len(hyp_purities) > 5 and len(euc_purities) > 5:
        _, results['purity_pval'] = mannwhitneyu(hyp_purities, euc_purities)

    return results


def compute_all_metrics(D_original, embedding, norms, scores, layer_names, D_embedding=None):
    """Compute standardized metrics."""
    if D_embedding is None:
        D_embedding = squareform(pdist(embedding))

    # Distance preservation
    mask = np.triu(np.ones(D_original.shape, dtype=bool), k=1)
    rho, pval = spearmanr(D_original[mask], D_embedding[mask])

    # k-NN retention
    from sklearn.neighbors import NearestNeighbors
    k = 15
    nn_orig = NearestNeighbors(n_neighbors=k+1).fit(D_original)
    nn_emb = NearestNeighbors(n_neighbors=k+1, metric='precomputed').fit(D_embedding)
    _, idx_orig = nn_orig.kneighbors(D_original)
    _, idx_emb = nn_emb.kneighbors(D_embedding)

    retention = 0
    for i in range(len(embedding)):
        set_orig = set(idx_orig[i, 1:])
        set_emb = set(idx_emb[i, 1:])
        retention += len(set_orig & set_emb) / k
    retention /= len(embedding)

    # NMI/ARI vs layers
    score_matrix = np.column_stack([scores[ln] for ln in layer_names])
    dominant = np.argmax(score_matrix, axis=1)
    km = KMeans(n_clusters=len(layer_names), random_state=42, n_init=10)
    pred = km.fit_predict(embedding)
    nmi = normalized_mutual_info_score(dominant, pred)
    ari = adjusted_rand_score(dominant, pred)

    return {
        'spearman_rho': rho,
        'spearman_pval': pval,
        'knn_retention': retention,
        'nmi_layers': nmi,
        'ari_layers': ari,
    }


def plot_slideseq(embedding, norms, scores, layer_names, benchmarks, output_path):
    """Generate Nature Methods style figure."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    theta = np.linspace(0, 2 * np.pi, 100)

    # a: Poincaré disk colored by dominant layer
    ax = axes[0, 0]
    score_matrix = np.column_stack([scores[ln] for ln in layer_names])
    dominant = np.array([layer_names[i] for i in np.argmax(score_matrix, axis=1)])
    colors = {ln: TOL_BRIGHT[i] for i, ln in enumerate(layer_names)}
    for ln in layer_names:
        mask = dominant == ln
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[ln], s=5, alpha=0.5, label=ln)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title('Poincaré MDS (cerebellar layers)', fontsize=10)
    ax.axis('off')

    # b: Poincaré disk colored by radius
    ax = axes[0, 1]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=norms, cmap='viridis', s=5, alpha=0.5)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Radius')
    ax.set_title('Poincaré radius', fontsize=10)
    ax.axis('off')

    # c: t-SNE comparison
    ax = axes[0, 2]
    if 't-SNE' in benchmarks:
        emb_tsne = benchmarks['t-SNE']
        for ln in layer_names:
            mask = dominant == ln
            ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                       c=colors[ln], s=5, alpha=0.5, label=ln)
        ax.set_title('t-SNE (comparison)', fontsize=10)
        ax.legend(fontsize=7, markerscale=3)
    ax.axis('off')

    # d: Radius vs layer (violin)
    ax = axes[1, 0]
    layer_data = [norms[dominant == ln] for ln in layer_names]
    parts = ax.violinplot(layer_data, positions=range(len(layer_names)), showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(TOL_BRIGHT[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, fontsize=8, rotation=20)
    ax.set_ylabel('Poincaré radius')
    ax.set_title('Radius by layer', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # e: Niche purity comparison
    ax = axes[1, 1]
    # Placeholder — will be filled by evaluate_hierarchy
    ax.text(0.5, 0.5, 'Niche purity\n(see results CSV)',
            transform=ax.transAxes, ha='center', va='center', fontsize=10)
    ax.set_title('Niche purity', fontsize=10)

    # f: PHATE or Euclidean MDS
    ax = axes[1, 2]
    if 'PHATE' in benchmarks:
        emb_phate = benchmarks['PHATE']
        for ln in layer_names:
            mask = dominant == ln
            ax.scatter(emb_phate[mask, 0], emb_phate[mask, 1],
                       c=colors[ln], s=5, alpha=0.5, label=ln)
        ax.set_title('PHATE (comparison)', fontsize=10)
        ax.legend(fontsize=7, markerscale=3)
    elif 'Euclidean MDS' in benchmarks:
        emb_mds = benchmarks['Euclidean MDS']
        for ln in layer_names:
            mask = dominant == ln
            ax.scatter(emb_mds[mask, 0], emb_mds[mask, 1],
                       c=colors[ln], s=5, alpha=0.5, label=ln)
        ax.set_title('Euclidean MDS (comparison)', fontsize=10)
        ax.legend(fontsize=7, markerscale=3)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=== Task 1: Slide-seq V2 Mouse Cerebellum Validation ===\n")

    # 1. Download and preprocess
    print("Step 1: Loading Slide-seq data...")
    adata = download_slideseq()
    print(f"  Raw: {adata.n_obs} beads, {adata.n_vars} genes")

    print("Step 2: Preprocessing...")
    adata = preprocess_slideseq(adata)
    print(f"  After preprocessing: {adata.n_obs} beads, {adata.n_vars} genes")
    X_pca = adata.obsm['X_pca'][:, :10]

    # 2. Define layers
    print("Step 3: Computing layer scores...")
    layer_markers = define_cerebellar_layers()
    scores = compute_layer_scores(adata, layer_markers)
    layer_names = list(layer_markers.keys())
    for ln in layer_names:
        print(f"    {ln}: mean={scores[ln].mean():.3f}, max={scores[ln].max():.3f}")

    # 3. Poincaré MDS
    print("Step 4: Running Poincaré MDS...")
    embedding, model = run_poincare_mds(X_pca)
    norms = model.get_norms()
    print(f"  Embedding: {embedding.shape}, radius: [{norms.min():.3f}, {norms.max():.3f}]")

    # 4. Benchmarks
    print("Step 5: Running benchmarks...")
    benchmarks = run_benchmarks(X_pca)

    # 5. Evaluate
    print("Step 6: Evaluating hierarchy recovery...")
    D_euc = squareform(pdist(X_pca[:, :10]))
    # Use geodesic distance for Poincaré MDS evaluation
    D_geodesic = model.get_distances()
    metrics_poincare = compute_all_metrics(D_euc, embedding, norms, scores, layer_names,
                                           D_embedding=D_geodesic)
    print(f"  Poincaré MDS metrics:")
    for k, v in metrics_poincare.items():
        print(f"    {k}: {v:.4f}")

    # Evaluate benchmarks too
    all_metrics_results = {'Poincaré MDS': metrics_poincare}
    for name, emb in benchmarks.items():
        norms_b = np.linalg.norm(emb, axis=1)
        m = compute_all_metrics(D_euc, emb, norms_b, scores, layer_names)
        all_metrics_results[name] = m
        print(f"  {name} metrics:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    # Niche analysis
    print("\nStep 7: Hyperbolic niche analysis...")
    hierarchy_results = evaluate_hierarchy(embedding, norms, scores, layer_names)
    print(f"  Radius-layer correlation: rho={hierarchy_results.get('radius_layer_rho', 'N/A')}")
    print(f"  Hyperbolic purity: {hierarchy_results.get('hyp_purity_mean', 'N/A'):.3f}")
    print(f"  Euclidean purity: {hierarchy_results.get('euc_purity_mean', 'N/A'):.3f}")

    # 6. Save results
    print("\nSaving results...")
    metrics_df = pd.DataFrame(all_metrics_results).T
    metrics_df.index.name = 'method'
    csv_path = os.path.join(output_dir, 'slideseq_metrics.csv')
    metrics_df.to_csv(csv_path)
    print(f"  Metrics: {csv_path}")

    hierarchy_df = pd.DataFrame([hierarchy_results])
    hierarchy_df.to_csv(os.path.join(output_dir, 'slideseq_hierarchy.csv'), index=False)

    # 7. Plot
    print("Generating figure...")
    fig_path = os.path.join(fig_dir, 'Figure_slideseq_validation.pdf')
    plot_slideseq(embedding, norms, scores, layer_names, benchmarks, fig_path)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
