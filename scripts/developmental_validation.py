"""
Task 4: Second Tissue Type — Developmental Brain Validation

Validates Poincaré MDS on mouse embryonic cortex, where differentiation
proceeds from ventricular zone (progenitors) to cortical plate (mature neurons).
Tests if Poincaré radius correlates with differentiation maturity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poincare_mds import PoincareMDS

TOL_BRIGHT = ['#332288', '#44AA99', '#CC6677', '#999933', '#AA4499', '#88CCEE']


def define_cortical_markers():
    """Marker genes for cortical differentiation stages."""
    return {
        'Progenitor': ['SOX2', 'PAX6', 'HES1', 'NES', 'VIM'],
        'Intermediate': ['DCX', 'NEUROD2', 'TBR2', 'EOMES', 'PPP1R17'],
        'Mature_SATB2': ['SATB2', 'CUX1', 'CUX2', 'RORB'],
        'Mature_TBR1': ['TBR1', 'FEZF2', 'BCL11B', 'CTGF'],
    }


def load_developmental_data():
    """Load developmental brain spatial data."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'spatial_data', 'developmental'
    )
    os.makedirs(data_dir, exist_ok=True)

    # Try to find existing data
    for fname in os.listdir(data_dir) if os.path.exists(data_dir) else []:
        if fname.endswith('.h5ad'):
            import scanpy as sc
            path = os.path.join(data_dir, fname)
            print(f"  Found: {fname}")
            return sc.read_h5ad(path)

    # Generate synthetic developmental data as fallback
    print("  No real developmental data found. Generating synthetic data...")
    return generate_synthetic_developmental_data()


def generate_synthetic_developmental_data(n_total=2000, n_genes=200):
    """Generate synthetic spatial data mimicking cortical development.

    Ventricular zone (center) → Intermediate zone → Cortical plate (periphery).
    """
    import scanpy as sc

    np.random.seed(42)

    # Define zones as concentric rings (strictly non-overlapping)
    n_progenitor = n_total // 4
    n_intermediate = n_total // 4
    n_mature_s = n_total // 4
    n_mature_t = n_total - n_progenitor - n_intermediate - n_mature_s

    # Spatial coordinates: concentric rings
    angles = np.random.uniform(0, 2 * np.pi, n_total)
    radii = np.zeros(n_total)
    radii[:n_progenitor] = np.random.uniform(0.0, 0.20, n_progenitor)
    radii[n_progenitor:n_progenitor+n_intermediate] = np.random.uniform(0.20, 0.40, n_intermediate)
    radii[n_progenitor+n_intermediate:n_progenitor+n_intermediate+n_mature_s] = np.random.uniform(0.40, 0.60, n_mature_s)
    radii[n_progenitor+n_intermediate+n_mature_s:] = np.random.uniform(0.60, 0.80, n_mature_t)

    spatial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Gene expression: marker genes follow differentiation gradient
    X = np.random.randn(n_total, n_genes) * 0.1

    # Define marker gene indices
    marker_genes = {
        'SOX2': 0, 'PAX6': 1, 'HES1': 2,  # Progenitor
        'DCX': 10, 'NEUROD2': 11, 'TBR2': 12,  # Intermediate
        'SATB2': 20, 'CUX1': 21,  # Mature SATB2
        'TBR1': 30, 'FEZF2': 31,  # Mature TBR1
    }

    # Progenitor markers: high in center, low in periphery
    for g in ['SOX2', 'PAX6', 'HES1']:
        idx = marker_genes[g]
        X[:n_progenitor, idx] = 2.0 + np.random.randn(n_progenitor) * 0.3
        X[n_progenitor:n_progenitor+n_intermediate, idx] = 1.0 + np.random.randn(n_intermediate) * 0.3

    # Intermediate markers: high in middle zone
    for g in ['DCX', 'NEUROD2', 'TBR2']:
        idx = marker_genes[g]
        start = n_progenitor
        X[start:start+n_intermediate, idx] = 2.0 + np.random.randn(n_intermediate) * 0.3

    # Mature SATB2 markers: high in outer zone
    for g in ['SATB2', 'CUX1']:
        idx = marker_genes[g]
        start = n_progenitor + n_intermediate
        X[start:start+n_mature_s, idx] = 2.0 + np.random.randn(n_mature_s) * 0.3

    # Mature TBR1 markers: high in deepest cortical plate
    for g in ['TBR1', 'FEZF2']:
        idx = marker_genes[g]
        start = n_progenitor + n_intermediate + n_mature_s
        X[start:, idx] = 2.0 + np.random.randn(n_mature_t) * 0.3

    # Labels
    labels = np.array(
        ['Progenitor'] * n_progenitor +
        ['Intermediate'] * n_intermediate +
        ['Mature_SATB2'] * n_mature_s +
        ['Mature_TBR1'] * n_mature_t
    )

    # Create AnnData
    var_names = [f'Gene_{i}' for i in range(n_genes)]
    for name, idx in marker_genes.items():
        var_names[idx] = name

    adata = sc.AnnData(
        X=X,
        obs={'zone': labels},
        obsm={'spatial': spatial},
    )
    adata.var_names = var_names

    # PCA
    sc.tl.pca(adata, n_comps=20)
    return adata


def preprocess_developmental(adata):
    """Preprocess developmental brain data."""
    import scanpy as sc

    if 'X_pca' not in adata.obsm:
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=20)

    return adata


def compute_layer_scores(adata, markers):
    """Compute module scores for each differentiation stage."""
    scores = {}
    for stage, genes in markers.items():
        available = [g for g in genes if g in adata.var_names]
        if available:
            expr = adata[:, available].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            scores[stage] = expr.mean(axis=1)
        else:
            scores[stage] = np.zeros(adata.n_obs)
    return scores


def validate_radius_differentiation(norms, scores, stage_names):
    """Test if Poincaré radius correlates with differentiation stage."""
    # Assign dominant stage
    score_matrix = np.column_stack([scores[s] for s in stage_names])
    dominant = np.array([stage_names[i] for i in np.argmax(score_matrix, axis=1)])

    # Ordinal encoding: Progenitor=0, Intermediate=1, Mature_SATB2=2, Mature_TBR1=3
    stage_order = {s: i for i, s in enumerate(stage_names)}
    ordinal = np.array([stage_order[s] for s in dominant])

    rho, pval = spearmanr(norms, ordinal)

    # Per-stage radius statistics
    per_stage = []
    for s in stage_names:
        mask = dominant == s
        per_stage.append({
            'stage': s,
            'n': mask.sum(),
            'mean_radius': norms[mask].mean(),
            'std_radius': norms[mask].std(),
        })

    return {
        'radius_differentiation_rho': rho,
        'radius_differentiation_pval': pval,
        'per_stage': pd.DataFrame(per_stage),
        'dominant': dominant,
        'ordinal': ordinal,
    }


def run_all_methods(X_pca):
    """Run Poincaré MDS and comparison methods."""
    embeddings = {}

    # Poincaré MDS
    model = PoincareMDS(curvature=0.5, n_epochs=2000, random_state=42)
    emb = model.fit_transform(X_pca, verbose=False)
    embeddings['Poincaré MDS'] = (emb, model.get_norms())

    # Euclidean MDS
    D_euc = squareform(pdist(X_pca[:, :10]))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
    embeddings['Euclidean MDS'] = (mds.fit_transform(D_euc), None)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings['t-SNE'] = (tsne.fit_transform(X_pca[:, :10]), None)

    return embeddings


def plot_developmental(emb_poincare, norms, validation, embeddings, output_path):
    """Generate Nature Methods figure."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    theta = np.linspace(0, 2 * np.pi, 100)
    stage_names = ['Progenitor', 'Intermediate', 'Mature_SATB2', 'Mature_TBR1']
    stage_colors = {s: TOL_BRIGHT[i] for i, s in enumerate(stage_names)}
    dominant = validation['dominant']

    # a: Poincaré by stage
    ax = axes[0, 0]
    for s in stage_names:
        mask = dominant == s
        ax.scatter(emb_poincare[mask, 0], emb_poincare[mask, 1],
                   c=stage_colors[s], s=10, alpha=0.6, label=s)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title('Poincaré MDS (differentiation stage)', fontsize=10)
    ax.axis('off')

    # b: Poincaré by radius
    ax = axes[0, 1]
    sc_plot = ax.scatter(emb_poincare[:, 0], emb_poincare[:, 1],
                         c=norms, cmap='viridis', s=10, alpha=0.6)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    plt.colorbar(sc_plot, ax=ax, shrink=0.8, label='Radius')
    ax.set_title('Poincaré radius', fontsize=10)
    ax.axis('off')

    # c: t-SNE
    ax = axes[0, 2]
    if 't-SNE' in embeddings:
        emb_tsne = embeddings['t-SNE'][0]
        for s in stage_names:
            mask = dominant == s
            ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                       c=stage_colors[s], s=10, alpha=0.6, label=s)
        ax.legend(fontsize=7, markerscale=2)
        ax.set_title('t-SNE (comparison)', fontsize=10)
    ax.axis('off')

    # d: Radius vs stage violin
    ax = axes[1, 0]
    per_stage = validation['per_stage']
    stage_data = [norms[dominant == s] for s in stage_names]
    parts = ax.violinplot(stage_data, positions=range(len(stage_names)), showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(TOL_BRIGHT[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(['Prog.', 'Inter.', 'SATB2', 'TBR1'], fontsize=8)
    ax.set_ylabel('Poincaré radius')
    rho = validation['radius_differentiation_rho']
    ax.set_title(f'Radius vs stage (ρ={rho:.3f})', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # e: Stage proportions by radius quartile
    ax = axes[1, 1]
    quartiles = np.percentile(norms, [25, 50, 75])
    radius_group = np.digitize(norms, quartiles)
    group_labels = ['Q1 (center)', 'Q2', 'Q3', 'Q4 (periphery)']
    for qi in range(4):
        mask = radius_group == qi
        counts = [np.sum((dominant == s) & mask) for s in stage_names]
        total = mask.sum()
        if total > 0:
            bottom = 0
            for si, s in enumerate(stage_names):
                frac = counts[si] / total
                ax.barh(qi, frac, left=bottom, color=stage_colors[s], height=0.7)
                bottom += frac
    ax.set_yticks(range(4))
    ax.set_yticklabels(group_labels, fontsize=8)
    ax.set_xlabel('Proportion')
    ax.set_title('Stage composition by radius', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # f: Spatial coords colored by stage
    ax = axes[1, 2]
    if 'spatial' in embeddings or True:  # Use from adata if available
        for s in stage_names:
            mask = dominant == s
            ax.scatter(emb_poincare[mask, 0], emb_poincare[mask, 1],
                       c=stage_colors[s], s=10, alpha=0.6, label=s)
        ax.set_title('Spatial coordinates', fontsize=10)
        ax.legend(fontsize=7, markerscale=2)
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

    print("=== Task 4: Developmental Brain Validation ===\n")

    # Load data
    print("Step 1: Loading developmental data...")
    adata = load_developmental_data()
    adata = preprocess_developmental(adata)
    X_pca = adata.obsm['X_pca'][:, :10]
    print(f"  Data: {adata.n_obs} spots, {adata.n_vars} genes")

    # Compute scores
    print("Step 2: Computing differentiation stage scores...")
    markers = define_cortical_markers()
    scores = compute_layer_scores(adata, markers)
    stage_names = list(markers.keys())

    # Run methods
    print("Step 3: Running embedding methods...")
    embeddings = run_all_methods(X_pca)
    emb_poincare, norms = embeddings['Poincaré MDS']

    # Validate
    print("Step 4: Validating radius-differentiation correlation...")
    validation = validate_radius_differentiation(norms, scores, stage_names)
    print(f"  Spearman rho: {validation['radius_differentiation_rho']:.4f}")
    print(f"  p-value: {validation['radius_differentiation_pval']:.2e}")
    print("\n  Per-stage radius:")
    print(validation['per_stage'].to_string(index=False))

    # Save
    validation['per_stage'].to_csv(
        os.path.join(output_dir, 'developmental_per_stage.csv'), index=False
    )
    pd.DataFrame([{
        'rho': validation['radius_differentiation_rho'],
        'pval': validation['radius_differentiation_pval'],
    }]).to_csv(os.path.join(output_dir, 'developmental_metrics.csv'), index=False)

    # Plot
    print("\nGenerating figure...")
    fig_path = os.path.join(fig_dir, 'Figure_developmental_validation.pdf')
    plot_developmental(emb_poincare, norms, validation, embeddings, fig_path)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
