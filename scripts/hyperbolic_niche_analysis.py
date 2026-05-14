"""
Hyperbolic Niche Analysis: Compare physical vs hyperbolic neighborhoods
for capturing biologically meaningful cell-cell relationships.

Key question: Does defining neighborhoods in hyperbolic space reveal
functional interactions that Euclidean/physical neighborhoods miss?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
from poincare_mds import PoincareMDS, HyperbolicNiche

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'spatial_data', '21_00731_hyperbolic_v3.h5ad')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Known ligand-receptor pairs relevant to tumor microenvironment
# These are well-established interactions in gastric cancer
LR_PAIRS = {
    'Epithelial_Macrophage': {
        'ligand': ['CD44', 'MIF', 'LGALS9', 'HGF'],
        'receptor': ['LYVE1', 'CD74', 'HAVCR2', 'MET'],
        'description': 'Tumor-immune signaling',
    },
    'CAF_Macrophage': {
        'ligand': ['CXCL12', 'CCL2', 'IL6', 'TGFB1'],
        'receptor': ['CXCR4', 'CCR2', 'IL6R', 'TGFBR1'],
        'description': 'Stromal-immune crosstalk',
    },
    'Tcell_Macrophage': {
        'ligand': ['CD40LG', 'IFNG', 'TNF', 'FASLG'],
        'receptor': ['CD40', 'IFNGR1', 'TNFRSF1A', 'FAS'],
        'description': 'Immune activation/exhaustion',
    },
    'Epithelial_Fibroblast': {
        'ligand': ['WNT5A', 'SHH', 'BMP4', 'FGF2'],
        'receptor': ['FZD5', 'PTCH1', 'BMPR2', 'FGFR1'],
        'description': 'Epithelial-stromal niche signaling',
    },
    'Endothelial_Macrophage': {
        'ligand': ['VEGFA', 'ANGPT2', 'CSF1', 'CXCL8'],
        'receptor': ['FLT1', 'TIE2', 'CSF1R', 'CXCR2'],
        'description': 'Angiogenesis-immune axis',
    },
}

# Cell type marker signatures (same as figure1_overview.py)
SIGNATURES = {
    'Epithelial':  ['EPCAM', 'KRT18', 'KRT19', 'KRT8'],
    'Fibroblast':  ['COL1A1', 'COL1A2', 'DCN', 'LUM'],
    'T_cell':      ['CD3D', 'CD3E', 'CD2'],
    'Macrophage':  ['CD68', 'C1QA', 'C1QB', 'C1QC'],
    'Endothelial': ['VWF', 'CDH5', 'ENG'],
    'CAF':         ['FAP', 'POSTN', 'ACTA2', 'MMP2'],
}


def compute_module_scores(adata, signatures):
    """Compute z-score normalized module scores."""
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


def assign_dominant_celltype(scores, cell_types):
    """Assign dominant cell type per spot using argmax of z-score module scores."""
    score_matrix = np.column_stack([scores[ct] for ct in cell_types])
    dominant_idx = np.argmax(score_matrix, axis=1)
    return np.array([cell_types[i] for i in dominant_idx])


def compute_coexpression_score(adata, ligand_genes, receptor_genes):
    """Compute co-expression score for ligand-receptor pair."""
    avail_l = [g for g in ligand_genes if g in adata.var_names]
    avail_r = [g for g in receptor_genes if g in adata.var_names]

    if not avail_l or not avail_r:
        return np.zeros(adata.n_obs)

    idx_l = [list(adata.var_names).index(g) for g in avail_l]
    idx_r = [list(adata.var_names).index(g) for g in avail_r]

    expr_l = adata.X[:, idx_l]
    expr_r = adata.X[:, idx_r]
    if hasattr(expr_l, 'toarray'):
        expr_l = expr_l.toarray()
    if hasattr(expr_r, 'toarray'):
        expr_r = expr_r.toarray()

    # Geometric mean of mean ligand and receptor expression
    mean_l = expr_l.mean(axis=1)
    mean_r = expr_r.mean(axis=1)
    return np.sqrt(mean_l * mean_r)


def build_neighborhoods(D, k):
    """Build k-NN neighborhoods from distance matrix."""
    n = D.shape[0]
    neighborhoods = []
    for i in range(n):
        idx = np.argsort(D[i])[1:k+1]  # exclude self
        neighborhoods.append(set(idx))
    return neighborhoods


def compute_pair_enrichment(neighborhoods, labels, ct_a, ct_b):
    """Compute enrichment of cell type pairs in neighborhoods."""
    n = len(neighborhoods)
    observed = 0
    expected_count = 0

    for i in range(n):
        if labels[i] != ct_a:
            continue
        neighbors = list(neighborhoods[i])
        if len(neighbors) == 0:
            continue
        # Count ct_b in neighborhood
        ct_b_count = sum(1 for j in neighbors if labels[j] == ct_b)
        observed += ct_b_count / len(neighbors)

    # Normalize by number of ct_a spots
    n_ct_a = sum(1 for l in labels if l == ct_a)
    if n_ct_a > 0:
        observed /= n_ct_a

    # Expected: proportion of ct_b in whole dataset
    expected = sum(1 for l in labels if l == ct_b) / len(labels)

    return observed, expected


def run_niche_analysis():
    print("=== Hyperbolic Niche Analysis ===\n")

    # Load data
    print("Step 1: Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  {adata.n_obs} spots, {adata.n_vars} genes")

    # Check available genes for LR pairs
    print("\nStep 2: Checking ligand-receptor gene availability...")
    available_lr = {}
    for pair_name, pair_info in LR_PAIRS.items():
        avail_l = [g for g in pair_info['ligand'] if g in adata.var_names]
        avail_r = [g for g in pair_info['receptor'] if g in adata.var_names]
        if avail_l and avail_r:
            available_lr[pair_name] = {
                'ligand': avail_l,
                'receptor': avail_r,
                'description': pair_info['description'],
            }
            print(f"  {pair_name}: {len(avail_l)} ligands, {len(avail_r)} receptors")
        else:
            print(f"  {pair_name}: SKIPPED (no genes found)")

    # Compute module scores and assign cell types
    print("\nStep 3: Computing cell type assignments...")
    scores = compute_module_scores(adata, SIGNATURES)
    cell_types = list(SIGNATURES.keys())
    dominant_ct = assign_dominant_celltype(scores, cell_types)
    for ct in cell_types:
        n = (dominant_ct == ct).sum()
        print(f"  {ct}: {n} spots")

    # Get embeddings
    emb_hyp = adata.obsm['X_poincare']
    spatial = adata.obsm['spatial']
    pca = adata.obsm['X_pca'][:, :10]

    # Compute distance matrices
    print("\nStep 4: Computing distance matrices...")
    D_hyp = squareform(pdist(emb_hyp))  # Euclidean distance in Poincaré embedding space
    D_spatial = squareform(pdist(spatial))  # Physical spatial distance
    D_pca = squareform(pdist(pca))  # PCA Euclidean distance

    # Compute Poincaré geodesic distance directly
    import torch
    from geoopt import PoincareBall
    c = 0.5
    ball = PoincareBall(c=c)
    tensor = torch.tensor(emb_hyp, dtype=torch.float64)
    n = len(emb_hyp)
    D_geodesic = np.zeros((n, n))
    bs = 500
    for i in range(0, n, bs):
        ei = min(i + bs, n)
        for j in range(0, n, bs):
            ej = min(j + bs, n)
            d = ball.dist(tensor[i:ei].unsqueeze(1), tensor[j:ej].unsqueeze(0))
            D_geodesic[i:ei, j:ej] = d.detach().numpy()

    # Build neighborhoods at different scales
    k_values = [10, 20, 30, 50]
    print(f"\nStep 5: Building neighborhoods (k = {k_values})...")

    results = []

    for k in k_values:
        print(f"\n  --- k = {k} ---")

        # Build neighborhoods
        nn_hyp = build_neighborhoods(D_geodesic, k)
        nn_spatial = build_neighborhoods(D_spatial, k)
        nn_pca = build_neighborhoods(D_pca, k)

        # 1. Cell type pair enrichment
        print(f"  Cell type pair enrichment:")
        for pair_name, pair_info in available_lr.items():
            ct_a, ct_b = pair_name.split('_')

            # Skip if cell types not in our signatures
            if ct_a not in cell_types or ct_b not in cell_types:
                continue

            obs_hyp, exp_hyp = compute_pair_enrichment(nn_hyp, dominant_ct, ct_a, ct_b)
            obs_spatial, exp_spatial = compute_pair_enrichment(nn_spatial, dominant_ct, ct_a, ct_b)
            obs_pca, exp_pca = compute_pair_enrichment(nn_pca, dominant_ct, ct_a, ct_b)

            # Enrichment ratio (observed / expected)
            enr_hyp = obs_hyp / (exp_hyp + 1e-10)
            enr_spatial = obs_spatial / (exp_spatial + 1e-10)
            enr_pca = obs_pca / (exp_pca + 1e-10)

            results.append({
                'k': k,
                'pair': pair_name,
                'description': pair_info['description'],
                'enrichment_hyperbolic': round(enr_hyp, 4),
                'enrichment_spatial': round(enr_spatial, 4),
                'enrichment_pca': round(enr_pca, 4),
                'ratio_hyp_vs_spatial': round(enr_hyp / (enr_spatial + 1e-10), 4),
            })

            print(f"    {pair_name}: hyp={enr_hyp:.2f}x, spatial={enr_spatial:.2f}x, "
                  f"pca={enr_pca:.2f}x, ratio={enr_hyp/(enr_spatial+1e-10):.2f}")

    # 2. Niche purity comparison
    print("\nStep 6: Niche purity comparison...")
    purity_results = []
    for k in [10, 20, 30]:
        nn_hyp = build_neighborhoods(D_geodesic, k)
        nn_spatial = build_neighborhoods(D_spatial, k)

        km = KMeans(n_clusters=8, random_state=42, n_init=10)
        km_labels = km.fit_predict(emb_hyp)

        hyp_purities = []
        spatial_purities = []

        step = max(1, len(emb_hyp) // 500)
        for i in range(0, len(emb_hyp), step):
            if len(nn_hyp[i]) >= 3:
                hyp_purities.append(max(np.bincount(km_labels[list(nn_hyp[i])])) / len(nn_hyp[i]))
            if len(nn_spatial[i]) >= 3:
                spatial_purities.append(max(np.bincount(km_labels[list(nn_spatial[i])])) / len(nn_spatial[i]))

        mean_hyp = np.mean(hyp_purities)
        mean_spatial = np.mean(spatial_purities)
        _, pval = mannwhitneyu(hyp_purities, spatial_purities)

        purity_results.append({
            'k': k,
            'hyperbolic_purity': round(mean_hyp, 4),
            'spatial_purity': round(mean_spatial, 4),
            'purity_pval': round(pval, 6),
        })

        print(f"  k={k}: hyp={mean_hyp:.4f}, spatial={mean_spatial:.4f}, p={pval:.2e}")

    # 3. Permutation test for specific cell type pairs
    print("\nStep 7: Permutation test for cell type co-occurrence...")
    n_permutations = 1000
    k = 30

    nn_hyp = build_neighborhoods(D_geodesic, k)
    nn_spatial = build_neighborhoods(D_spatial, k)

    permutation_results = []

    for pair_name in available_lr:
        ct_a, ct_b = pair_name.split('_')
        if ct_a not in cell_types or ct_b not in cell_types:
            continue

        # Observed co-occurrence in hyperbolic neighborhoods
        obs_hyp = 0
        for i in range(len(nn_hyp)):
            if dominant_ct[i] != ct_a:
                continue
            for j in nn_hyp[i]:
                if dominant_ct[j] == ct_b:
                    obs_hyp += 1

        # Observed co-occurrence in spatial neighborhoods
        obs_spatial = 0
        for i in range(len(nn_spatial)):
            if dominant_ct[i] != ct_a:
                continue
            for j in nn_spatial[i]:
                if dominant_ct[j] == ct_b:
                    obs_spatial += 1

        # Permutation test: shuffle labels
        perm_hyp = []
        perm_spatial = []
        for _ in range(n_permutations):
            perm_labels = np.random.permutation(dominant_ct)

            pH = 0
            for i in range(len(nn_hyp)):
                if perm_labels[i] != ct_a:
                    continue
                for j in nn_hyp[i]:
                    if perm_labels[j] == ct_b:
                        pH += 1
            perm_hyp.append(pH)

            pS = 0
            for i in range(len(nn_spatial)):
                if perm_labels[i] != ct_a:
                    continue
                for j in nn_spatial[i]:
                    if perm_labels[j] == ct_b:
                        pS += 1
            perm_spatial.append(pS)

        # Z-scores
        z_hyp = (obs_hyp - np.mean(perm_hyp)) / (np.std(perm_hyp) + 1e-10)
        z_spatial = (obs_spatial - np.mean(perm_spatial)) / (np.std(perm_spatial) + 1e-10)

        # P-values
        p_hyp = np.mean(np.array(perm_hyp) >= obs_hyp)
        p_spatial = np.mean(np.array(perm_spatial) >= obs_spatial)

        permutation_results.append({
            'pair': pair_name,
            'obs_hyperbolic': obs_hyp,
            'obs_spatial': obs_spatial,
            'z_hyperbolic': round(z_hyp, 2),
            'z_spatial': round(z_spatial, 2),
            'p_hyperbolic': round(p_hyp, 6),
            'p_spatial': round(p_spatial, 6),
        })

        print(f"  {pair_name}: hyp_z={z_hyp:.2f} (p={p_hyp:.4f}), "
              f"spatial_z={z_spatial:.2f} (p={p_spatial:.4f})")

    # Save results
    print("\nSaving results...")
    df_enrichment = pd.DataFrame(results)
    df_enrichment.to_csv(os.path.join(RESULTS_DIR, 'niche_enrichment.csv'), index=False)

    df_purity = pd.DataFrame(purity_results)
    df_purity.to_csv(os.path.join(RESULTS_DIR, 'niche_purity.csv'), index=False)

    df_permutation = pd.DataFrame(permutation_results)
    df_permutation.to_csv(os.path.join(RESULTS_DIR, 'niche_permutation.csv'), index=False)

    # Summary
    print("\n=== Summary ===")
    print("\nCell type pair enrichment (k=30, hyperbolic vs spatial):")
    df_k30 = df_enrichment[df_enrichment['k'] == 30]
    for _, row in df_k30.iterrows():
        hyp = row['enrichment_hyperbolic']
        sp = row['enrichment_spatial']
        ratio = row['ratio_hyp_vs_spatial']
        better = "HYPERBOLIC" if hyp > sp else "SPATIAL"
        print(f"  {row['pair']:>25s}: hyp={hyp:.2f}x, spatial={sp:.2f}x, "
              f"ratio={ratio:.2f} [{better}]")

    print("\nNiche purity (KMeans k=8):")
    for _, row in df_purity.iterrows():
        print(f"  k={row['k']}: hyp={row['hyperbolic_purity']:.4f}, "
              f"spatial={row['spatial_purity']:.4f}, p={row['purity_pval']:.2e}")

    print("\nPermutation test (k=30):")
    for _, row in df_permutation.iterrows():
        print(f"  {row['pair']:>25s}: hyp_z={row['z_hyperbolic']:.2f} (p={row['p_hyperbolic']:.4f}), "
              f"spatial_z={row['z_spatial']:.2f} (p={row['p_spatial']:.4f})")

    print("\n=== Done ===")


if __name__ == '__main__':
    run_niche_analysis()
