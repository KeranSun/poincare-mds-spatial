"""
Ligand-Receptor Co-expression in Hyperbolic vs Physical Niches.

Computes inter-cellular LR interaction potential within neighborhoods
defined by geodesic (hyperbolic) vs Euclidean (spatial) distance.

For each spot i and LR pair (ligand L, receptor R):
  interaction(i, j) = sqrt(expr_L[i] * expr_R[j])

Mean interaction within neighborhood = mean over j in N_k(i) of interaction(i, j)

Compares hyperbolic vs spatial neighborhoods using Mann-Whitney U test.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
import scanpy as sc
import torch
from geoopt import PoincareBall

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'spatial_data', '21_00731_hyperbolic_v3.h5ad')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Cell type marker signatures
SIGNATURES = {
    'Epithelial':  ['EPCAM', 'KRT18', 'KRT19', 'KRT8'],
    'Fibroblast':  ['COL1A1', 'COL1A2', 'DCN', 'LUM'],
    'T_cell':      ['CD3D', 'CD3E', 'CD2'],
    'Macrophage':  ['CD68', 'C1QA', 'C1QB', 'C1QC'],
    'Endothelial': ['VWF', 'CDH5', 'ENG'],
    'CAF':         ['FAP', 'POSTN', 'ACTA2', 'MMP2'],
}

# Known ligand-receptor pairs
LR_PAIRS = {
    'Epithelial_Macrophage': {
        'ligand': ['CD44', 'MIF', 'LGALS9', 'HGF'],
        'receptor': ['LYVE1', 'CD74', 'HAVCR2', 'MET'],
    },
    'CAF_Macrophage': {
        'ligand': ['CXCL12', 'CCL2', 'IL6', 'TGFB1'],
        'receptor': ['CXCR4', 'CCR2', 'IL6R', 'TGFBR1'],
    },
    'Tcell_Macrophage': {
        'ligand': ['CD40LG', 'IFNG', 'TNF', 'FASLG'],
        'receptor': ['CD40', 'IFNGR1', 'TNFRSF1A', 'FAS'],
    },
    'Epithelial_Fibroblast': {
        'ligand': ['WNT5A', 'SHH', 'BMP4', 'FGF2'],
        'receptor': ['FZD5', 'PTCH1', 'BMPR2', 'FGFR1'],
    },
    'Endothelial_Macrophage': {
        'ligand': ['VEGFA', 'ANGPT2', 'CSF1', 'CXCL8'],
        'receptor': ['FLT1', 'TIE2', 'CSF1R', 'CXCR2'],
    },
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


def assign_dominant_celltype(scores, cell_types):
    score_matrix = np.column_stack([scores[ct] for ct in cell_types])
    dominant_idx = np.argmax(score_matrix, axis=1)
    return np.array([cell_types[i] for i in dominant_idx])


def build_neighborhoods(D, k):
    n = D.shape[0]
    neighborhoods = []
    for i in range(n):
        idx = np.argsort(D[i])[1:k+1]
        neighborhoods.append(set(idx))
    return neighborhoods


def compute_lr_interaction_potential(adata, neighborhoods, ligand_genes, receptor_genes):
    """
    For each spot i, compute mean inter-cellular LR interaction with neighbors.
    interaction(i, j) = sqrt(mean_ligand_expr[i] * mean_receptor_expr[j])
    Returns array of per-spot mean interaction potentials.
    """
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

    # Per-spot mean ligand and receptor expression
    mean_l = expr_l.mean(axis=1)  # (n,)
    mean_r = expr_r.mean(axis=1)  # (n,)

    n = adata.n_obs
    potentials = np.zeros(n)
    for i in range(n):
        nbrs = list(neighborhoods[i])
        if not nbrs:
            continue
        # Interaction: spot i's ligand with neighbor j's receptor
        interactions = np.sqrt(mean_l[i] * mean_r[nbrs])
        potentials[i] = np.mean(interactions)

    return potentials


def run_lr_coexpression_analysis():
    print("=== LR Co-expression in Hyperbolic vs Physical Niches ===\n")

    # Load data
    print("Step 1: Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  {adata.n_obs} spots, {adata.n_vars} genes")

    # Compute cell types
    print("\nStep 2: Computing cell type assignments...")
    scores = compute_module_scores(adata, SIGNATURES)
    cell_types = list(SIGNATURES.keys())
    dominant_ct = assign_dominant_celltype(scores, cell_types)

    # Get embeddings
    emb_hyp = adata.obsm['X_poincare']
    spatial = adata.obsm['spatial']

    # Compute geodesic distance
    print("\nStep 3: Computing geodesic distance...")
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

    D_spatial = squareform(pdist(spatial))

    # Build neighborhoods
    k = 30
    print(f"\nStep 4: Building neighborhoods (k={k})...")
    nn_hyp = build_neighborhoods(D_geodesic, k)
    nn_spatial = build_neighborhoods(D_spatial, k)

    # Check available LR pairs
    print("\nStep 5: Computing LR interaction potentials...")
    available_lr = {}
    for pair_name, pair_info in LR_PAIRS.items():
        avail_l = [g for g in pair_info['ligand'] if g in adata.var_names]
        avail_r = [g for g in pair_info['receptor'] if g in adata.var_names]
        if avail_l and avail_r:
            available_lr[pair_name] = {'ligand': avail_l, 'receptor': avail_r}

    results = []

    for pair_name, pair_info in available_lr.items():
        ct_a, ct_b = pair_name.split('_')
        if ct_a not in cell_types or ct_b not in cell_types:
            continue

        # Compute interaction potentials
        pot_hyp = compute_lr_interaction_potential(
            adata, nn_hyp, pair_info['ligand'], pair_info['receptor'])
        pot_spatial = compute_lr_interaction_potential(
            adata, nn_spatial, pair_info['ligand'], pair_info['receptor'])

        # Filter to spots of cell type A (the ligand-expressing cells)
        mask_a = (dominant_ct == ct_a)
        if mask_a.sum() < 10:
            continue

        pot_hyp_a = pot_hyp[mask_a]
        pot_spatial_a = pot_spatial[mask_a]

        # Mann-Whitney U test: hyperbolic vs spatial
        stat, pval = mannwhitneyu(pot_hyp_a, pot_spatial_a, alternative='two-sided')

        # Effect size: mean difference
        mean_hyp = np.mean(pot_hyp_a)
        mean_spatial = np.mean(pot_spatial_a)

        results.append({
            'pair': pair_name,
            'n_spots': int(mask_a.sum()),
            'mean_hyperbolic': round(mean_hyp, 6),
            'mean_spatial': round(mean_spatial, 6),
            'ratio': round(mean_hyp / (mean_spatial + 1e-10), 4),
            'mann_whitney_U': round(stat, 1),
            'p_value': f'{pval:.2e}',
            'higher_in': 'Hyperbolic' if mean_hyp > mean_spatial else 'Spatial',
        })

        print(f"  {pair_name}: hyp={mean_hyp:.6f}, spatial={mean_spatial:.6f}, "
              f"ratio={mean_hyp/(mean_spatial+1e-10):.4f}, p={pval:.2e} "
              f"[{results[-1]['higher_in']}]")

    # Also compute for ALL pairs combined (aggregate)
    print("\n  --- Aggregate (all pairs) ---")
    all_pot_hyp = []
    all_pot_spatial = []
    for pair_name, pair_info in available_lr.items():
        ct_a, ct_b = pair_name.split('_')
        if ct_a not in cell_types or ct_b not in cell_types:
            continue
        pot_hyp = compute_lr_interaction_potential(
            adata, nn_hyp, pair_info['ligand'], pair_info['receptor'])
        pot_spatial = compute_lr_interaction_potential(
            adata, nn_spatial, pair_info['ligand'], pair_info['receptor'])
        mask_a = (dominant_ct == ct_a)
        all_pot_hyp.extend(pot_hyp[mask_a])
        all_pot_spatial.extend(pot_spatial[mask_a])

    all_pot_hyp = np.array(all_pot_hyp)
    all_pot_spatial = np.array(all_pot_spatial)
    stat, pval = mannwhitneyu(all_pot_hyp, all_pot_spatial, alternative='two-sided')
    print(f"  All pairs: hyp={np.mean(all_pot_hyp):.6f}, spatial={np.mean(all_pot_spatial):.6f}, "
          f"ratio={np.mean(all_pot_hyp)/(np.mean(all_pot_spatial)+1e-10):.4f}, p={pval:.2e}")

    results.append({
        'pair': 'ALL_PAIRS',
        'n_spots': len(all_pot_hyp),
        'mean_hyperbolic': round(np.mean(all_pot_hyp), 6),
        'mean_spatial': round(np.mean(all_pot_spatial), 6),
        'ratio': round(np.mean(all_pot_hyp) / (np.mean(all_pot_spatial) + 1e-10), 4),
        'mann_whitney_U': round(stat, 1),
        'p_value': f'{pval:.2e}',
        'higher_in': 'Hyperbolic' if np.mean(all_pot_hyp) > np.mean(all_pot_spatial) else 'Spatial',
    })

    # Save results
    print("\nStep 6: Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'lr_coexpression_niche.csv'), index=False)
    print(df.to_string(index=False))

    print("\n=== Done ===")
    return df


if __name__ == '__main__':
    run_lr_coexpression_analysis()
