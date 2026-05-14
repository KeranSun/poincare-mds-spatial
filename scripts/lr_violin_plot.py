"""
Generate violin plots for LR interaction potential distributions
in hyperbolic vs spatial neighborhoods.
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import style
from figure_style import setup_style, label_panel, save_figure

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'spatial_data', '21_00731_hyperbolic_v3.h5ad')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

SIGNATURES = {
    'Epithelial':  ['EPCAM', 'KRT18', 'KRT19', 'KRT8'],
    'Fibroblast':  ['COL1A1', 'COL1A2', 'DCN', 'LUM'],
    'T_cell':      ['CD3D', 'CD3E', 'CD2'],
    'Macrophage':  ['CD68', 'C1QA', 'C1QB', 'C1QC'],
    'Endothelial': ['VWF', 'CDH5', 'ENG'],
    'CAF':         ['FAP', 'POSTN', 'ACTA2', 'MMP2'],
}

# Hypothesis-driven pairs: known diffusible-signal-mediated interactions
LR_DIFFUSIBLE = {
    'CAF-Macrophage\n(CXCL12-CXCR4)': {
        'ct_a': 'CAF', 'ct_b': 'Macrophage',
        'ligand': ['CXCL12', 'CCL2', 'IL6', 'TGFB1'],
        'receptor': ['CXCR4', 'CCR2', 'IL6R', 'TGFBR1'],
    },
    'Endothelial-Macrophage\n(VEGFA-FLT1)': {
        'ct_a': 'Endothelial', 'ct_b': 'Macrophage',
        'ligand': ['VEGFA', 'ANGPT2', 'CSF1', 'CXCL8'],
        'receptor': ['FLT1', 'TIE2', 'CSF1R', 'CXCR2'],
    },
}

# Contact-dependent pair (negative control)
LR_CONTACT = {
    'Epithelial-Macrophage\n(CD44-LYVE1)': {
        'ct_a': 'Epithelial', 'ct_b': 'Macrophage',
        'ligand': ['CD44', 'MIF', 'LGALS9', 'HGF'],
        'receptor': ['LYVE1', 'CD74', 'HAVCR2', 'MET'],
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


def compute_lr_potential_per_spot(adata, neighborhoods, ligand_genes, receptor_genes):
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
    mean_l = expr_l.mean(axis=1)
    mean_r = expr_r.mean(axis=1)
    n = adata.n_obs
    potentials = np.zeros(n)
    for i in range(n):
        nbrs = list(neighborhoods[i])
        if not nbrs:
            continue
        interactions = np.sqrt(mean_l[i] * mean_r[nbrs])
        potentials[i] = np.mean(interactions)
    return potentials


def main():
    setup_style()
    print("=== LR Interaction Potential Violin Plot ===\n")

    # Load data
    adata = sc.read_h5ad(DATA_PATH)
    scores = compute_module_scores(adata, SIGNATURES)
    cell_types = list(SIGNATURES.keys())
    dominant_ct = assign_dominant_celltype(scores, cell_types)

    emb_hyp = adata.obsm['X_poincare']
    spatial = adata.obsm['spatial']

    # Compute distances
    print("Computing geodesic distance...")
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

    k = 30
    nn_hyp = build_neighborhoods(D_geodesic, k)
    nn_spatial = build_neighborhoods(D_spatial, k)

    # Combine all pairs
    all_pairs = {}
    all_pairs.update(LR_DIFFUSIBLE)
    all_pairs.update(LR_CONTACT)

    # Create figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(wspace=0.4, bottom=0.2, top=0.85)

    for idx, (pair_name, pair_info) in enumerate(all_pairs.items()):
        ax = axes[idx]
        ct_a = pair_info['ct_a']
        mask = (dominant_ct == ct_a)
        if mask.sum() < 10:
            continue

        pot_hyp = compute_lr_potential_per_spot(
            adata, nn_hyp, pair_info['ligand'], pair_info['receptor'])
        pot_spatial = compute_lr_potential_per_spot(
            adata, nn_spatial, pair_info['ligand'], pair_info['receptor'])

        data_hyp = pot_hyp[mask]
        data_spatial = pot_spatial[mask]

        # Mann-Whitney
        stat, pval = mannwhitneyu(data_hyp, data_spatial, alternative='two-sided')

        # Violin plot
        parts = ax.violinplot([data_spatial, data_hyp], positions=[0, 1],
                              showmeans=True, showmedians=False, showextrema=False)
        for pc, color in zip(parts['bodies'], ['#CC6677', '#332288']):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('black')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Spatial', 'Hyperbolic'], fontsize=7)
        ax.set_ylabel('LR interaction potential', fontsize=7)
        ax.set_title(pair_name, fontsize=7, fontweight='bold')

        # Significance annotation
        if pval < 0.001:
            sig = f'p < 10$^{{{int(np.floor(np.log10(pval)))}}}$'
        elif pval < 0.01:
            sig = f'p = {pval:.3f}'
        else:
            sig = f'p = {pval:.3f}'
        ax.text(0.5, 0.95, sig, transform=ax.transAxes, fontsize=7,
                ha='center', va='top')

        # Fold change
        fc = np.mean(data_hyp) / (np.mean(data_spatial) + 1e-10)
        direction = 'higher' if fc > 1 else 'lower'
        ax.text(0.5, 0.85, f'{fc:.2f}x ({direction})', transform=ax.transAxes,
                fontsize=6, ha='center', va='top', color='#555555')

        ax.tick_params(labelsize=7)

    label_panel(axes[0], 'a')
    label_panel(axes[1], 'b')
    label_panel(axes[2], 'c')

    save_figure(fig, 'Figure3e_lr_violin', FIG_DIR)
    print(f"Saved to {FIG_DIR}/Figure3e_lr_violin.pdf")

    # Print summary
    print("\n=== Summary ===")
    for pair_name, pair_info in all_pairs.items():
        ct_a = pair_info['ct_a']
        mask = (dominant_ct == ct_a)
        pot_hyp = compute_lr_potential_per_spot(
            adata, nn_hyp, pair_info['ligand'], pair_info['receptor'])
        pot_spatial = compute_lr_potential_per_spot(
            adata, nn_spatial, pair_info['ligand'], pair_info['receptor'])
        data_hyp = pot_hyp[mask]
        data_spatial = pot_spatial[mask]
        stat, pval = mannwhitneyu(data_hyp, data_spatial, alternative='two-sided')
        fc = np.mean(data_hyp) / (np.mean(data_spatial) + 1e-10)
        print(f"  {pair_name.replace(chr(10), ' ')}: hyp={np.mean(data_hyp):.6f}, "
              f"spatial={np.mean(data_spatial):.6f}, FC={fc:.4f}, p={pval:.2e}")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
