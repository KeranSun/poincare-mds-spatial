"""
Task 8: Theoretical Analysis — Stress vs Tree Depth Scaling

Validates that Poincaré MDS embedding stress scales as O(log D)
while Euclidean MDS stress scales as O(D) for trees of depth D.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poincare_mds import PoincareMDS


def generate_binary_tree(depth, n_per_leaf=20, noise=0.3, n_features=50):
    """Generate binary tree hierarchical data.

    Returns
    -------
    X : ndarray (n_samples, n_features)
    D_tree : ndarray (n_samples, n_samples) — tree distance matrix
    leaf_codes : list of str — binary path codes
    labels : ndarray — leaf assignment for each sample
    """
    n_leaves = 2 ** depth
    leaf_codes = [format(i, f'0{depth}b') for i in range(n_leaves)]

    X_list, labels_list = [], []
    for leaf_idx, code in enumerate(leaf_codes):
        # Feature vector encodes tree path
        center = np.zeros(n_features)
        for d, bit in enumerate(code):
            center[d * 2 + int(bit)] = 3.0
        X = np.random.randn(n_per_leaf, n_features) * noise + center
        X_list.append(X)
        labels_list.extend([leaf_idx] * n_per_leaf)

    X = np.vstack(X_list)
    labels = np.array(labels_list)

    # Tree distance between leaves
    D_leaves = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        for j in range(n_leaves):
            # Path length through LCA
            common_prefix = 0
            for d in range(depth):
                if leaf_codes[i][d] == leaf_codes[j][d]:
                    common_prefix += 1
                else:
                    break
            D_leaves[i, j] = 2 * (depth - common_prefix)

    # Expand to sample-level distance
    n = len(X)
    D_tree = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D_tree[i, j] = D_leaves[labels[i], labels[j]]

    return X, D_tree, leaf_codes, labels


def compute_stress(D_target, D_embedding):
    """Normalized MDS stress."""
    mask = np.triu(np.ones(D_target.shape, dtype=bool), k=1)
    d_tgt = D_target[mask]
    d_emb = D_embedding[mask]
    return np.sqrt(np.sum((d_tgt - d_emb) ** 2) / np.sum(d_tgt ** 2))


def log_model(d, a, b):
    return a * np.log(d) + b


def linear_model(d, a, b):
    return a * d + b


def run_depth_scaling(depths=(2, 3, 4, 5, 6, 7, 8), n_per_leaf=20, n_repeats=3):
    """Run stress scaling experiment across tree depths."""
    results = []

    for depth in depths:
        for rep in range(n_repeats):
            np.random.seed(42 + rep)
            n_leaves = 2 ** depth
            n_total = n_leaves * n_per_leaf
            print(f"  Depth {depth}, repeat {rep+1}/{n_repeats} "
                  f"({n_leaves} leaves, {n_total} points)...")

            X, D_tree, _, _ = generate_binary_tree(depth, n_per_leaf=n_per_leaf)

            # Normalize D_tree to [0, 1] for fair comparison
            D_tree_norm = D_tree / D_tree.max()

            # Both methods use D_tree as target
            # Poincaré MDS with precomputed D_tree
            model = PoincareMDS(curvature=1.0, n_epochs=2000, random_state=42 + rep)
            emb_hyp = model.fit_transform(X, verbose=False, D_target=D_tree_norm)
            D_hyp = model.get_distances()
            # Stress against normalized D_tree (same scale as embedding distances)
            stress_hyp = compute_stress(D_tree_norm, D_hyp)

            # Euclidean MDS with same D_tree
            mds = MDS(n_components=2, dissimilarity='precomputed',
                      random_state=42 + rep, max_iter=300)
            emb_mds = mds.fit_transform(D_tree_norm)
            D_mds = squareform(pdist(emb_mds))
            stress_mds = compute_stress(D_tree_norm, D_mds)

            results.append({
                'depth': depth,
                'n_leaves': n_leaves,
                'n_points': n_total,
                'repeat': rep,
                'stress_poincare': stress_hyp,
                'stress_euclidean_mds': stress_mds,
            })

    return pd.DataFrame(results)


def fit_scaling_laws(df):
    """Fit log and linear scaling laws to stress data."""
    # Mean stress per depth
    summary = df.groupby('depth').agg({
        'stress_poincare': ['mean', 'std'],
        'stress_euclidean_mds': ['mean', 'std'],
    }).reset_index()
    summary.columns = ['depth', 'hyp_mean', 'hyp_std', 'euc_mean', 'euc_std']

    depths = summary['depth'].values

    # Fit log model to Poincaré
    popt_log, _ = curve_fit(log_model, depths, summary['hyp_mean'])
    hyp_pred_log = log_model(depths, *popt_log)
    ss_res_log = np.sum((summary['hyp_mean'] - hyp_pred_log) ** 2)
    ss_tot = np.sum((summary['hyp_mean'] - summary['hyp_mean'].mean()) ** 2)
    r2_log = 1 - ss_res_log / ss_tot

    # Fit linear model to Euclidean MDS
    popt_lin, _ = curve_fit(linear_model, depths, summary['euc_mean'])
    euc_pred_lin = linear_model(depths, *popt_lin)
    ss_res_lin = np.sum((summary['euc_mean'] - euc_pred_lin) ** 2)
    ss_tot_euc = np.sum((summary['euc_mean'] - summary['euc_mean'].mean()) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot_euc

    return summary, popt_log, r2_log, popt_lin, r2_lin


def plot_scaling(summary, popt_log, r2_log, popt_lin, r2_lin, output_path):
    """Plot stress vs tree depth with fitted curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    depths = summary['depth'].values
    d_fine = np.linspace(depths.min(), depths.max(), 100)

    # Panel a: Poincaré stress
    ax = axes[0]
    ax.errorbar(depths, summary['hyp_mean'], yerr=summary['hyp_std'],
                fmt='o-', color='#332288', capsize=3, label='Poincaré MDS')
    ax.plot(d_fine, log_model(d_fine, *popt_log), '--', color='#332288', alpha=0.7,
            label=f'log fit (R²={r2_log:.3f})')
    ax.set_xlabel('Tree depth', fontsize=10)
    ax.set_ylabel('MDS stress', fontsize=10)
    ax.set_title('Poincaré MDS stress scaling', fontsize=11)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel b: Euclidean MDS stress
    ax = axes[1]
    ax.errorbar(depths, summary['euc_mean'], yerr=summary['euc_std'],
                fmt='s-', color='#CC6677', capsize=3, label='Euclidean MDS')
    ax.plot(d_fine, linear_model(d_fine, *popt_lin), '--', color='#CC6677', alpha=0.7,
            label=f'linear fit (R²={r2_lin:.3f})')
    ax.set_xlabel('Tree depth', fontsize=10)
    ax.set_ylabel('MDS stress', fontsize=10)
    ax.set_title('Euclidean MDS stress scaling', fontsize=11)
    ax.legend(fontsize=8)
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

    print("=== Theoretical Analysis: Stress vs Tree Depth ===")
    print("Generating trees of depth 2-8 and computing embedding stress...")
    print("Hypothesis: Poincaré stress ~ O(log D), Euclidean stress ~ O(D)\n")

    df = run_depth_scaling(depths=(2, 3, 4, 5, 6, 7, 8), n_per_leaf=20, n_repeats=3)

    # Save raw results
    csv_path = os.path.join(output_dir, 'theoretical_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Raw results saved: {csv_path}")

    # Fit scaling laws
    summary, popt_log, r2_log, popt_lin, r2_lin = fit_scaling_laws(df)

    print(f"\n  === Scaling Law Results ===")
    print(f"  Poincare MDS: stress = {popt_log[0]:.4f} * log(depth) + {popt_log[1]:.4f}")
    print(f"    R2 = {r2_log:.4f}")
    print(f"  Euclidean MDS: stress = {popt_lin[0]:.4f} * depth + {popt_lin[1]:.4f}")
    print(f"    R2 = {r2_lin:.4f}")

    # Plot
    fig_path = os.path.join(fig_dir, 'Figure_theoretical_analysis.pdf')
    plot_scaling(summary, popt_log, r2_log, popt_lin, r2_lin, fig_path)

    # Save summary
    summary_path = os.path.join(output_dir, 'theoretical_analysis_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
