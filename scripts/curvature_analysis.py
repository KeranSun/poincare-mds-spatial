"""
Task 5: Adaptive Curvature Learning — Experiments

Tests the hypothesis that deeper hierarchies lead to higher learned curvature.
Runs adaptive curvature on synthetic trees of varying depth and real spatial data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poincare_mds import PoincareMDS


def generate_binary_tree(depth, n_per_leaf=20, noise=0.3, n_features=50):
    """Generate binary tree hierarchical data."""
    n_leaves = 2 ** depth
    leaf_codes = [format(i, f'0{depth}b') for i in range(n_leaves)]

    X_list, labels_list = [], []
    for leaf_idx, code in enumerate(leaf_codes):
        center = np.zeros(n_features)
        for d, bit in enumerate(code):
            center[d * 2 + int(bit)] = 3.0
        X = np.random.randn(n_per_leaf, n_features) * noise + center
        X_list.append(X)
        labels_list.extend([leaf_idx] * n_per_leaf)

    return np.vstack(X_list), np.array(labels_list)


def generate_flat_data(n=400, n_features=50):
    """Generate flat (non-hierarchical) data — random clusters at same depth."""
    np.random.seed(42)
    n_clusters = 8
    X_list, labels_list = [], []
    for i in range(n_clusters):
        center = np.zeros(n_features)
        center[i % n_features] = 3.0
        X = np.random.randn(n // n_clusters, n_features) * 0.3 + center
        X_list.append(X)
        labels_list.extend([i] * (n // n_clusters))
    return np.vstack(X_list), np.array(labels_list)


def run_curvature_vs_depth(depths=(2, 3, 4, 5, 6), n_repeats=3):
    """Test: deeper trees → higher learned curvature."""
    results = []
    for depth in depths:
        for rep in range(n_repeats):
            np.random.seed(42 + rep)
            X, labels = generate_binary_tree(depth, n_per_leaf=20)

            model = PoincareMDS(
                curvature=0.5,
                n_epochs=1500,
                adaptive_curvature=True,
                random_state=42 + rep,
            )
            emb = model.fit_transform(X, verbose=False)
            learned_c = model.learned_curvature
            norms = model.get_norms()

            results.append({
                'depth': depth,
                'n_leaves': 2 ** depth,
                'repeat': rep,
                'learned_curvature': learned_c,
                'mean_radius': norms.mean(),
                'max_radius': norms.max(),
            })
            print(f"  Depth {depth}, rep {rep+1}: c={learned_c:.4f}")

    return pd.DataFrame(results)


def run_flat_vs_hierarchical():
    """Test: flat data learns lower curvature than hierarchical."""
    results = []

    # Hierarchical (depth 4)
    np.random.seed(42)
    X_hier, _ = generate_binary_tree(4, n_per_leaf=20)
    model_hier = PoincareMDS(curvature=0.5, n_epochs=1500, adaptive_curvature=True, random_state=42)
    model_hier.fit_transform(X_hier, verbose=False)
    c_hier = model_hier.learned_curvature

    # Flat
    X_flat, _ = generate_flat_data(n=320, n_features=50)
    model_flat = PoincareMDS(curvature=0.5, n_epochs=1500, adaptive_curvature=True, random_state=42)
    model_flat.fit_transform(X_flat, verbose=False)
    c_flat = model_flat.learned_curvature

    results.append({'data_type': 'hierarchical_d4', 'learned_curvature': c_hier})
    results.append({'data_type': 'flat_clusters', 'learned_curvature': c_flat})

    print(f"\n  Hierarchical (depth 4): c = {c_hier:.4f}")
    print(f"  Flat clusters:          c = {c_flat:.4f}")
    print(f"  Ratio (hier/flat):      {c_hier/c_flat:.2f}x")

    return pd.DataFrame(results)


def run_real_data_curvature():
    """Run adaptive curvature on gastric cancer data (if available)."""
    h5ad_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'spatial_data', '21_00731_processed.h5ad'
    )
    if not os.path.exists(h5ad_path):
        print("  Skipping real data (h5ad not found)")
        return None

    import scanpy as sc
    adata = sc.read_h5ad(h5ad_path)
    X_pca = adata.obsm['X_pca'][:, :10]

    model = PoincareMDS(
        curvature=0.5, n_epochs=2000, adaptive_curvature=True, random_state=42
    )
    emb = model.fit_transform(X_pca, verbose=False)
    c_real = model.learned_curvature
    norms = model.get_norms()

    print(f"\n  Gastric cancer data: c = {c_real:.4f}")
    print(f"  Mean radius: {norms.mean():.4f}")

    return {'data': 'gastric_cancer', 'learned_curvature': c_real, 'mean_radius': norms.mean()}


def plot_curvature_results(df_depth, df_flat, output_path):
    """Plot curvature vs depth and flat vs hierarchical comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel a: Curvature vs tree depth
    ax = axes[0]
    summary = df_depth.groupby('depth').agg(
        {'learned_curvature': ['mean', 'std']}).reset_index()
    summary.columns = ['depth', 'mean_c', 'std_c']
    ax.errorbar(summary['depth'], summary['mean_c'], yerr=summary['std_c'],
                fmt='o-', color='#332288', capsize=4, markersize=8)
    ax.set_xlabel('Tree depth', fontsize=10)
    ax.set_ylabel('Learned curvature', fontsize=10)
    ax.set_title('Adaptive curvature vs hierarchy depth', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel b: Flat vs hierarchical
    ax = axes[1]
    if df_flat is not None:
        types = df_flat['data_type'].values
        values = df_flat['learned_curvature'].values
        colors = ['#44AA99', '#CC6677']
        bars = ax.bar(range(len(types)), values, color=colors, width=0.5)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(['Hierarchical\n(depth 4)', 'Flat\n(clusters)'], fontsize=9)
        ax.set_ylabel('Learned curvature', fontsize=10)
        ax.set_title('Curvature: hierarchical vs flat data', fontsize=11)
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

    print("=== Task 5: Adaptive Curvature Learning ===\n")

    # Experiment 1: Curvature vs tree depth
    print("Experiment 1: Curvature vs tree depth (depths 2-6)...")
    df_depth = run_curvature_vs_depth(depths=(2, 3, 4, 5, 6), n_repeats=3)
    df_depth.to_csv(os.path.join(output_dir, 'curvature_vs_depth.csv'), index=False)

    # Experiment 2: Flat vs hierarchical
    print("\nExperiment 2: Flat vs hierarchical comparison...")
    df_flat = run_flat_vs_hierarchical()
    df_flat.to_csv(os.path.join(output_dir, 'curvature_flat_vs_hierarchical.csv'), index=False)

    # Experiment 3: Real data
    print("\nExperiment 3: Gastric cancer real data...")
    real_result = run_real_data_curvature()
    if real_result:
        pd.DataFrame([real_result]).to_csv(
            os.path.join(output_dir, 'curvature_real_data.csv'), index=False
        )

    # Plot
    fig_path = os.path.join(fig_dir, 'Figure_curvature_learning.pdf')
    plot_curvature_results(df_depth, df_flat, fig_path)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
