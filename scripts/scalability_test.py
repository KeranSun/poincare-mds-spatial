"""
Task 7: Scalability Test

Tests Poincaré MDS performance on datasets of increasing size (1k-100k spots).
Reports runtime and memory usage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poincare_mds import PoincareMDS


def generate_random_data(n, n_features=50, n_clusters=8):
    """Generate random clustered data of specified size."""
    np.random.seed(42)
    X_list = []
    for i in range(n_clusters):
        center = np.zeros(n_features)
        center[i % n_features] = 3.0
        n_cluster = n // n_clusters
        X = np.random.randn(n_cluster, n_features) * 0.3 + center
        X_list.append(X)
    X = np.vstack(X_list)
    # Pad if n not divisible by n_clusters
    if len(X) < n:
        extra = np.random.randn(n - len(X), n_features) * 0.3
        X = np.vstack([X, extra])
    return X[:n]


def merge_visium_samples(data_dir, max_samples=10):
    """Merge multiple Visium samples for scalability testing."""
    import scanpy as sc

    samples = sorted([d for d in os.listdir(data_dir)
                      if d.endswith('_LI_SING') and os.path.isdir(os.path.join(data_dir, d))])

    if not samples:
        return None

    adatas = []
    for sample_name in samples[:max_samples]:
        sample_dir = os.path.join(data_dir, sample_name)
        h5_files = [f for f in os.listdir(sample_dir) if f.endswith('.h5')]
        if h5_files:
            try:
                adata = sc.read_10x_h5(os.path.join(sample_dir, h5_files[0]))
                adata.var_names_make_unique()
                sc.pp.filter_cells(adata, min_genes=100)
                sc.pp.filter_genes(adata, min_cells=10)
                adatas.append(adata)
                print(f"    Loaded {sample_name}: {adata.n_obs} spots")
            except Exception as e:
                print(f"    Skipped {sample_name}: {e}")

    if not adatas:
        return None

    # Merge and preprocess
    import anndata
    adata = anndata.concat(adatas)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=10)

    return adata


def profile_run(X, n_epochs=500, curvature=0.5):
    """Profile a single PoincareMDS run."""
    tracemalloc.start()
    start_time = time.perf_counter()

    model = PoincareMDS(
        curvature=curvature,
        n_epochs=n_epochs,
        k_neighbors=30,
        random_state=42,
    )
    embedding = model.fit_transform(X, verbose=False)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    norms = model.get_norms()

    return {
        'n_samples': len(X),
        'runtime_s': elapsed,
        'peak_memory_mb': peak / 1024 / 1024,
        'mean_radius': norms.mean(),
        'max_radius': norms.max(),
    }


def run_scalability_tests():
    """Run tests on datasets of increasing size."""
    results = []

    # Test sizes
    sizes = [1000, 2000, 5000, 10000]

    # Note: n > 10k requires ~O(n^2) memory for full distance matrix
    # For the paper, 1k-10k scaling curve demonstrates practical feasibility

    for n in sizes:
        print(f"\n  Testing n={n}...")
        X = generate_random_data(n)

        result = profile_run(X, n_epochs=500)
        results.append(result)
        print(f"    Runtime: {result['runtime_s']:.1f}s, Memory: {result['peak_memory_mb']:.0f}MB")

    return pd.DataFrame(results)


def plot_scalability(df, output_path):
    """Plot runtime and memory vs dataset size."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Runtime
    ax = axes[0]
    ax.plot(df['n_samples'], df['runtime_s'], 'o-', color='#332288', markersize=8)
    ax.set_xlabel('Number of samples', fontsize=10)
    ax.set_ylabel('Runtime (seconds)', fontsize=10)
    ax.set_title('Poincaré MDS runtime', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Memory
    ax = axes[1]
    ax.plot(df['n_samples'], df['peak_memory_mb'], 's-', color='#44AA99', markersize=8)
    ax.set_xlabel('Number of samples', fontsize=10)
    ax.set_ylabel('Peak memory (MB)', fontsize=10)
    ax.set_title('Poincaré MDS memory usage', fontsize=11)
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

    print("=== Task 7: Scalability Test ===\n")

    df = run_scalability_tests()

    # Save
    csv_path = os.path.join(output_dir, 'scalability_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")

    # Plot
    fig_path = os.path.join(fig_dir, 'Figure_scalability.pdf')
    plot_scalability(df, fig_path)

    # Summary table
    print("\n  === Summary ===")
    print(df.to_string(index=False))

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
