"""
Scalability test on REAL combined Visium data.
Combines all 10 gastric cancer samples (~32K spots total).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scanpy as sc
import time
import tracemalloc
from poincare_mds import PoincareMDS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spatial_data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def load_all_visium_samples():
    """Load and combine all 10 Visium samples."""
    samples = []
    for d in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, d)
        if not os.path.isdir(full) or not d.startswith('2'):
            continue
        h5_path = os.path.join(full, 'filtered_feature_bc_matrix.h5')
        if not os.path.exists(h5_path):
            continue
        try:
            adata = sc.read_10x_h5(h5_path)
            adata.var_names_make_unique()
            # Add sample ID
            adata.obs['sample'] = d
            # Basic QC
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            samples.append(adata)
            print(f"  Loaded {d}: {adata.n_obs} spots")
        except Exception as e:
            print(f"  Skip {d}: {str(e)[:60]}")

    # Combine
    adata = sc.concat(samples, join='inner')
    print(f"\n  Combined: {adata.n_obs} spots, {adata.n_vars} genes")
    return adata


def preprocess(adata):
    """Standard preprocessing."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)
    return adata


def run_scalability_test():
    print("=== Real Data Scalability Test ===\n")

    # Load all samples
    print("Step 1: Loading all Visium samples...")
    adata = load_all_visium_samples()

    # Preprocess
    print("\nStep 2: Preprocessing...")
    adata = preprocess(adata)
    print(f"  After preprocessing: {adata.n_obs} spots, {adata.n_vars} genes")

    # Subsample to different sizes
    sizes = [1000, 2000, 5000, 10000, 20000]
    sizes = [s for s in sizes if s <= adata.n_obs]

    results = []
    for n in sizes:
        print(f"\nStep 3: Testing n={n}...")
        np.random.seed(42)
        if n < adata.n_obs:
            idx = np.random.choice(adata.n_obs, n, replace=False)
            X = adata.obsm['X_pca'][idx]
        else:
            X = adata.obsm['X_pca']

        tracemalloc.start()
        t0 = time.perf_counter()

        model = PoincareMDS(curvature=0.5, n_epochs=500, random_state=42)
        emb = model.fit_transform(X, verbose=False)

        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_mem / 1024 / 1024

        norms = np.linalg.norm(emb, axis=1)
        print(f"  n={n}: {elapsed:.1f}s, {peak_mb:.0f}MB, "
              f"mean_r={norms.mean():.3f}, max_r={norms.max():.3f}")

        results.append({
            'n_samples': n,
            'runtime_s': round(elapsed, 2),
            'peak_memory_mb': round(peak_mb, 1),
            'mean_radius': round(norms.mean(), 4),
            'max_radius': round(norms.max(), 4),
            'data_type': 'real_visium',
        })

    # Save results
    import pandas as pd
    df_new = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, 'scalability_real_data.csv')
    df_new.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")

    # Print summary
    print("\n=== Scalability Summary ===")
    for r in results:
        print(f"  {r['n_samples']:>6d} spots: {r['runtime_s']:>8.1f}s, "
              f"{r['peak_memory_mb']:>8.0f}MB")


if __name__ == '__main__':
    run_scalability_test()
