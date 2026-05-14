"""Sensitivity analysis for Poincaré MDS hyperparameters.

Sweeps k (k-NN neighbors), c (curvature), and lambda (repulsion weight),
recording radius-depth correlation (synthetic tree) and niche purity (real data).
Outputs CSV files for supplementary figures S3-S5.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
from poincare_mds.core import PoincareMDS
from poincare_mds.metrics import radius_label_correlation

# --- Synthetic tree data (for radius-depth correlation) ---
def generate_binary_tree(depth=4, n_per_leaf=20, noise=0.3, n_features=50, seed=42):
    rng = np.random.RandomState(seed)
    n_leaves = 2 ** depth
    n_total = n_leaves * n_per_leaf
    X = np.zeros((n_total, n_features))
    labels = np.zeros(n_total, dtype=int)
    leaf_codes = []
    for i in range(n_leaves):
        code = format(i, f'0{depth}b')
        leaf_codes.append(code)
        start = i * n_per_leaf
        end = start + n_per_leaf
        # Position in feature space determined by tree path
        for d in range(depth):
            X[start:end, d] = float(code[d]) + rng.randn(n_per_leaf) * noise
        X[start:end, depth:] = rng.randn(n_per_leaf, n_features - depth) * noise
        labels[start:end] = i
    # Compute tree distance matrix
    n = n_total
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            li = leaf_codes[labels[i]]
            lj = leaf_codes[labels[j]]
            # Tree distance = 2 * (depth - common prefix length)
            common = 0
            for d in range(depth):
                if li[d] == lj[d]:
                    common += 1
                else:
                    break
            dist = 2 * (depth - common)
            D[i, j] = D[j, i] = dist
    # Assign depth labels (number of 1s in leaf code = rough depth proxy)
    depth_labels = np.array([sum(int(c) for c in leaf_codes[l]) for l in labels])
    return X, D, depth_labels


def run_sensitivity_k():
    """Sweep k-NN parameter k and measure radius-depth correlation."""
    print("=== Sensitivity: k-NN parameter k ===")
    results = []
    k_values = [10, 15, 20, 30, 40, 50]
    X, D_tree, depth_labels = generate_binary_tree(depth=4, n_per_leaf=20)

    for k in k_values:
        for rep in range(3):
            t0 = time.time()
            model = PoincareMDS(curvature=0.5, n_epochs=2000, k_neighbors=k,
                                batch_size=30000, random_state=42 + rep)
            emb = model.fit_transform(X)
            elapsed = time.time() - t0
            norms = np.linalg.norm(emb, axis=1)
            rho, p = radius_label_correlation(norms, depth_labels)
            results.append({'k': k, 'repeat': rep, 'rho': rho, 'p': p,
                           'mean_radius': norms.mean(), 'max_radius': norms.max(),
                           'loss_final': model.loss_history[-1] if model.loss_history else np.nan,
                           'runtime_s': elapsed})
            print(f"  k={k:3d} rep={rep} | rho={rho:.4f} p={p:.2e} | {elapsed:.1f}s")

    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity_k.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


def run_sensitivity_c():
    """Sweep curvature c and measure radius-depth correlation."""
    print("\n=== Sensitivity: curvature c ===")
    results = []
    c_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    X, D_tree, depth_labels = generate_binary_tree(depth=4, n_per_leaf=20)

    for c in c_values:
        for rep in range(3):
            t0 = time.time()
            model = PoincareMDS(curvature=c, n_epochs=2000, k_neighbors=30,
                                batch_size=30000, random_state=42 + rep)
            emb = model.fit_transform(X, D_target=D_tree)
            elapsed = time.time() - t0
            norms = np.linalg.norm(emb, axis=1)
            rho, p = radius_label_correlation(norms, depth_labels)
            results.append({'c': c, 'repeat': rep, 'rho': rho, 'p': p,
                           'mean_radius': norms.mean(), 'max_radius': norms.max(),
                           'loss_final': model.loss_history[-1] if model.loss_history else np.nan,
                           'runtime_s': elapsed})
            print(f"  c={c:.1f} rep={rep} | rho={rho:.4f} p={p:.2e} | {elapsed:.1f}s")

    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity_c.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


def run_sensitivity_lambda():
    """Sweep repulsion weight lambda and observe radius distribution."""
    print("\n=== Sensitivity: repulsion weight lambda ===")
    results = []
    lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
    X, D_tree, depth_labels = generate_binary_tree(depth=4, n_per_leaf=20)

    for lam in lambda_values:
        t0 = time.time()
        model = PoincareMDS(curvature=0.5, n_epochs=2000, k_neighbors=30,
                            repulsion_weight=lam, batch_size=30000, random_state=42)
        emb = model.fit_transform(X, D_target=D_tree)
        elapsed = time.time() - t0
        norms = np.linalg.norm(emb, axis=1)
        rho, p = radius_label_correlation(norms, depth_labels)
        results.append({
            'lambda': lam, 'rho': rho, 'p': p,
            'mean_radius': norms.mean(), 'std_radius': norms.std(),
            'min_radius': norms.min(), 'max_radius': norms.max(),
            'loss_final': model.loss_history[-1] if model.loss_history else np.nan,
            'runtime_s': elapsed
        })
        print(f"  lambda={lam:.1f} | rho={rho:.4f} | mean_r={norms.mean():.3f} max_r={norms.max():.3f}")

    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity_lambda.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


def run_convergence():
    """Record loss curves for convergence supplementary figure."""
    print("\n=== Convergence curves ===")
    results = []
    X, D_tree, depth_labels = generate_binary_tree(depth=4, n_per_leaf=20)

    for epochs in [500, 1000, 2000, 3000]:
        model = PoincareMDS(curvature=0.5, n_epochs=epochs, k_neighbors=30,
                            batch_size=30000, random_state=42)
        emb = model.fit_transform(X, D_target=D_tree)
        for ep, loss_val in enumerate(model.loss_history):
            results.append({'target_epochs': epochs, 'epoch': ep + 1, 'loss': loss_val})

    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'convergence_curves.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_k()
    run_sensitivity_c()
    run_sensitivity_lambda()
    run_convergence()
    print("\nAll sensitivity analyses complete.")
