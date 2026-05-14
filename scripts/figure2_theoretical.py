"""
Figure 2: Theoretical Validation (Nature Methods style)
- Tree schematic
- Poincaré vs Euclidean tree embeddings
- Stress scaling with LOWESS smoothing
- Curvature learning
- Method comparison (hierarchical recovery)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import setup_style, label_panel, draw_poincare_disk, save_figure

setup_style()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def log_model(d, a, b):
    return a * np.log(d) + b


def linear_model(d, a, b):
    return a * d + b


def lowess_smooth(x, y, frac=0.6):
    """Simple LOWESS smoothing."""
    n = len(x)
    y_smooth = np.zeros_like(y)
    for i in range(n):
        dists = np.abs(x - x[i])
        bandwidth = np.sort(dists)[max(1, int(frac * n))]
        weights = np.maximum(0, 1 - (dists / bandwidth) ** 3) ** 3
        w_sum = weights.sum()
        if w_sum > 0:
            y_smooth[i] = np.sum(weights * y) / w_sum
        else:
            y_smooth[i] = y[i]
    return y_smooth


def plot_tree_schematic(ax, depth=4):
    """Draw a schematic binary tree."""
    nodes = []
    for d in range(depth + 1):
        n_nodes = 2 ** d
        for i in range(n_nodes):
            x = (i + 0.5) / n_nodes
            y = 1.0 - d / depth
            nodes.append((d, i, x, y))

    for d in range(depth):
        for i in range(2 ** d):
            parent_idx = sum(2 ** j for j in range(d)) + i
            child1_idx = sum(2 ** j for j in range(d + 1)) + 2 * i
            child2_idx = child1_idx + 1
            px, py = nodes[parent_idx][2], nodes[parent_idx][3]
            c1x, c1y = nodes[child1_idx][2], nodes[child1_idx][3]
            c2x, c2y = nodes[child2_idx][2], nodes[child2_idx][3]
            ax.plot([px, c1x], [py, c1y], '-', color='0.5', lw=0.8)
            ax.plot([px, c2x], [py, c2y], '-', color='0.5', lw=0.8)

    leaf_colors = plt.cm.Set3(np.linspace(0, 1, 2 ** depth))
    for d, i, x, y in nodes:
        if d < depth:
            ax.scatter(x, y, s=30, c='0.5', edgecolors='white', linewidths=0.5, zorder=5)
        else:
            ax.scatter(x, y, s=40, c=[leaf_colors[i]], edgecolors='0.3',
                       linewidths=0.5, zorder=5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Binary tree (depth=4)', fontsize=7, pad=3)


def plot_tree_embedding(ax, emb, labels, title, show_disk=False):
    """Plot a tree embedding."""
    if show_disk:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), '-', color='0.5', lw=0.5)

    unique_labels = sorted(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[i]], s=3, alpha=0.5,
                   rasterized=True)

    ax.set_aspect('equal')
    ax.set_xlabel('Component 1', fontsize=7)
    ax.set_ylabel('Component 2', fontsize=7)
    ax.set_title(title, fontsize=7, pad=3)
    if show_disk:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)


def plot_stress_vs_depth(ax, summary_df, raw_df, fit_params):
    """Plot stress vs tree depth with LOWESS smoothing and CI bands."""
    depths = summary_df['depth'].values
    d_fine = np.linspace(depths.min(), depths.max(), 100)

    # Compute CI bands from raw data
    hyp_ci_lo = []
    hyp_ci_hi = []
    euc_ci_lo = []
    euc_ci_hi = []
    for d in depths:
        subset = raw_df[raw_df['depth'] == d]
        hyp_vals = subset['stress_poincare'].values
        euc_vals = subset['stress_euclidean_mds'].values
        hyp_ci_lo.append(np.percentile(hyp_vals, 2.5))
        hyp_ci_hi.append(np.percentile(hyp_vals, 97.5))
        euc_ci_lo.append(np.percentile(euc_vals, 2.5))
        euc_ci_hi.append(np.percentile(euc_vals, 97.5))

    # Poincaré MDS with LOWESS
    ax.errorbar(depths, summary_df['hyp_mean'],
                yerr=[summary_df['hyp_mean'].values - np.array(hyp_ci_lo),
                      np.array(hyp_ci_hi) - summary_df['hyp_mean'].values],
                fmt='o', color='#332288', capsize=2, markersize=4, lw=0,
                label='Poincaré MDS', elinewidth=0.8, capthick=0.6)

    # LOWESS smoothing
    x_smooth = np.linspace(depths.min(), depths.max(), 100)
    y_smooth = lowess_smooth(depths.astype(float), summary_df['hyp_mean'].values, frac=0.7)
    from scipy.interpolate import interp1d
    f_interp = interp1d(depths, y_smooth, kind='cubic', fill_value='extrapolate')
    ax.plot(x_smooth, f_interp(x_smooth), '-', color='#332288', lw=2, zorder=5)

    # Log fit
    ax.plot(d_fine, log_model(d_fine, *fit_params['log_params']), '--',
            color='#332288', alpha=0.5, lw=0.8,
            label=f'log fit (R²={fit_params["r2_log"]:.3f})')

    # Euclidean MDS with LOWESS
    ax.errorbar(depths, summary_df['euc_mean'],
                yerr=[summary_df['euc_mean'] - np.array(euc_ci_lo),
                      np.array(euc_ci_hi) - summary_df['euc_mean']],
                fmt='s', color='#CC6677', capsize=2, markersize=4, lw=0,
                label='Euclidean MDS', elinewidth=0.8, capthick=0.6)

    y_smooth_euc = lowess_smooth(depths.astype(float), summary_df['euc_mean'].values, frac=0.7)
    f_interp_euc = interp1d(depths, y_smooth_euc, kind='cubic', fill_value='extrapolate')
    ax.plot(x_smooth, f_interp_euc(x_smooth), '-', color='#CC6677', lw=2, zorder=5)

    # Linear fit
    ax.plot(d_fine, linear_model(d_fine, *fit_params['lin_params']), '--',
            color='#CC6677', alpha=0.5, lw=0.8,
            label=f'linear fit (R²={fit_params["r2_lin"]:.3f})')

    # Mark crossover
    ax.axvline(x=4, color='0.7', ls=':', lw=0.8, zorder=0)
    ax.text(4.1, ax.get_ylim()[1] * 0.95, 'crossover\ndepth=4',
            fontsize=5, color='0.5', va='top')

    # Spearman rho annotation
    rho, p = spearmanr(depths, summary_df['hyp_mean'])
    rho_euc, p_euc = spearmanr(depths, summary_df['euc_mean'])
    ax.text(0.05, 0.95,
            f'Poincaré: ρ = {rho:.3f}, p = {p:.2e}\n'
            f'Euclidean: ρ = {rho_euc:.3f}, p = {p_euc:.2e}',
            transform=ax.transAxes, fontsize=5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9))

    ax.set_xlabel('Tree depth', fontsize=7)
    ax.set_ylabel('MDS stress', fontsize=7)
    ax.legend(fontsize=5, loc='upper left')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def plot_curvature_vs_depth(ax, df_curvature):
    """Plot learned curvature vs tree depth with CI."""
    depths = sorted(df_curvature['depth'].unique())
    means = []
    ci_lo = []
    ci_hi = []
    for d in depths:
        vals = df_curvature[df_curvature['depth'] == d]['learned_curvature'].values
        means.append(np.mean(vals))
        ci_lo.append(np.percentile(vals, 2.5))
        ci_hi.append(np.percentile(vals, 97.5))

    ax.errorbar(depths, means,
                yerr=[np.array(means) - np.array(ci_lo), np.array(ci_hi) - np.array(means)],
                fmt='o-', color='#332288', capsize=2, markersize=4, lw=1,
                elinewidth=0.8, capthick=0.6)
    ax.set_xlabel('Tree depth', fontsize=7)
    ax.set_ylabel('Learned curvature (c)', fontsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def plot_flat_vs_hierarchical(ax, df_flat, df_real):
    """Bar chart: curvature for hierarchical vs flat vs real data."""
    labels = ['Hierarchical\n(depth=4)', 'Flat\nclusters', 'Gastric\ncancer']
    curvatures = [
        df_flat[df_flat['data_type'] == 'hierarchical_d4']['learned_curvature'].values[0],
        df_flat[df_flat['data_type'] == 'flat_clusters']['learned_curvature'].values[0],
        df_real['learned_curvature'].values[0],
    ]
    colors = ['#332288', '#CC6677', '#44AA99']

    bars = ax.bar(labels, curvatures, color=colors, edgecolor='0.3', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, curvatures):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.1f}', ha='center', va='bottom', fontsize=6)

    ax.set_ylabel('Learned curvature (c)', fontsize=7)
    ax.set_ylim(0, 2.8)


def plot_method_comparison(ax, hierarchy_df):
    """Parallel boxplots: NMI/ARI for each method."""
    methods = hierarchy_df['method'].unique()
    metrics = ['nmi', 'ari']
    positions = np.arange(len(methods))

    for mi, metric in enumerate(metrics):
        offset = (mi - 0.5) * 0.3
        data = [hierarchy_df[(hierarchy_df['method'] == m)][metric].values for m in methods]
        bp = ax.boxplot(data, positions=positions + offset, widths=0.25, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color='black', lw=0.8),
                        whiskerprops=dict(lw=0.5, color='0.3'),
                        capprops=dict(lw=0.5, color='0.3'))
        colors = ['#332288', '#CC6677']
        for patch in bp['boxes']:
            patch.set_facecolor(colors[mi])
            patch.set_alpha(0.6)
            patch.set_edgecolor('0.3')
            patch.set_linewidth(0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=5)
    ax.set_ylabel('Score', fontsize=7)
    ax.legend([matplotlib.patches.Patch(facecolor='#332288', alpha=0.6),
               matplotlib.patches.Patch(facecolor='#CC6677', alpha=0.6)],
              ['NMI', 'ARI'], fontsize=5, loc='upper right')


def main():
    print("=== Figure 2: Theoretical Validation (Nature Methods) ===")

    # Load data
    summary_df = pd.read_csv(os.path.join(RESULTS_DIR, 'theoretical_analysis_summary.csv'))
    raw_df = pd.read_csv(os.path.join(RESULTS_DIR, 'theoretical_analysis.csv'))
    df_curvature = pd.read_csv(os.path.join(RESULTS_DIR, 'curvature_vs_depth.csv'))
    df_flat = pd.read_csv(os.path.join(RESULTS_DIR, 'curvature_flat_vs_hierarchical.csv'))
    df_real = pd.read_csv(os.path.join(RESULTS_DIR, 'curvature_real_data.csv'))

    # Load hierarchy optimization for method comparison
    hierarchy_df = pd.read_csv(os.path.join(RESULTS_DIR, 'hierarchy_optimization.csv'))
    # Use the optimized configuration (n_zones=4, kmeans_k=6)
    hierarchy_opt = hierarchy_df[(hierarchy_df['n_zones'] == 4) & (hierarchy_df['kmeans_k'] == 6)]

    # Fit scaling models
    depths = summary_df['depth'].values
    popt_log, _ = curve_fit(log_model, depths, summary_df['hyp_mean'])
    hyp_pred = log_model(depths, *popt_log)
    ss_res = np.sum((summary_df['hyp_mean'] - hyp_pred) ** 2)
    ss_tot = np.sum((summary_df['hyp_mean'] - summary_df['hyp_mean'].mean()) ** 2)
    r2_log = 1 - ss_res / ss_tot

    popt_lin, _ = curve_fit(linear_model, depths, summary_df['euc_mean'])
    euc_pred = linear_model(depths, *popt_lin)
    ss_res_lin = np.sum((summary_df['euc_mean'] - euc_pred) ** 2)
    ss_tot_lin = np.sum((summary_df['euc_mean'] - summary_df['euc_mean'].mean()) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot_lin

    fit_params = {
        'log_params': popt_log, 'r2_log': r2_log,
        'lin_params': popt_lin, 'r2_lin': r2_lin,
    }

    print(f"  Poincare: stress = {popt_log[0]:.4f} * log(D) + {popt_log[1]:.4f}, R2 = {r2_log:.4f}")
    print(f"  Euclidean: stress = {popt_lin[0]:.4f} * D + {popt_lin[1]:.4f}, R2 = {r2_lin:.4f}")

    # Generate tree embeddings for panels b and c
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.theoretical_analysis import generate_binary_tree
    from poincare_mds import PoincareMDS
    from sklearn.manifold import MDS
    from scipy.spatial.distance import squareform, pdist

    np.random.seed(42)
    X, D_tree, _, labels = generate_binary_tree(depth=4, n_per_leaf=30, noise=0.3)
    D_tree_norm = D_tree / D_tree.max()

    model = PoincareMDS(curvature=1.0, n_epochs=2000, random_state=42)
    emb_hyp = model.fit_transform(X, verbose=False, D_target=D_tree_norm)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
    emb_mds = mds.fit_transform(D_tree_norm)

    labels_arr = np.array(labels)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.subplots_adjust(wspace=0.45, hspace=0.55)

    # Panel a: Tree schematic
    plot_tree_schematic(axes[0, 0], depth=4)
    label_panel(axes[0, 0], 'a')

    # Panel b: Poincare embedding
    plot_tree_embedding(axes[0, 1], emb_hyp, labels_arr, 'Poincaré MDS', show_disk=True)
    label_panel(axes[0, 1], 'b')

    # Panel c: Euclidean MDS embedding
    plot_tree_embedding(axes[0, 2], emb_mds, labels_arr, 'Euclidean MDS', show_disk=False)
    label_panel(axes[0, 2], 'c')

    # Panel d: Stress vs depth with LOWESS and CI
    plot_stress_vs_depth(axes[1, 0], summary_df, raw_df, fit_params)
    label_panel(axes[1, 0], 'd')

    # Panel e: Curvature vs depth with CI
    plot_curvature_vs_depth(axes[1, 1], df_curvature)
    label_panel(axes[1, 1], 'e')

    # Panel f: Method comparison (NMI/ARI boxplots)
    if len(hierarchy_opt) > 0:
        plot_method_comparison(axes[1, 2], hierarchy_opt)
    else:
        plot_flat_vs_hierarchical(axes[1, 2], df_flat, df_real)
    label_panel(axes[1, 2], 'f')

    save_figure(fig, 'Figure2_theoretical', FIG_DIR)
    print("=== Done ===")


if __name__ == '__main__':
    main()
