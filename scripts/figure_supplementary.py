"""
Supplementary Figures (Nature Methods style):
- Supp Fig 1: Scalability (runtime + memory)
- Supp Fig 2: Curvature analysis
- Supp Fig 3: k sensitivity (radius-depth correlation vs k)
- Supp Fig 4: lambda sensitivity (radius distribution)
- Supp Fig 5: Convergence curves
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import setup_style, label_panel, save_figure

setup_style()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def plot_supp_fig1_scalability():
    """Supplementary Figure 1: Scalability."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'scalability_results.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.4)

    # Panel a: Runtime
    ax = axes[0]
    ax.plot(df['n_samples'], df['runtime_s'], 'o-', color='#332288', lw=1.5, markersize=5)
    ax.set_xlabel('Number of samples', fontsize=7)
    ax.set_ylabel('Runtime (seconds)', fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for _, row in df.iterrows():
        ax.annotate(f'{row["runtime_s"]:.1f}s', (row['n_samples'], row['runtime_s']),
                    textcoords="offset points", xytext=(5, 5), fontsize=5)
    label_panel(ax, 'a')

    # Panel b: Memory
    ax = axes[1]
    ax.plot(df['n_samples'], df['peak_memory_mb'], 's-', color='#CC6677', lw=1.5, markersize=5)
    ax.set_xlabel('Number of samples', fontsize=7)
    ax.set_ylabel('Peak memory (MB)', fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for _, row in df.iterrows():
        ax.annotate(f'{row["peak_memory_mb"]:.0f}MB', (row['n_samples'], row['peak_memory_mb']),
                    textcoords="offset points", xytext=(5, 5), fontsize=5)
    label_panel(ax, 'b')

    fig.suptitle('Supplementary Figure 1: Scalability', fontsize=9, y=1.02)
    save_figure(fig, 'SuppFig1_scalability', FIG_DIR)


def plot_supp_fig2_curvature():
    """Supplementary Figure 2: Curvature analysis."""
    df_depth = pd.read_csv(os.path.join(RESULTS_DIR, 'curvature_vs_depth.csv'))
    df_flat = pd.read_csv(os.path.join(RESULTS_DIR, 'curvature_flat_vs_hierarchical.csv'))
    df_real = pd.read_csv(os.path.join(RESULTS_DIR, 'curvature_real_data.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.4)

    # Panel a: Curvature vs depth
    ax = axes[0]
    depths = sorted(df_depth['depth'].unique())
    means = []
    stds = []
    for d in depths:
        vals = df_depth[df_depth['depth'] == d]['learned_curvature'].values
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    ax.errorbar(depths, means, yerr=stds, fmt='o-', color='#332288',
                capsize=2, markersize=4, lw=1)
    ax.set_xlabel('Tree depth', fontsize=7)
    ax.set_ylabel('Learned curvature (c)', fontsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    label_panel(ax, 'a')

    # Panel b: Flat vs hierarchical
    ax = axes[1]
    labels_data = ['Hierarchical\n(depth=4)', 'Flat\nclusters', 'Gastric\ncancer']
    curvatures = [
        df_flat[df_flat['data_type'] == 'hierarchical_d4']['learned_curvature'].values[0],
        df_flat[df_flat['data_type'] == 'flat_clusters']['learned_curvature'].values[0],
        df_real['learned_curvature'].values[0],
    ]
    colors = ['#332288', '#CC6677', '#44AA99']

    bars = ax.bar(labels_data, curvatures, color=colors, edgecolor='0.3', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, curvatures):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.1f}', ha='center', va='bottom', fontsize=6)
    ax.set_ylabel('Learned curvature (c)', fontsize=7)
    ax.set_ylim(0, 2.8)
    label_panel(ax, 'b')

    fig.suptitle('Supplementary Figure 2: Adaptive Curvature Learning', fontsize=9, y=1.02)
    save_figure(fig, 'SuppFig2_curvature', FIG_DIR)


def plot_supp_fig3_k_sensitivity():
    """Supplementary Figure 3: k-NN sensitivity analysis."""
    csv_path = os.path.join(RESULTS_DIR, 'sensitivity_k.csv')
    if not os.path.exists(csv_path):
        print("  Skipping Supp Fig 3: sensitivity_k.csv not found")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.4)

    # Panel a: Radius-depth correlation vs k
    ax = axes[0]
    k_vals = sorted(df['k'].unique())
    means = [df[df['k'] == k]['rho'].mean() for k in k_vals]
    stds = [df[df['k'] == k]['rho'].std() for k in k_vals]

    ax.errorbar(k_vals, means, yerr=stds, fmt='o-', color='#332288',
                capsize=2, markersize=5, lw=1.5)
    ax.set_xlabel('k (k-NN neighbors)', fontsize=7)
    ax.set_ylabel('Spearman ρ (radius-depth)', fontsize=7)
    ax.set_title('Radius-depth correlation', fontsize=7, pad=3)
    label_panel(ax, 'a')

    # Panel b: Runtime vs k
    ax = axes[1]
    runtimes = [df[df['k'] == k]['runtime_s'].mean() for k in k_vals]
    runtime_stds = [df[df['k'] == k]['runtime_s'].std() for k in k_vals]

    ax.errorbar(k_vals, runtimes, yerr=runtime_stds, fmt='s-', color='#CC6677',
                capsize=2, markersize=5, lw=1.5)
    ax.set_xlabel('k (k-NN neighbors)', fontsize=7)
    ax.set_ylabel('Runtime (seconds)', fontsize=7)
    ax.set_title('Computational cost', fontsize=7, pad=3)
    label_panel(ax, 'b')

    fig.suptitle('Supplementary Figure 3: k-NN Parameter Sensitivity', fontsize=9, y=1.02)
    save_figure(fig, 'SuppFig3_k_sensitivity', FIG_DIR)


def plot_supp_fig4_lambda_sensitivity():
    """Supplementary Figure 4: Repulsion weight lambda sensitivity."""
    csv_path = os.path.join(RESULTS_DIR, 'sensitivity_lambda.csv')
    if not os.path.exists(csv_path):
        print("  Skipping Supp Fig 4: sensitivity_lambda.csv not found")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.4)

    # Panel a: Radius distribution stats vs lambda
    ax = axes[0]
    ax.errorbar(df['lambda'], df['mean_radius'], yerr=df['std_radius'],
                fmt='o-', color='#332288', capsize=2, markersize=5, lw=1.5,
                label='Mean ± std')
    ax.fill_between(df['lambda'],
                     df['mean_radius'] - df['std_radius'],
                     df['mean_radius'] + df['std_radius'],
                     alpha=0.2, color='#332288')
    ax.set_xlabel('Repulsion weight λ', fontsize=7)
    ax.set_ylabel('Poincaré radius', fontsize=7)
    ax.set_title('Radius distribution', fontsize=7, pad=3)
    ax.legend(fontsize=5)
    label_panel(ax, 'a')

    # Panel b: Radius-depth correlation vs lambda
    ax = axes[1]
    ax.plot(df['lambda'], df['rho'], 'o-', color='#332288', markersize=5, lw=1.5)
    ax.set_xlabel('Repulsion weight λ', fontsize=7)
    ax.set_ylabel('Spearman ρ (radius-depth)', fontsize=7)
    ax.set_title('Hierarchical recovery', fontsize=7, pad=3)
    label_panel(ax, 'b')

    fig.suptitle('Supplementary Figure 4: Repulsion Weight Sensitivity', fontsize=9, y=1.02)
    save_figure(fig, 'SuppFig4_lambda_sensitivity', FIG_DIR)


def plot_supp_fig5_convergence():
    """Supplementary Figure 5: Convergence curves."""
    csv_path = os.path.join(RESULTS_DIR, 'convergence_curves.csv')
    if not os.path.exists(csv_path):
        print("  Skipping Supp Fig 5: convergence_curves.csv not found")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.0))

    colors = ['#332288', '#44AA99', '#CC6677', '#DDCC77']
    for i, target_epochs in enumerate(sorted(df['target_epochs'].unique())):
        subset = df[df['target_epochs'] == target_epochs]
        ax.plot(subset['epoch'], subset['loss'], '-', color=colors[i % len(colors)],
                lw=1.5, label=f'{target_epochs} epochs', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=7)
    ax.set_ylabel('Loss', fontsize=7)
    ax.set_yscale('log')
    ax.legend(fontsize=5, loc='upper right')
    ax.set_title('Convergence analysis', fontsize=7, pad=3)

    # Mark 2000 epoch line
    ax.axvline(x=2000, color='0.7', ls=':', lw=0.8, zorder=0)
    ax.text(2050, ax.get_ylim()[1] * 0.9, '2000\n(default)', fontsize=5, color='0.5', va='top')

    fig.suptitle('Supplementary Figure 5: Convergence Curves', fontsize=9, y=1.02)
    save_figure(fig, 'SuppFig5_convergence', FIG_DIR)


def main():
    print("=== Supplementary Figures ===")
    plot_supp_fig1_scalability()
    plot_supp_fig2_curvature()
    plot_supp_fig3_k_sensitivity()
    plot_supp_fig4_lambda_sensitivity()
    plot_supp_fig5_convergence()
    print("=== Done ===")


if __name__ == '__main__':
    main()
