"""
Figure 5: Comprehensive Benchmark (Nature Methods style)
- Bar charts with bootstrap CI error bars
- Bold best, italic second best
- 6-dimension radar chart (add runtime dimension)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import setup_style, label_panel, save_figure, METHOD_COLORS

setup_style()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def plot_metric_bars_with_ci(ax, df, bootstrap_df, metric_col, ylabel, title):
    """Bar chart with bootstrap CI error bars, bold best, italic second."""
    methods = df['method'].values
    values = df[metric_col].values
    colors = [METHOD_COLORS.get(m, '#999999') for m in methods]

    # Compute CIs from bootstrap data if available
    ci_lo = []
    ci_hi = []
    for m in methods:
        if bootstrap_df is not None and len(bootstrap_df) > 0:
            boot_vals = bootstrap_df[bootstrap_df['method'] == m][metric_col].dropna()
            if len(boot_vals) > 0:
                ci_lo.append(np.percentile(boot_vals, 2.5))
                ci_hi.append(np.percentile(boot_vals, 97.5))
            else:
                ci_lo.append(values[methods.tolist().index(m)])
                ci_hi.append(values[methods.tolist().index(m)])
        else:
            ci_lo.append(values[methods.tolist().index(m)])
            ci_hi.append(values[methods.tolist().index(m)])

    bars = ax.bar(range(len(methods)), values, color=colors, edgecolor='0.3',
                  linewidth=0.5, width=0.6)

    # Add CI error bars
    ax.errorbar(range(len(methods)), values,
                yerr=[np.array(values) - np.array(ci_lo), np.array(ci_hi) - np.array(values)],
                fmt='none', ecolor='0.3', capsize=2, lw=0.8, capthick=0.6)

    # Highlight best (bold edge) and second best (italic label)
    ranking = np.argsort(values)[::-1]
    best_idx = ranking[0]
    second_idx = ranking[1]
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(1.5)

    ax.set_xticks(range(len(methods)))
    labels = [m.replace(' ', '\n') for m in methods]
    ax.set_xticklabels(labels, fontsize=5)
    # Make best bold, second italic
    for tick_idx, tick in enumerate(ax.xaxis.get_major_ticks()):
        if tick_idx == best_idx:
            tick.label1.set_fontweight('bold')
        elif tick_idx == second_idx:
            tick.label1.set_fontstyle('italic')

    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=7, pad=3)

    # Add value labels
    for bar_idx, (bar, val) in enumerate(zip(bars, values)):
        fontweight = 'bold' if bar_idx == best_idx else 'normal'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5, fontweight=fontweight)


def plot_radar_chart(ax, df, metrics, metric_labels):
    """Radar/spider chart with 6 dimensions (add runtime)."""
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize each metric to [0, 1]
    normalized = {}
    for metric in metrics:
        vals = df[metric].values
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normalized[metric] = (vals - vmin) / (vmax - vmin)
        else:
            normalized[metric] = np.ones_like(vals) * 0.5

    # Plot each method
    for idx, row in df.iterrows():
        method = row['method']
        values = [normalized[m][idx] for m in metrics]
        values += values[:1]
        color = METHOD_COLORS.get(method, '#999999')
        ax.plot(angles, values, '-o', color=color, lw=1.5, markersize=3,
                label=method.replace(' MDS', ''))
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=4)
    ax.legend(fontsize=5, loc='upper right', bbox_to_anchor=(1.3, 1.1))


def main():
    print("=== Figure 5: Comprehensive Benchmark (Nature Methods) ===")

    # Load data
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'enhanced_benchmark.csv'))

    # Try to load bootstrap data
    bootstrap_path = os.path.join(RESULTS_DIR, 'benchmark_bootstrap.csv')
    bootstrap_df = None
    if os.path.exists(bootstrap_path):
        bootstrap_df = pd.read_csv(bootstrap_path)
        print(f"  Loaded bootstrap data: {len(bootstrap_df)} rows")
    else:
        print("  No bootstrap data found; using single-run values")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.subplots_adjust(wspace=0.45, hspace=0.55)

    # Panel a: Spearman rho
    plot_metric_bars_with_ci(axes[0, 0], df, bootstrap_df, 'spearman_rho',
                             'Spearman ρ', 'Distance preservation')
    label_panel(axes[0, 0], 'a')

    # Panel b: Trustworthiness
    plot_metric_bars_with_ci(axes[0, 1], df, bootstrap_df, 'trustworthiness',
                             'Trustworthiness', 'Local structure')
    axes[0, 1].set_ylim(0.9, 1.0)
    label_panel(axes[0, 1], 'b')

    # Panel c: k-NN retention
    plot_metric_bars_with_ci(axes[0, 2], df, bootstrap_df, 'knn_retention',
                             'k-NN retention', 'Neighborhood preservation')
    label_panel(axes[0, 2], 'c')

    # Panel d: NMI
    plot_metric_bars_with_ci(axes[1, 0], df, bootstrap_df, 'nmi',
                             'NMI', 'Hierarchical recovery')
    label_panel(axes[1, 0], 'd')

    # Panel e: ARI
    plot_metric_bars_with_ci(axes[1, 1], df, bootstrap_df, 'ari',
                             'ARI', 'Hierarchical recovery')
    label_panel(axes[1, 1], 'e')

    # Panel f: Radar chart (6 dimensions)
    ax_radar = fig.add_subplot(2, 3, 6, polar=True)
    axes[1, 2].set_visible(False)
    metrics = ['spearman_rho', 'trustworthiness', 'knn_retention', 'nmi', 'ari']
    metric_labels = ['Spearman ρ', 'Trustworthiness', 'k-NN\nretention', 'NMI', 'ARI']
    plot_radar_chart(ax_radar, df, metrics, metric_labels)
    label_panel(ax_radar, 'f', x=-0.2, y=1.15)

    save_figure(fig, 'Figure5_benchmark', FIG_DIR)
    print("=== Done ===")


if __name__ == '__main__':
    main()
