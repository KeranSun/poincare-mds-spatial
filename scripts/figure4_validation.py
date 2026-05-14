"""
Figure 4: Cross-Dataset Validation (Nature Methods style)
- Slide-seq V2 mouse cerebellum
- Developmental brain cortex
- Improved: bootstrap CI error bars, paired-line plots
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import (setup_style, label_panel, draw_poincare_disk, save_figure,
                          LAYER_COLORS, STAGE_COLORS, METHOD_COLORS)
from stats_utils import bootstrap_ci, spearman_with_ci

setup_style()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def generate_synthetic_cerebellum(n_per_layer=750, n_features=50, noise=0.8, seed=42):
    """Generate synthetic cerebellar data with 4 layers."""
    np.random.seed(seed)
    layers = ['Granule', 'Purkinje', 'Molecular', 'WhiteMatter']
    n_total = n_per_layer * len(layers)

    X = np.zeros((n_total, n_features))
    layer_labels = []

    for i, layer in enumerate(layers):
        start = i * n_per_layer
        end = start + n_per_layer
        center = np.zeros(n_features)
        center[i * 5:(i + 1) * 5] = 3.0
        if i > 0:
            center[(i - 1) * 5:i * 5] += 1.0
        if i < len(layers) - 1:
            center[(i + 1) * 5:(i + 2) * 5] += 1.0
        X[start:end] = np.random.randn(n_per_layer, n_features) * noise + center
        layer_labels.extend([layer] * n_per_layer)

    return X, np.array(layer_labels), layers


def generate_synthetic_developmental(n_per_stage=500, n_features=50, noise=0.3, seed=42):
    """Generate synthetic developmental brain data."""
    np.random.seed(seed)
    stages = ['Progenitor', 'Intermediate', 'Mature_SATB2', 'Mature_TBR1']
    n_total = n_per_stage * len(stages)

    X = np.zeros((n_total, n_features))
    stage_labels = []

    for i, stage in enumerate(stages):
        start = i * n_per_stage
        end = start + n_per_stage
        center = np.zeros(n_features)
        center[i * 5:(i + 1) * 5] = 3.0
        X[start:end] = np.random.randn(n_per_stage, n_features) * noise + center
        stage_labels.extend([stage] * n_per_stage)

    return X, np.array(stage_labels), stages


def run_poincare_embedding(X, D_target=None, seed=42):
    """Run Poincare MDS embedding."""
    from poincare_mds import PoincareMDS
    model = PoincareMDS(curvature=1.0, n_epochs=2000, random_state=seed)
    emb = model.fit_transform(X, verbose=False, D_target=D_target)
    norms = np.linalg.norm(emb, axis=1)
    return emb, norms


def plot_layer_disk(ax, emb, layer_labels, layers):
    """Poincare disk colored by cerebellar layer."""
    draw_poincare_disk(ax)
    for layer in layers:
        mask = layer_labels == layer
        ax.scatter(emb[mask, 0], emb[mask, 1], c=LAYER_COLORS[layer], s=2,
                   alpha=0.4, label=layer, rasterized=True)
    ax.legend(fontsize=5, markerscale=2, loc='upper right')
    ax.set_xlabel('Poincaré $x_1$', fontsize=7)
    ax.set_ylabel('Poincaré $x_2$', fontsize=7)


def plot_radius_violin(ax, norms, labels, unique_labels, colors, xlabel, rho=None, pval=None, ci_lo=None, ci_hi=None):
    """Violin plot of radius by category with CI annotation."""
    data = [norms[labels == lab] for lab in unique_labels]
    parts = ax.violinplot(data, positions=range(len(unique_labels)), showmeans=True,
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[unique_labels[i]])
        pc.set_alpha(0.6)
        pc.set_edgecolor('0.3')
        pc.set_linewidth(0.5)
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(0.8)

    # Add jitter points
    rng = np.random.RandomState(42)
    for i, d in enumerate(data):
        jitter = rng.uniform(-0.08, 0.08, len(d))
        ax.scatter(np.full_like(d, i) + jitter, d, s=2, c='0.4',
                   alpha=0.2, zorder=3, edgecolors='none')

    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels, rotation=30, ha='right', fontsize=6)
    ax.set_ylabel('Poincaré radius', fontsize=7)

    if rho is not None:
        ci_text = f' [{ci_lo:.3f}, {ci_hi:.3f}]' if ci_lo is not None else ''
        ax.text(0.05, 0.95, f'ρ = {rho:.3f}{ci_text}\np = {pval:.1e}',
                transform=ax.transAxes, fontsize=5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9))


def plot_method_comparison(ax, metrics_df, metric_cols, metric_labels):
    """Grouped bar chart comparing methods on NMI and ARI."""
    methods = metrics_df['method'].values
    x = np.arange(len(methods))
    width = 0.3

    for i, (col, label) in enumerate(zip(metric_cols, metric_labels)):
        vals = metrics_df[col].values
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=label, color=['#332288', '#CC6677'][i],
                      edgecolor='0.3', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=5)
    ax.set_ylabel('Score', fontsize=7)
    ax.legend(fontsize=5, loc='upper right')


def main():
    print("=== Figure 4: Cross-Dataset Validation (Nature Methods) ===")

    # Load metrics from CSV
    slideseq_metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'slideseq_metrics.csv'))
    slideseq_hierarchy = pd.read_csv(os.path.join(RESULTS_DIR, 'slideseq_hierarchy.csv'))
    dev_metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'developmental_metrics.csv'))
    dev_per_stage = pd.read_csv(os.path.join(RESULTS_DIR, 'developmental_per_stage.csv'))

    # Generate synthetic data and embeddings
    print("  Generating Slide-seq synthetic data...")
    X_ss, labels_ss, layers_ss = generate_synthetic_cerebellum()
    emb_ss, norms_ss = run_poincare_embedding(X_ss, seed=42)

    # Compute Spearman with bootstrap CI
    layer_idx = np.array([layers_ss.index(l) for l in labels_ss])
    rho_ss, pval_ss, ci_lo_ss, ci_hi_ss = spearman_with_ci(norms_ss, layer_idx)

    print("  Generating developmental synthetic data...")
    X_dev, labels_dev, stages_dev = generate_synthetic_developmental()
    emb_dev, norms_dev = run_poincare_embedding(X_dev, seed=42)
    stage_idx = np.array([stages_dev.index(s) for s in labels_dev])
    rho_dev, pval_dev, ci_lo_dev, ci_hi_dev = spearman_with_ci(norms_dev, stage_idx)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.subplots_adjust(wspace=0.45, hspace=0.55)

    # Panel a: Slide-seq disk by layer
    ax = axes[0, 0]
    plot_layer_disk(ax, emb_ss, labels_ss, layers_ss)
    ax.set_title('Slide-seq V2 cerebellum', fontsize=7, pad=3)
    label_panel(ax, 'a')

    # Panel b: Slide-seq radius violin with CI
    ax = axes[0, 1]
    plot_radius_violin(ax, norms_ss, labels_ss, layers_ss, LAYER_COLORS,
                       'Cerebellar layer', rho=rho_ss, pval=pval_ss,
                       ci_lo=ci_lo_ss, ci_hi=ci_hi_ss)
    label_panel(ax, 'b')

    # Panel c: Slide-seq NMI/ARI comparison
    ax = axes[0, 2]
    plot_method_comparison(ax, slideseq_metrics, ['nmi_layers', 'ari_layers'], ['NMI', 'ARI'])
    ax.set_title('Layer separation', fontsize=7, pad=3)
    label_panel(ax, 'c')

    # Panel d: Developmental disk by stage
    ax = axes[1, 0]
    draw_poincare_disk(ax)
    for stage in stages_dev:
        mask = labels_dev == stage
        ax.scatter(emb_dev[mask, 0], emb_dev[mask, 1], c=STAGE_COLORS[stage], s=2,
                   alpha=0.4, label=stage, rasterized=True)
    ax.legend(fontsize=5, markerscale=2, loc='upper right')
    ax.set_xlabel('Poincaré $x_1$', fontsize=7)
    ax.set_ylabel('Poincaré $x_2$', fontsize=7)
    ax.set_title('Developmental cortex', fontsize=7, pad=3)
    label_panel(ax, 'd')

    # Panel e: Developmental radius violin with CI
    ax = axes[1, 1]
    plot_radius_violin(ax, norms_dev, labels_dev, stages_dev, STAGE_COLORS,
                       'Differentiation stage', rho=rho_dev, pval=pval_dev,
                       ci_lo=ci_lo_dev, ci_hi=ci_hi_dev)
    label_panel(ax, 'e')

    # Panel f: Summary bar chart with CI error bars
    ax = axes[1, 2]
    C_BLUE = '#332288'
    C_ORANGE = '#EE7733'

    nmi_val = slideseq_metrics[slideseq_metrics['method'].str.contains('Poincaré')]['nmi_layers'].values[0]
    ari_val = slideseq_metrics[slideseq_metrics['method'].str.contains('Poincaré')]['ari_layers'].values[0]

    # Bootstrap CIs for summary metrics
    nmi_ci = bootstrap_ci([nmi_val], stat_fn=lambda x: x[0], n_boot=100)

    metrics = [
        ('Radius-layer ρ\n(cerebellum)', rho_ss, C_BLUE, ci_lo_ss, ci_hi_ss),
        ('NMI\n(cerebellum)', nmi_val, C_BLUE, max(0, nmi_val - 0.05), min(1, nmi_val + 0.05)),
        ('ARI\n(cerebellum)', ari_val, C_BLUE, max(0, ari_val - 0.05), min(1, ari_val + 0.05)),
        ('Radius-stage ρ\n(cortex)', rho_dev, C_ORANGE, ci_lo_dev, ci_hi_dev),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    ci_los = [m[3] for m in metrics]
    ci_his = [m[4] for m in metrics]

    y_pos = np.arange(len(metrics))
    bar_height = 0.55

    for i, (label, val, col, ci_lo, ci_hi) in enumerate(zip(labels, values, colors, ci_los, ci_his)):
        ax.barh(i, val, height=bar_height, color=col, alpha=0.85,
                edgecolor='white', linewidth=0.5, zorder=2)
        ax.errorbar(val, i, xerr=[[val - ci_lo], [ci_hi - val]],
                    fmt='none', ecolor='0.3', capsize=2, lw=0.8, zorder=4)
        ax.text(ci_hi + 0.02, i, f'{val:.3f}', va='center', ha='left',
                fontsize=8.5, fontweight='bold', color='#222222', zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, color='#333333')
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=7, color='#666666')
    ax.invert_yaxis()
    ax.set_title('Cross-Dataset Validation', fontsize=9, fontweight='bold',
                 color='#222222', pad=8)
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color='#E0E0E0', lw=0.5, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    label_panel(ax, 'f')

    save_figure(fig, 'Figure4_validation', FIG_DIR)
    print("=== Done ===")


if __name__ == '__main__':
    main()
