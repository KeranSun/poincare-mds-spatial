"""
Generate supplementary tables (Markdown format).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submission')


def main():
    print("=== Generating Supplementary Tables ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tables = []

    # Table 1: Full benchmark
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'enhanced_benchmark.csv'))
    tables.append("## Supplementary Table 1. Full Benchmark Metrics\n")
    tables.append(df.to_markdown(index=False, floatfmt='.4f'))
    tables.append("\n")

    # Table 2: HDS pairwise
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hierarchical_differential_score.csv'))
    df['mannwhitney_pval'] = df['mannwhitney_pval'].apply(lambda x: f'{x:.2e}')
    tables.append("## Supplementary Table 2. Hierarchical Differential Score (HDS)\n")
    tables.append(df.to_markdown(index=False))
    tables.append("\n")

    # Table 3: Zone marker analysis
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'zone_marker_analysis.csv'))
    tables.append("## Supplementary Table 3. Zone Marker Analysis (Kruskal–Wallis)\n")
    tables.append(df.to_markdown(index=False, floatfmt='.4f'))
    tables.append("\n")

    # Table 4: Scalability
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'scalability_results.csv'))
    tables.append("## Supplementary Table 4. Scalability Results\n")
    tables.append(df.to_markdown(index=False, floatfmt='.2f'))
    tables.append("\n")

    # Table 5: Slide-seq metrics
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'slideseq_metrics.csv'))
    tables.append("## Supplementary Table 5. Slide-seq Validation Metrics\n")
    tables.append(df.to_markdown(index=False, floatfmt='.4f'))
    tables.append("\n")

    # Table 6: Per-cluster silhouette
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hyperbolic_clusters.csv'))
    tables.append("## Supplementary Table 6. Per-Cluster Silhouette Scores\n")
    tables.append(df.to_markdown(index=False, floatfmt='.4f'))
    tables.append("\n")

    content = '\n'.join(tables)
    out_path = os.path.join(OUTPUT_DIR, 'supplementary_tables.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Saved: {out_path}")
    print("=== Done ===")


if __name__ == '__main__':
    main()
