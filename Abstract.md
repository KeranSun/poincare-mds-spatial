# Hyperbolic Geometry for Spatial Transcriptomics: Abstract

## Title
Poincaré MDS: Hyperbolic Embedding Reveals Hierarchical Organization in Spatial Transcriptomics

## Abstract

Spatial transcriptomics technologies generate high-dimensional gene expression data with spatial coordinates, but existing dimensionality reduction methods (UMAP, t-SNE) do not explicitly model the hierarchical organization of tissues. Here we introduce **Poincaré MDS**, a method that embeds spatial transcriptomics data into the Poincaré disk model of hyperbolic space. Unlike Euclidean embeddings, the radial coordinate in the Poincaré disk naturally encodes hierarchical depth: central positions represent general cell types, while peripheral positions represent specialized subtypes.

We further introduce the **Hyperbolic Niche**, a neighborhood definition based on geodesic distance in the Poincaré disk, and demonstrate that it captures cell-cell interactions more faithfully than Euclidean neighborhoods. Applied to gastric cancer spatial transcriptomics (10x Visium, 4,252 spots), Poincaré MDS achieves:

- **Radial hierarchy**: Poincaré radius correlates with tissue organization (tumor core at center, stroma at periphery)
- **Hyperbolic Niche purity**: 0.672 vs 0.439 for Euclidean niches (p = 5.1 × 10^{-135})
- **Interaction enrichment**: Hyperbolic niches better capture T cell–Macrophage and Endothelial–Fibroblast interactions
- **Ground truth validation**: On synthetic hierarchical data, Poincaré radius recovers tree depth (r = 0.853), a property absent in Euclidean MDS and t-SNE

Poincaré MDS provides a geometric framework for analyzing spatial transcriptomics that respects the inherent hierarchy of tissue organization.

## Key Results

| Metric | Poincaré MDS | Euclidean MDS | t-SNE |
|--------|-------------|---------------|-------|
| Tree distance correlation | 0.919 | 0.926 | 0.427 |
| Radius-depth correlation | **0.853** | -0.104 | -0.088 |
| k-NN retention | 0.478 | 0.501 | 0.384 |
| Niche purity | **0.672** | 0.439 | - |
| Hyp-Spatial correlation | **0.397** | 0.363 | - |

## Significance for Nature Methods

1. **Novel method**: First application of Poincaré MDS to spatial transcriptomics
2. **New concept**: Hyperbolic Niche — geodesic-distance neighborhoods for cell-cell interaction analysis
3. **Ground truth validation**: Synthetic hierarchical data with known tree structure
4. **Biological insight**: Radial hierarchy in Poincaré disk corresponds to tissue organization (tumor core → stroma → immune)
5. **Quantitative advantage**: Statistically significant improvements over Euclidean methods (p < 10^{-100})

## Figure Overview

- **Figure 1**: Method overview — Poincaré disk embedding, spatial mapping, cell type distribution, distance fidelity
- **Figure 2**: Hyperbolic Niche analysis — niche definition, interaction enrichment, purity comparison
- **Figure 3**: Ground truth validation — binary tree simulation, radius-depth correlation, method comparison
- **Figure 4**: Quantitative summary — benchmark results, summary table, Poincaré geometry diagram
