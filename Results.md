# Results

## Poincaré MDS embeds spatial transcriptomics into hyperbolic space

We developed Poincaré MDS, a method that embeds spatial transcriptomics data into the Poincaré disk model of hyperbolic space. Unlike Euclidean dimensionality reduction methods (UMAP, t-SNE), the Poincaré disk has a natural radial coordinate that encodes hierarchical depth: central positions correspond to general cell types, while peripheral positions correspond to specialized subtypes (**Figure 1A**).

Applied to gastric cancer 10x Visium data (4,252 spots, 21,503 genes), Poincaré MDS produced a well-structured embedding with all points inside the unit disk (max radius = 0.641, mean = 0.384, std = 0.063). The embedding preserved distance structure with high fidelity: the correlation between Poincaré geodesic distances and PCA Euclidean distances was r = 0.887, and the correlation with k-NN graph distances was r = 0.913 (**Figure 1F**).

Importantly, Poincaré MDS captured spatial information better than PCA alone. The correlation between Poincaré distances and spatial coordinates (r = 0.397) exceeded that of PCA distances (r = 0.363), indicating that hyperbolic geometry provides a more faithful representation of tissue organization.

## Radial hierarchy reflects tissue organization

The Poincaré radius showed meaningful biological correlations (**Figure 1E**). Tumor marker scores (MUC5AC, CEACAM5, REG4) were negatively correlated with radius (r = −0.10), indicating that tumor cells concentrate toward the disk center. In contrast, fibroblast markers (COL1A1, COL1A2, DCN, LUM) and CAF markers (FAP, POSTN, ACTA2) showed positive correlations (r = 0.17 and r = 0.18, respectively), positioning stromal cells at the periphery (**Figure 1D**).

K-means clusters exhibited distinct radial distributions (**Figure 1C**): some clusters were concentrated near the center (tumor-enriched), while others spanned the full radius (stromal/immune), consistent with the known hierarchical organization of gastric cancer tissue.

## Hyperbolic Niche captures cell-cell interactions

We introduced the Hyperbolic Niche, a neighborhood definition based on geodesic distance in the Poincaré disk. For each spot, we defined its hyperbolic niche as all spots within the 10th-percentile geodesic distance threshold, and compared this against Euclidean niches defined by spatial distance.

Hyperbolic niches achieved significantly higher purity than Euclidean niches (0.672 ± 0.187 vs. 0.439 ± 0.141; Mann-Whitney U test, p = 5.1 × 10⁻¹³⁵; **Figure 2C**). This indicates that hyperbolic distance better captures neighborhoods of similar cell types, consistent with the hierarchical structure encoded in the Poincaré disk.

Analysis of cell-cell interaction pairs revealed that hyperbolic niches better captured immune-stromal interactions: T cell–Macrophage enrichment was higher in hyperbolic niches (1.266 vs. 1.241, p = 1.12 × 10⁻⁹), as was Endothelial–Fibroblast enrichment (1.811 vs. 1.756, p = 4.51 × 10⁻¹³; **Figure 2B**). Euclidean niches performed better for tumor-associated interactions (CAF–Tumor and Epithelial–CAF), likely because these interactions are primarily driven by physical proximity rather than functional hierarchy.

## Multi-sample validation confirms reproducibility

To assess reproducibility, we applied Poincaré MDS to all 10 Visium samples in the dataset (**Supplementary Figure 1**). Across all samples:

- Poincaré distances correlated with spatial distances (mean r = 0.324 ± 0.128), exceeding PCA-spatial correlations in 9/10 samples (mean r = 0.291 ± 0.115)
- Hyperbolic niche purity exceeded Euclidean niche purity in all 10 samples (10/10; mean Hyp purity = 0.794 ± 0.055 vs. mean Euc purity = 0.693 ± 0.096; sign test p = 9.8 × 10⁻⁴)
- All embeddings remained within the Poincaré disk (10/10 samples)
- Hyp-PCA distance correlation was consistently high (mean r = 0.897 ± 0.044)
- Hyp-graph distance correlation was consistently high (mean r = 0.931 ± 0.032)

These results demonstrate that the advantages of hyperbolic embedding are robust across samples and not specific to a single tissue section.

## Ground truth validation with hierarchical synthetic data

To validate that Poincaré MDS recovers hierarchical structure, we generated synthetic data with a binary tree topology (depth = 4, 16 leaf nodes, 480 samples). Each leaf node represented a distinct cell type, with tree distances defined by path length through the lowest common ancestor (**Figure 3A**).

Poincaré MDS achieved a radius-depth correlation of r = 0.853 (Spearman), meaning that points deeper in the tree hierarchy were positioned farther from the disk center (**Figure 3E**). In contrast, Euclidean MDS showed no radius-depth correlation (r = −0.104), and t-SNE also showed no correlation (r = −0.088). This result confirms that the radial coordinate in the Poincaré disk is a unique feature of hyperbolic embedding, absent in Euclidean methods.

Tree distance preservation was comparable between Poincaré MDS (r = 0.919) and Euclidean MDS (r = 0.926), but the hierarchical encoding provided by the radial coordinate is an additional layer of information not available in Euclidean embeddings (**Figure 3F**).

## Benchmark against existing methods

We compared Poincaré MDS against Euclidean MDS and t-SNE on multiple metrics (**Figure 4**, **Table 1**):

| Metric | Poincaré MDS | Euclidean MDS | t-SNE |
|--------|-------------|---------------|-------|
| Tree distance correlation | 0.919 | 0.926 | 0.427 |
| Radius-depth correlation | **0.853** | −0.104 | −0.088 |
| k-NN retention (k=15) | 0.478 | 0.501 | 0.384 |
| Niche purity | **0.672** | 0.439 | — |
| Spatial correlation | **0.397** | 0.363 | — |

Poincaré MDS matched or exceeded Euclidean MDS on distance preservation metrics while providing the unique advantage of hierarchical encoding via the radial coordinate. The niche purity advantage was consistent across all 10 samples tested.

## Radial organization reveals biological structure

The Poincaré radius encoded meaningful biological information beyond simple spatial position. Examining the relationship between radius and cell type marker expression (**Figure 1E**), we observed:

- **Tumor markers** (MUC5AC, CEACAM5): concentrated at the disk center (r = −0.10), reflecting the compact nature of tumor regions
- **CAF markers** (FAP, POSTN): enriched at the periphery (r = 0.18), consistent with CAFs forming a surrounding stromal shell
- **Fibroblast markers** (COL1A1, DCN): peripheral distribution (r = 0.17), reflecting normal stromal architecture

This radial organization mirrors the biological hierarchy of gastric cancer: tumor core → invasive margin → stroma → immune infiltrate. The Poincaré disk naturally encodes this hierarchy through its geometry, providing a geometric framework for understanding tissue organization.
