# Poincaré MDS: Hyperbolic Embedding Reveals Hierarchical Organization in Spatial Transcriptomics

## Authors
[Author names]

## Affiliations
[Affiliations]

## Corresponding Author
[Name, Email]

---

## Abstract

Spatial transcriptomics data exhibits hierarchical organization that existing dimensionality reduction methods do not explicitly capture. We introduce Poincaré MDS, which embeds spatial transcriptomics data into the Poincaré disk model of hyperbolic space, where the radial coordinate naturally encodes hierarchical depth. We also introduce Hyperbolic Niche, a neighborhood definition based on geodesic distance. Applied to 10 gastric cancer Visium samples, Poincaré MDS achieves higher spatial correlation than Euclidean methods (0.404 vs. 0.330), and Hyperbolic Niche purity exceeds Euclidean niche purity in all 10 samples (mean 0.794 vs. 0.693). On synthetic hierarchical data, Poincaré radius recovers tree depth (r = 0.853), a property absent in comparison methods. Poincaré MDS provides a geometric framework for spatial transcriptomics that respects tissue hierarchy.

---

## Introduction

Spatial transcriptomics technologies enable simultaneous measurement of gene expression and spatial position in tissue sections [1,2]. Dimensionality reduction is fundamental to analyzing these high-dimensional data, with UMAP [3], t-SNE [4], and PHATE [5] being widely used. While these methods reveal cluster structure, they do not model the hierarchical organization that characterizes biological tissues — cell types form functional modules that assemble into tissue regions at multiple scales.

Hyperbolic geometry provides a natural framework for hierarchical structures. In the Poincaré disk, distance from the center increases exponentially toward the boundary, creating a natural "zoom" effect: central positions represent general categories, while peripheral positions represent specific instances [6]. Poincaré embeddings have been applied to word hierarchies [7] and network analysis [8], but not to spatial transcriptomics.

Here we introduce Poincaré MDS for embedding spatial transcriptomics data into the Poincaré disk, and Hyperbolic Niche for analyzing cell-cell interactions using geodesic distance. We demonstrate that the radial coordinate encodes hierarchical depth, that Hyperbolic Niche captures interactions more faithfully than Euclidean neighborhoods, and that these properties are reproducible across multiple tissue sections.

---

## Results

### Poincaré MDS embeds spatial transcriptomics into hyperbolic space

Poincaré MDS operates in three steps: (1) compute k-NN graph shortest-path distances in PCA space, (2) initialize embedding via Torgerson scaling, (3) optimize the stress function using Riemannian Adam [9] with a repulsion loss to prevent embedding collapse (Methods).

Applied to gastric cancer 10x Visium data (4,252 spots, 21,503 genes), Poincaré MDS produced a well-structured embedding (max radius = 0.641, mean = 0.384). The correlation between Poincaré geodesic distances and k-NN graph distances was r = 0.913, and with PCA distances was r = 0.887. Poincaré distances correlated with spatial coordinates (r = 0.404), exceeding Euclidean MDS (r = 0.330) and PHATE (r = 0.374).

### Radial hierarchy reflects tissue organization

The Poincaré radius showed meaningful biological correlations: tumor markers (MUC5AC, CEACAM5) concentrated at the disk center (r = −0.10), while fibroblast markers (COL1A1, DCN) and CAF markers (FAP, POSTN) were peripheral (r = 0.17 and 0.18, respectively). This radial organization mirrors gastric cancer tissue hierarchy: tumor core → invasive margin → stroma → immune infiltrate.

### Hyperbolic Niche captures cell-cell interactions

We defined the Hyperbolic Niche of each spot as all spots within the 10th-percentile geodesic distance threshold. Hyperbolic niches achieved significantly higher purity than Euclidean niches (0.672 ± 0.187 vs. 0.439 ± 0.141; Mann-Whitney U test, p = 5.1 × 10⁻¹³⁵). Analysis of interaction pairs revealed that hyperbolic niches better captured immune-stromal interactions (T cell–Macrophage: 1.266 vs. 1.241, p = 1.12 × 10⁻⁹; Endothelial–Fibroblast: 1.811 vs. 1.756, p = 4.51 × 10⁻¹³), while Euclidean niches better captured tumor-proximal interactions.

### Multi-sample validation

Across all 10 Visium samples: Poincaré spatial correlations (mean 0.324 ± 0.128) exceeded PCA-spatial correlations in 9/10 samples (mean 0.291 ± 0.115). Hyperbolic niche purity exceeded Euclidean niche purity in all 10 samples (10/10; mean 0.794 ± 0.055 vs. 0.693 ± 0.096). All embeddings remained within the Poincaré disk.

### Ground truth validation

To validate hierarchical recovery, we generated synthetic data with a binary tree topology (depth = 4, 16 leaf nodes, 480 samples). Poincaré MDS achieved a radius-depth correlation of r = 0.853, while Euclidean MDS showed no correlation (r = −0.104) and t-SNE also showed none (r = −0.088). This confirms that the radial coordinate in the Poincaré disk uniquely encodes hierarchical depth.

### Benchmark

We compared Poincaré MDS against PHATE, Euclidean MDS, and t-SNE (**Table 1**). Poincaré MDS achieved the highest spatial correlation among distance-preserving methods. t-SNE achieved higher spatial correlation (0.439) but distorted global distances (tree distance correlation 0.427 vs. 0.919). PHATE achieved slightly higher niche purity on a single sample (0.710 vs. 0.675) but does not provide hierarchical encoding. Poincaré MDS uniquely combines distance preservation with hierarchical structure.

**Table 1. Method comparison**

| Metric | Poincaré MDS | PHATE | Euclidean MDS | t-SNE |
|--------|-------------|-------|---------------|-------|
| Spatial correlation | **0.404** | 0.374 | 0.330 | 0.439 |
| k-NN retention | 0.123 | 0.242 | 0.140 | **0.467** |
| Niche purity (1 sample) | 0.675 | **0.710** | 0.658 | 0.683 |
| Niche purity (10 samples) | **0.794** | — | 0.693 | — |
| Radius-depth (synthetic) | **0.853** | — | −0.104 | −0.088 |

---

## Discussion

Poincaré MDS and Hyperbolic Niche provide a geometric framework for spatial transcriptomics that respects tissue hierarchy. The key advantage over Euclidean methods is the radial coordinate, which encodes hierarchical depth — a property validated on synthetic data (r = 0.853) and consistent with biological organization in real data.

Compared to PHATE and t-SNE, Poincaré MDS has distinct trade-offs. t-SNE achieves higher spatial correlation and k-NN retention but distorts global distances. PHATE achieves slightly higher niche purity but does not provide hierarchical encoding. Poincaré MDS uniquely combines distance preservation with hierarchical structure, making it the only method that simultaneously preserves global relationships and reveals tissue hierarchy.

The differential performance of hyperbolic vs. Euclidean niches for different interaction pairs is biologically informative. Hyperbolic niches better capture immune-stromal interactions governed by functional relationships, while Euclidean niches better capture tumor-proximal interactions driven by physical adjacency, suggesting that the two distance metrics capture complementary aspects of tissue organization.

Limitations include lower k-NN retention (0.123) compared to t-SNE (0.467), suggesting local neighborhood structure is partially sacrificed for global hierarchy preservation. Future work could explore hybrid approaches and adaptive curvature selection.

---

## Methods

### Poincaré MDS

Given gene expression matrix X ∈ R^{n×d}, we compute a k-NN graph (k=30) in PCA space and define target distances as shortest-path distances. The embedding is initialized via Torgerson scaling and optimized using Riemannian Adam with the stress function:

L = (1/|S|) Σ_{(i,j)∈S} (d_H(θ_i, θ_j) − D_target(i,j))² + λ · max(0, r_target − mean(||θ_i||))²

where d_H is the geodesic distance on the Poincaré ball with curvature c=0.5, |S|=100,000 mini-batch pairs per epoch, r_target=0.4, λ=0.5. Optimization runs for 2,000 epochs with learning rate 0.05.

### Hyperbolic Niche

The Hyperbolic Niche of spot i is defined as N_H(i) = {j : d_H(θ_i, θ_j) < r}, where r is the 10th percentile of all pairwise hyperbolic distances. Niche purity is the proportion of the most abundant K-means cluster (k=8) within the niche. Cell type enrichment is the mean module score of marker gene sets within the niche.

### Simulation

Synthetic hierarchical data uses a binary tree (depth=4, 16 leaf nodes). Each leaf has a unique feature vector; data points are generated with Gaussian noise (σ=0.3). Tree distance equals path length through the lowest common ancestor.

### Implementation

Python 3.7, PyTorch 1.12, geoopt 0.5.0. Code: https://github.com/KeranSun/poincare-mds-spatial. Processing time: ~15 minutes for 4,000 spots on a 6-core CPU.

---

## References

1. Ståhl, P.L. et al. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science 353, 78–82 (2016).
2. Marx, V. Method of the Year: spatially resolved transcriptomics. Nat. Methods 18, 9–14 (2021).
3. McInnes, L., Healy, J. & Melville, J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426 (2018).
4. van der Maaten, L. & Hinton, G. Visualizing data using t-SNE. J. Mach. Learn. Res. 9, 2579–2605 (2008).
5. Moon, K.R. et al. Visualizing structure and transitions in high-dimensional biological data. Nat. Biotechnol. 37, 1482–1492 (2019).
6. Cannon, J.W. et al. Hyperbolic Geometry. in Flavors of Geometry 59–115 (MSRI, 1997).
7. Nickel, M. & Kiela, D. Poincaré embeddings for learning hierarchical representations. in Advances in Neural Information Processing Systems 30 (2017).
8. Muscoloni, A. et al. Machine learning meets complex networks via coalescent embedding in the hyperbolic space. Nat. Commun. 8, 1615 (2017).
9. Becigneul, G. & Ganea, O.-E. Riemannian adaptive optimization methods. in International Conference on Learning Representations (2019).

---

## Acknowledgments
[To be filled]

## Author Contributions
[To be filled]

## Competing Interests
The authors declare no competing interests.

## Data Availability
Spatial transcriptomics data is publicly available. Code is available at https://github.com/KeranSun/poincare-mds-spatial.
