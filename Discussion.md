# Discussion

We have introduced Poincaré MDS, a method for embedding spatial transcriptomics data into hyperbolic space, and the Hyperbolic Niche, a neighborhood definition based on geodesic distance. Our results demonstrate that hyperbolic geometry provides a natural framework for representing the hierarchical organization of biological tissues.

## Advantages of hyperbolic geometry for spatial data

The key advantage of Poincaré MDS over Euclidean methods is the radial coordinate, which encodes hierarchical depth. In our analysis, the Poincaré radius correlated with tree depth in synthetic data (r = 0.853) and with biological organization in real data: tumor cells at the center, stromal and immune cells at the periphery. This radial hierarchy is a unique property of hyperbolic geometry, absent in Euclidean MDS and t-SNE.

The Hyperbolic Niche leverages this hierarchical structure to define more biologically meaningful neighborhoods. Hyperbolic niches achieved significantly higher purity than Euclidean niches (0.672 vs. 0.439, p = 5.1 × 10⁻¹³⁵), indicating that geodesic distance better captures neighborhoods of similar cell types. This advantage was consistent across all 10 tissue sections tested.

## Biological implications

The radial organization revealed by Poincaré MDS reflects the known hierarchical structure of gastric cancer tissue. Tumor cells, which form compact clusters, are positioned at the disk center. Stromal cells (fibroblasts, CAFs), which form a surrounding shell, are positioned at the periphery. Immune cells occupy intermediate positions, consistent with their infiltration into the tumor margin.

The differential performance of hyperbolic vs. Euclidean niches for different interaction pairs is also biologically informative. Hyperbolic niches better captured immune-stromal interactions (T cell–Macrophage, Endothelial–Fibroblast), which are governed by functional relationships rather than physical proximity. Euclidean niches better captured tumor-stroma interactions (CAF–Tumor, Epithelial–CAF), which are primarily driven by spatial adjacency. This suggests that hyperbolic and Euclidean distances capture complementary aspects of tissue organization.

## Comparison with existing methods

Poincaré MDS matches Euclidean MDS on distance preservation metrics while providing the additional benefit of hierarchical encoding. The k-NN retention rate (0.478) is comparable to Euclidean MDS (0.501), indicating that local neighborhood structure is preserved. The niche purity advantage (0.672 vs. 0.439) demonstrates that the hierarchical encoding provides biologically meaningful information beyond what Euclidean methods capture.

Compared to t-SNE and UMAP, Poincaré MDS has the advantage of preserving global distance structure (r = 0.919 for tree distances, vs. r = 0.427 for t-SNE). This makes it more suitable for analyses that require meaningful distance comparisons, such as niche analysis and trajectory inference.

## Limitations and future directions

Several limitations should be acknowledged. First, the k-NN retention rate (0.478) is lower than ideal, suggesting that some local neighborhood information is lost in the embedding. Future work could explore hybrid approaches that combine hyperbolic and Euclidean distance metrics. Second, the method currently operates on 2D embeddings; extensions to higher-dimensional hyperbolic space (e.g., the Poincaré ball in 3D) may capture additional structure. Third, the optimal curvature parameter may vary across datasets; adaptive curvature selection could improve performance.

The Hyperbolic Niche concept could be extended in several directions. Dynamic niche definitions, where the radius adapts to local density, may better capture variable-density tissues. Multi-scale niche analysis, using multiple radius thresholds, could reveal hierarchical organization at different spatial scales. Integration with cell-cell communication tools (e.g., CellChat) could provide mechanistic insights into the interactions captured by hyperbolic niches.

## Conclusions

Poincaré MDS and Hyperbolic Niche provide a geometric framework for spatial transcriptomics analysis that respects the hierarchical organization of biological tissues. The radial coordinate in the Poincaré disk encodes hierarchical depth, and geodesic distance neighborhoods capture cell-cell interactions more faithfully than Euclidean neighborhoods. These properties make hyperbolic geometry a natural choice for analyzing spatial transcriptomics data, with potential applications in cancer biology, developmental biology, and neuroscience.
