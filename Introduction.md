# Introduction

Spatial transcriptomics technologies, including 10x Visium, MERFISH, and Slide-seq, enable simultaneous measurement of gene expression and spatial position in tissue sections. These methods generate high-dimensional data with inherent spatial structure, requiring computational tools that can capture both molecular and spatial relationships.

Dimensionality reduction is a fundamental step in the analysis of spatial transcriptomics data. Current methods, including UMAP, t-SNE, and PHATE, embed high-dimensional gene expression data into low-dimensional Euclidean space for visualization and downstream analysis. While these methods are effective for revealing cluster structure, they do not explicitly model the hierarchical organization that characterizes biological tissues.

Biological tissues exhibit hierarchical organization at multiple scales: cell types form functional modules, which assemble into tissue regions, which compose organs. This hierarchy is particularly evident in cancer, where tumor cells, stromal cells, and immune cells form nested organizational units with distinct spatial relationships. Euclidean embeddings do not naturally capture this hierarchy — all points are equidistant from the origin, and the radial coordinate has no intrinsic meaning.

Hyperbolic geometry provides a natural framework for representing hierarchical structures. In the Poincaré disk model of hyperbolic space, the distance from the center increases exponentially toward the boundary, creating a natural "zoom" effect: points near the center represent general categories, while points near the boundary represent specific instances. This property has been exploited in natural language processing (Poincaré embeddings for word hierarchies) and network analysis, but has not been applied to spatial transcriptomics.

Here we introduce Poincaré MDS, a method for embedding spatial transcriptomics data into the Poincaré disk. We demonstrate that:

1. The radial coordinate in the Poincaré disk encodes hierarchical depth, with tumor cells at the center and stromal/immune cells at the periphery
2. Hyperbolic distance neighborhoods ("Hyperbolic Niche") capture cell-cell interactions more faithfully than Euclidean neighborhoods
3. The method recovers known hierarchical structure in synthetic data with ground truth
4. These properties are reproducible across multiple tissue sections
