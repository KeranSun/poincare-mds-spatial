# Methods: Hyperbolic Geometry for Spatial Transcriptomics

## Poincaré Multidimensional Scaling (Poincaré MDS)

We developed Poincaré MDS, a method for embedding high-dimensional spatial transcriptomics data into the Poincaré disk model of hyperbolic space. Given a gene expression matrix X ∈ R^{n×d} with n spots and d genes, the method proceeds as follows:

### Distance Matrix Computation

We first compute a k-nearest neighbor (k-NN) graph in PCA space (k=30, using the first 10 principal components). The target distance matrix D_target is defined as the shortest-path distance on this k-NN graph, which preserves local neighborhood structure while providing a global distance metric that respects the data manifold.

### Initialization via Classical MDS

The embedding is initialized using Torgerson scaling (classical MDS) applied to D_target. Specifically, we compute the double-centered matrix B = -0.5 H D_target^2 H, where H = I - 1/n · 11^T is the centering matrix. The top 2 eigenvectors of B, scaled by the square root of their eigenvalues, provide initial coordinates in R^2. These coordinates are scaled to lie within the unit disk and projected onto the Poincaré ball using the exponential map.

### Optimization

The embedding coordinates θ ∈ R^{n×2} are optimized to minimize the stress function:

L_MDS = (1/|S|) Σ_{(i,j)∈S} (d_H(θ_i, θ_j) - D_target(i,j))²

where d_H is the geodesic distance on the Poincaré ball with curvature c:

d_H(x, y) = (1/√c) · arcosh(1 + 2c · ||x - y||² / ((1 - c||x||²)(1 - c||y||²)))

and S is a mini-batch of randomly sampled point pairs (|S| = 100,000 per epoch). We use Riemannian Adam optimization (learning rate = 0.05, weight decay = 10^{-5}) with projection onto the ball after each step. The curvature parameter c is set to 0.5 to allow sufficient embedding spread while maintaining hyperbolic geometry properties.

A repulsion loss L_radius = max(0, r_target - mean(||θ_i||))² is added to prevent embedding collapse toward the disk center, with r_target = 0.4.

The total loss is L = L_MDS + 0.5 · L_radius, optimized for 2,000 epochs.

### Hyperbolic Niche Analysis

We define the **Hyperbolic Niche** of a spot i as the set of spots within a geodesic distance threshold r on the Poincaré disk:

N_H(i) = {j : d_H(θ_i, θ_j) < r}

The threshold r is set to the 10th percentile of all pairwise hyperbolic distances, ensuring comparable niche sizes across methods. For comparison, we define the **Euclidean Niche** analogously using spatial Euclidean distances.

For each niche, we compute:
1. **Cell type enrichment**: The mean module score of each cell type within the niche, where module scores are defined as the average expression of marker gene sets (e.g., EPCAM, KRT18, KRT19, KRT8 for Epithelial cells).
2. **Niche purity**: The proportion of the most abundant cluster (K-means, k=8) within the niche.

Statistical comparisons between hyperbolic and Euclidean niches use the Mann-Whitney U test.

### Simulation with Ground Truth Hierarchy

To validate that Poincaré MDS recovers hierarchical structure, we generated synthetic data with a binary tree topology (depth = 4, 16 leaf nodes). Each leaf node corresponds to a distinct cell type, and the tree distance between leaves equals the path length through their lowest common ancestor. Data points are generated as leaf-specific feature vectors with Gaussian noise (σ = 0.3).

We embed the synthetic data using Poincaré MDS, Euclidean MDS, and t-SNE, and evaluate:
- **Tree distance correlation**: Pearson r between embedded and true tree distances
- **Radius-depth correlation**: Spearman r between Poincaré radius and tree depth (distance from root)
- **k-NN retention rate**: Proportion of k-nearest neighbors (k=15) in tree distance preserved in the embedding

### Implementation

All computations were performed in Python 3.7 using PyTorch 1.12 and geoopt 0.5.0 for Riemannian optimization. Spatial transcriptomics data (10x Visium, gastric cancer) were preprocessed using Scanpy (1.9.1). The full pipeline processes 4,252 spots in approximately 15 minutes on a 6-core CPU with 32 GB RAM.

### Data Availability

The spatial transcriptomics data used in this study is available at [repository]. Code is available at [repository].

### Code Availability

The Poincaré MDS implementation is available as a Python package at [repository]. The analysis pipeline includes:
- `hyperbolic_embedding_v3.py`: Core Poincaré MDS implementation
- `hyperbolic_niche_full.py`: Hyperbolic Niche analysis
- `simulate_tree.py`: Ground truth validation with binary tree structure
- `figure_assembly.py`: Figure generation
