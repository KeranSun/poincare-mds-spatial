# Poincaré MDS: Hyperbolic Embedding for Spatial Transcriptomics

A Python package for embedding spatial transcriptomics data into the Poincaré disk model of hyperbolic space.

## Key Features

- **Poincaré MDS**: Embed high-dimensional data into the Poincaré disk preserving local neighborhood structure
- **Hyperbolic Niche**: Define cell neighborhoods using geodesic distance for cell-cell interaction analysis
- **Radial Hierarchy**: The Poincaré radius naturally encodes hierarchical depth (center = general, periphery = specialized)

## Installation

```bash
pip install -e .
```

### Dependencies

- Python >= 3.7
- PyTorch >= 1.9
- geoopt >= 0.5.0
- scipy >= 1.5
- scikit-learn >= 0.24
- scanpy >= 1.8

## Quick Start

```python
from poincare_mds import PoincareMDS, HyperbolicNiche
import scanpy as sc

# Load spatial transcriptomics data
adata = sc.read_h5ad('spatial_data.h5ad')
X_pca = adata.obsm['X_pca'][:, :10]

# Poincaré MDS embedding
model = PoincareMDS(curvature=0.5, n_epochs=2000)
embedding = model.fit_transform(X_pca)

# Get radial norms (hierarchy depth)
norms = model.get_norms()

# Hyperbolic Niche analysis
niche = HyperbolicNiche(curvature=0.5)
results = niche.analyze(
    hyp_embedding=embedding,
    euclidean_distances=D_euclidean,
    scores=cell_type_scores,
)

print(f"Niche purity: Hyperbolic={results['hyp_purity_mean']:.3f} vs "
      f"Euclidean={results['euc_purity_mean']:.3f}")
print(f"p-value: {results['purity_pval']:.2e}")
```

## Method Overview

### Poincaré MDS

1. Compute k-NN graph shortest-path distances in PCA space
2. Initialize embedding via Torgerson scaling (classical MDS)
3. Optimize stress function using Riemannian Adam:
   - L = Σ(d_H(θ_i, θ_j) - D_target(i,j))² + λ · max(0, r_target - mean(||θ_i||))²
4. Project onto Poincaré ball after each optimization step

### Hyperbolic Niche

Define the niche of spot i as all spots within geodesic distance threshold r:

N_H(i) = {j : d_H(θ_i, θ_j) < r}

Compare against Euclidean niches using:
- Niche purity (cluster homogeneity)
- Cell type enrichment scores

## Citation

If you use this method in your research, please cite:

```
[Citation to be added upon publication]
```

## License

MIT License
