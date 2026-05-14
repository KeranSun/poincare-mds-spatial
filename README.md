# Poincaré MDS

**Hyperbolic Embedding for Spatial Transcriptomics**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Poincaré MDS embeds spatial transcriptomics data into hyperbolic space, where **radial distance from the disk center encodes hierarchical layer position**. The companion framework **Hyperbolic Niche** defines cell neighborhoods using geodesic distance and detects spatially coherent ligand-receptor interactions.

**Key results** (Slide-seq V2 mouse olfactory bulb):
- Radius-layer Spearman ρ = 0.796 (p = 5.1 × 10⁻¹³⁵)
- Hyperbolic niche purity exceeds Euclidean niche purity (Mann-Whitney p < 10⁻¹⁰)
- Geodesic L-R co-expression detects receptor-enriched spatial niches (FDR q < 0.05)

---

## Installation

### Conda (recommended)

```bash
git clone https://github.com/KeranSun/PoincareMDS.git
cd PoincareMDS
conda env create -f environment.yml
conda activate poincare-mds
pip install -e .
```

### Pip only

```bash
pip install -r requirements.txt
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | >= 1.9 | Automatic differentiation |
| geoopt | >= 0.5.0 | Riemannian optimization on Poincaré ball |
| scikit-learn | >= 0.24 | k-NN graph, metrics |
| scipy | >= 1.5 | Shortest paths, sparse graphs |
| scanpy | >= 1.8 | Spatial data I/O |
| anndata | >= 0.8 | Annotated data matrices |
| matplotlib | >= 3.3 | Visualization |
| seaborn | >= 0.11 | Statistical plots |

---

## Quick Start

```python
import numpy as np
from poincare_mds import PoincareMDS, HyperbolicNiche

# Generate synthetic hierarchical data (3 layers)
rng = np.random.default_rng(42)
X = np.vstack([
    rng.normal(loc=[4, 0, 0, 0, 0], scale=0.3, size=(100, 5)),  # Layer 1
    rng.normal(loc=[0, 4, 0, 0, 0], scale=0.3, size=(100, 5)),  # Layer 2
    rng.normal(loc=[0, 0, 4, 0, 0], scale=0.3, size=(100, 5)),  # Layer 3
])

# Embed into Poincaré disk
model = PoincareMDS(curvature=0.5, n_epochs=2000)
coords = model.fit_transform(X)         # shape (300, 2)
norms = np.linalg.norm(coords, axis=1)  # radial distance = hierarchy depth

# Hyperbolic niche analysis
niche = HyperbolicNiche(curvature=0.5, percentile=10)
dist_matrix = niche.compute_distances(coords)
purity, labels = niche.compute_niche_purity(coords, layer_labels)
```

---

## API Reference

### `PoincareMDS`

```python
PoincareMDS(
    curvature=0.5,         # Poincaré ball curvature
    n_components=2,        # embedding dimension (2 for disk)
    n_epochs=2000,         # optimization epochs
    learning_rate=0.05,    # Riemannian Adam LR
    batch_size=100000,     # pairs per mini-batch
    k_neighbors=30,        # k-NN graph neighbors
    target_radius=0.4,     # mean radius target (prevents collapse)
    repulsion_weight=0.5,  # repulsion loss weight
    random_state=42,       # random seed
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `fit_transform(X)` | `ndarray (n, 2)` | Embed matrix X into Poincaré disk |
| `get_norms()` | `ndarray (n,)` | Radial distances from center |
| `loss_history` | `list[float]` | MDS loss per epoch (after fitting) |

### `HyperbolicNiche`

```python
HyperbolicNiche(
    curvature=0.5,     # must match embedding curvature
    percentile=10,     # distance percentile for niche radius
    min_niche_size=3,  # minimum spots per niche
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `compute_distances(embedding)` | `ndarray (n, n)` | Pairwise geodesic distance matrix |
| `compute_niche_purity(embedding, labels)` | `(float, ndarray)` | Mean purity + per-spot niche labels |
| `compute_lr_coexpression(embedding, lr_pairs, expr_matrix)` | `DataFrame` | L-R co-expression within niches |

### `poincare_mds.metrics`

| Function | Description |
|----------|-------------|
| `radius_layer_correlation(embedding, labels)` | Spearman ρ between radius and layer |
| `embedding_stress(embedding, dist_matrix)` | Normalized MDS stress |
| `neighborhood_preservation(X, embedding, k)` | Trustworthiness + continuity |

---

## Data Access

This repository does **not** include raw spatial transcriptomics data. Download from:

| Dataset | Source | Link |
|---------|--------|------|
| Slide-seq V2 (olfactory bulb) | NIH Brain Initiative | [GEO: GSE251950](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251950) |
| Slide-seq V2 (cerebellum) | NIH Brain Initiative | [GEO: GSE251950](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251950) |

Place downloaded files in `spatial_data/` before running analysis scripts.

---

## Reproducing Figures

### Main Figures

```bash
cd scripts
python figure1_overview.py        # Figure 1: Poincaré disk overview
python figure2_theoretical.py     # Figure 2: Synthetic validation
python figure3_biological.py      # Figure 3: Hyperbolic niche analysis
python figure4_validation.py      # Figure 4: Multi-sample validation
python figure5_benchmark.py       # Figure 5: Method benchmark
```

### Supplementary Figures

```bash
python figure_supplementary.py    # Figures S1-S6
python sensitivity_analysis.py    # Generates data for S3-S5
```

All figures are saved to `figures/` as both PDF and PNG.

---

## Reproducing Results

### Core Benchmarks

```bash
# Full benchmark (all methods, all datasets)
python enhanced_benchmark.py

# Bootstrap confidence intervals (50 repeats)
python benchmark_bootstrap.py

# Sensitivity analysis (k, curvature, lambda sweeps)
python sensitivity_analysis.py
```

### Biological Discovery

```bash
# Hyperbolic niche purity analysis
python hyperbolic_niche_analysis.py

# Ligand-receptor co-expression in niches
python lr_coexpression_niche.py

# Developmental trajectory validation
python developmental_validation.py
```

Results are saved to `results/` as CSV files.

---

## Project Structure

```
PoincareMDS/
├── poincare_mds/                 Core Python package
│   ├── __init__.py               Exports: PoincareMDS, HyperbolicNiche, metrics
│   ├── core.py                   Poincaré MDS embedding algorithm
│   ├── niche.py                  Hyperbolic Niche analysis
│   └── metrics.py                Evaluation metrics
│
├── scripts/                      Analysis and figure scripts
│   ├── figure_style.py           Nature Methods visual standards
│   ├── figure1_overview.py       Figure 1: Poincaré disk overview
│   ├── figure2_theoretical.py    Figure 2: Synthetic validation
│   ├── figure3_biological.py     Figure 3: Niche analysis
│   ├── figure4_validation.py     Figure 4: Multi-sample validation
│   ├── figure5_benchmark.py      Figure 5: Method benchmark
│   ├── figure_supplementary.py   Supplementary figures S1-S6
│   ├── enhanced_benchmark.py     Full method comparison
│   ├── benchmark_bootstrap.py    Bootstrap CI computation
│   ├── sensitivity_analysis.py   Hyperparameter sweeps
│   ├── hyperbolic_niche_analysis.py  Niche purity analysis
│   ├── lr_coexpression_niche.py  L-R co-expression detection
│   ├── developmental_validation.py   Trajectory validation
│   ├── stats_utils.py            Bootstrap CI, effect sizes, FDR
│   └── ...                       Additional analysis scripts
│
├── results/                      CSV outputs from analyses
├── figures/                      Generated figures (PDF + PNG)
├── spatial_data/                 Raw data (not in repo, see Data Access)
│
├── poincare_mds/                 Installable package
├── setup.py                      Package configuration
├── environment.yml               Conda environment
├── requirements.txt              Pip dependencies
├── LICENSE                       MIT License
└── README.md                     This file
```

---

## Algorithm Overview

Poincaré MDS performs three steps:

1. **Graph construction**: Build a k-nearest-neighbor graph from PCA-reduced expression data. Compute all-pairs shortest-path distances (Dijkstra).

2. **Hyperbolic embedding**: Optimize a stress function in the Poincaré ball using Riemannian Adam (geoopt). A repulsion loss prevents embedding collapse toward the origin.

3. **Niche analysis**: Define neighborhoods by geodesic distance at a given percentile threshold. Compare niche purity against Euclidean baselines. Detect ligand-receptor co-expression within niches.

The Poincaré ball has constant negative curvature c. Points near the center are close in Euclidean terms; points near the boundary are exponentially far apart. This geometry naturally encodes hierarchical structure: **central nodes = high-level layers, peripheral nodes = low-level layers**.

---

## Citation

```bibtex
@article{sun2026poincare,
  title={Poincar\'{e} MDS: Hyperbolic Embedding Reveals Hierarchical
         Organization in Spatial Transcriptomics},
  author={Sun, Keran and Jia, Keqi and Li, Hongru and Wei, Lin
          and Ma, Cuiqing and Yin, Fei},
  journal={Nature Methods},
  year={2026}
}
```

---

## License

[MIT License](LICENSE) -- Copyright (c) 2024 Keran Sun

---

## Contact

- **Keran Sun** -- keransun@hebmu.edu.cn
- **Fei Yin** (corresponding author) -- yinfei@hebmu.edu.cn

Department of Oral Pathology, School of Stomatology, Hebei Medical University
