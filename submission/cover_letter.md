# Cover Letter

**Manuscript Title**: Poincaré MDS: Hyperbolic Embedding Reveals Hierarchical Organization in Spatial Transcriptomics

**Article Type**: Methods

**Target Journal**: Nature Methods

---

Dear Editors,

We are pleased to submit our manuscript "Poincaré MDS: Hyperbolic Embedding Reveals Hierarchical Organization in Spatial Transcriptomics" for consideration as a Methods article in Nature Methods.

## Summary

Spatial transcriptomics technologies generate high-dimensional data with inherent spatial structure, but existing dimensionality reduction methods (UMAP, t-SNE, PHATE) do not explicitly model the hierarchical organization of biological tissues. We introduce **Poincaré MDS**, a method that embeds spatial transcriptomics data into the Poincaré disk model of hyperbolic space, where the radial coordinate naturally encodes hierarchical depth. We further introduce the **Hyperbolic Niche**, a neighborhood definition based on geodesic distance, for analyzing cell-cell interactions.

## Key Results

1. **Hierarchical encoding**: On synthetic data with known tree structure, Poincaré radius correlates with tree depth (r = 0.853), a property absent in Euclidean MDS (r = −0.104), PHATE, and t-SNE (r = −0.088).

2. **Hyperbolic Niche superiority**: Hyperbolic niches achieve significantly higher purity than Euclidean niches across all 10 tissue sections tested (mean 0.794 vs. 0.693; single-sample p = 5.1 × 10⁻¹³⁵).

3. **Spatial fidelity**: Poincaré MDS achieves higher spatial correlation (r = 0.404) than Euclidean MDS (r = 0.330) and PHATE (r = 0.374).

4. **Reproducibility**: Results are consistent across 10 independent Visium gastric cancer samples.

## Significance for Nature Methods

This work introduces a novel computational framework (hyperbolic geometry) to a new domain (spatial transcriptomics), with a new concept (Hyperbolic Niche) and rigorous validation (synthetic ground truth + multi-sample reproducibility). The method is implemented as an open-source Python package and is computationally tractable (~15 minutes for 4,000 spots on a standard laptop).

## Why Nature Methods

Nature Methods is the ideal venue because:
- The manuscript introduces a **new method** with broad applicability to spatial transcriptomics data
- The method addresses a **fundamental challenge** (hierarchical tissue organization) that existing methods do not solve
- The validation includes **synthetic ground truth** and **multi-sample reproducibility**, meeting the journal's standards for rigor
- The method is **practically useful**: open-source, computationally efficient, and applicable to any spatial transcriptomics platform

## Declarations

- **Conflict of interest**: The authors declare no competing interests.
- **Data availability**: Spatial transcriptomics data is publicly available. Code is available at https://github.com/KeranSun/poincare-mds-spatial (private, will be made public upon publication).
- **Ethics**: No human subjects were directly involved; data was obtained from publicly available datasets.
- **Author contributions**: [To be filled]

We believe this manuscript presents a significant methodological advance that will be of broad interest to the spatial transcriptomics and computational biology communities. We look forward to your consideration.

Sincerely,

[Corresponding Author Name]
[Affiliation]
[Email]

---

*This manuscript has not been submitted elsewhere and is not under consideration by another journal.*
