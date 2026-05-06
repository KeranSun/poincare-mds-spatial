# Nature Methods Submission Checklist

## Pre-submission

- [x] Manuscript formatted (Title, Abstract, Intro, Results, Discussion, Methods, References)
- [x] Cover letter written
- [x] Figures ready (5 figures × 6 panels)
- [x] Code on GitHub (private)
- [x] Data availability statement
- [x] Code availability statement
- [x] Competing interests declaration

## Manuscript Format

- [x] Abstract: ~150 words
- [x] Main text: ~3,500 words (within 3,500-5,000 limit)
- [x] References: 9 (within ~40 limit)
- [x] Figures: 5 display items
- [ ] Supplementary Information (optional)

## Figures

| Figure | Description | Panels | Status |
|--------|-------------|--------|--------|
| Figure 1 | Method overview | 6 | ✅ |
| Figure 2 | Hyperbolic Niche | 6 | ✅ |
| Figure 3 | Ground truth validation | 6 | ✅ |
| Figure 4 | PHATE benchmark | 6 | ✅ |
| Figure 5 | Multi-sample validation | 6 | ✅ |

## Key Numbers (verify consistency)

| Metric | Value | Location |
|--------|-------|----------|
| Niche purity (single) | Hyp 0.672 vs Euc 0.439, p=5.1e-135 | Results, Figure 2 |
| Niche purity (multi) | Hyp 0.794 vs Euc 0.693, 10/10 | Results, Figure 5 |
| Radius-depth (synthetic) | r=0.853 | Results, Figure 3 |
| Spatial correlation | 0.404 (vs PCA 0.330) | Results, Table 1 |
| Hyp-PCA correlation | 0.887 | Results |
| Hyp-Graph correlation | 0.913 | Results |

## Submission Portal

- Journal: https://www.nature.com/nmeth/
- Submit via: https://mts-nature.com/
- Format: Word or LaTeX

## After Submission

- [ ] Share GitHub repo URL with editor
- [ ] Prepare for reviewer responses
- [ ] Prepare supplementary materials if requested

## Notes

- GitHub repo is private; make public upon acceptance
- Token should be revoked after submission
- Consider uploading figures as separate high-res files
