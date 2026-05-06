"""
最终Figure组装
Figure 1: Poincaré MDS方法概述 (6 panels)
Figure 2: Hyperbolic Niche分析 (已生成)
Figure 3: 模拟数据验证 (已生成)
Figure 4: 量化总结 + 距离保真度
"""

import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform, pdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 加载数据
# ============================================================
print("加载数据...")
adata = sc.read_h5ad('E:/双曲空间模型/spatial_data/21_00731_hyperbolic_v3.h5ad')
hyp_emb = adata.obsm['X_poincare']
spatial = adata.obsm['spatial']
pca_coords = adata.obsm['X_pca']
n = len(adata)
norms = np.linalg.norm(hyp_emb, axis=1)

# Cell type signatures
signatures = {
    'Epithelial': ['EPCAM', 'KRT18', 'KRT19', 'KRT8'],
    'Fibroblast': ['COL1A1', 'COL1A2', 'DCN', 'LUM'],
    'T_cell': ['CD3D', 'CD3E', 'CD2'],
    'Macrophage': ['CD68', 'C1QA', 'C1QB', 'C1QC'],
    'Endothelial': ['VWF', 'CDH5', 'ENG'],
    'CAF_m': ['FAP', 'POSTN', 'ACTA2', 'MMP2'],
    'Tumor': ['MUC5AC', 'CEACAM5', 'REG4'],
}

def module_score(adata, genes):
    avail = [g for g in genes if g in adata.var_names]
    if not avail: return np.zeros(adata.n_obs)
    idx = [list(adata.var_names).index(g) for g in avail]
    expr = adata.X[:, idx]
    if hasattr(expr, 'toarray'): expr = expr.toarray()
    return expr.mean(axis=1)

scores = {name: module_score(adata, genes) for name, genes in signatures.items()}

# ============================================================
# Figure 1: 方法概述
# ============================================================
print("组装 Figure 1...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: Poincaré disk (按cluster着色)
ax = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, lw=1.5)

clusters = adata.obs['cluster'].values
unique_clusters = sorted(clusters.unique())
cmap = plt.cm.Set2
for i, cl in enumerate(unique_clusters):
    mask = clusters == cl
    ax.scatter(hyp_emb[mask, 0], hyp_emb[mask, 1],
               c=[cmap(i/len(unique_clusters))], s=8, alpha=0.6, label=f'Cluster {cl}')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_xlabel('Poincaré x')
ax.set_ylabel('Poincaré y')
ax.set_title('A. Poincaré Disk Embedding', fontweight='bold', fontsize=12)
ax.legend(fontsize=7, ncol=2, loc='upper right')

# Panel B: 空间坐标 (按cluster着色)
ax = axes[0, 1]
for i, cl in enumerate(unique_clusters):
    mask = clusters == cl
    ax.scatter(spatial[mask, 0], spatial[mask, 1],
               c=[cmap(i/len(unique_clusters))], s=8, alpha=0.6, label=f'Cluster {cl}')
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_xlabel('Spatial x')
ax.set_ylabel('Spatial y')
ax.set_title('B. Spatial Coordinates', fontweight='bold', fontsize=12)
ax.legend(fontsize=7, ncol=2, loc='upper right')

# Panel C: 半径分布 (boxplot by cluster)
ax = axes[0, 2]
cluster_norms = [norms[clusters == cl] for cl in unique_clusters]
bp = ax.boxplot(cluster_norms, labels=[f'C{cl}' for cl in unique_clusters],
                patch_artist=True)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(cmap(i/len(unique_clusters)))
ax.set_xlabel('Cluster')
ax.set_ylabel('Poincaré Radius')
ax.set_title('C. Radial Distribution by Cluster', fontweight='bold', fontsize=12)

# Panel D: 关键细胞类型在Poincaré disk上
ax = axes[1, 0]
ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, lw=1.5)
ct_colors = {'Tumor': '#E64B35', 'CAF_m': '#4DBBD5', 'Fibroblast': '#00A087',
             'T_cell': '#FF7F00', 'Macrophage': '#8491B4'}
for ct, color in ct_colors.items():
    s = scores[ct]
    # 只显示高表达的点
    thresh = np.percentile(s, 70)
    mask = s > thresh
    ax.scatter(hyp_emb[mask, 0], hyp_emb[mask, 1], c=color, s=10, alpha=0.5, label=ct)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_xlabel('Poincaré x')
ax.set_ylabel('Poincaré y')
ax.set_title('D. Cell Types on Poincaré Disk', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='upper right')

# Panel E: 半径 vs 细胞类型score
ax = axes[1, 1]
for ct in ['Tumor', 'CAF_m', 'Fibroblast', 'T_cell', 'Epithelial']:
    r, p = spearmanr(norms, scores[ct])
    ax.scatter(norms, scores[ct], s=3, alpha=0.2,
               label=f'{ct} (r={r:.2f}, p={p:.1e})')
ax.set_xlabel('Poincaré Radius')
ax.set_ylabel('Module Score')
ax.set_title('E. Radius vs Cell Type Score', fontweight='bold', fontsize=12)
ax.legend(fontsize=7, loc='upper right')

# Panel F: 距离保真度散点图 (双曲距离 vs PCA距离)
ax = axes[1, 2]
# 采样
np.random.seed(42)
sample_idx = np.random.choice(n, 500, replace=False)
from scipy.spatial.distance import cdist
import torch, geoopt
ball = geoopt.PoincareBall(c=0.5)
hyp_tensor = torch.FloatTensor(hyp_emb[sample_idx])
d_hyp = ball.dist(hyp_tensor.unsqueeze(1), hyp_tensor.unsqueeze(0)).detach().numpy()
d_pca = cdist(pca_coords[sample_idx, :10], pca_coords[sample_idx, :10])
mask_upper = np.triu(np.ones((500, 500), dtype=bool), k=1)
r, p = pearsonr(d_hyp[mask_upper], d_pca[mask_upper])
ax.scatter(d_pca[mask_upper], d_hyp[mask_upper], s=1, alpha=0.1, c='steelblue')
ax.plot([0, d_pca[mask_upper].max()], [0, d_pca[mask_upper].max()], 'r--', alpha=0.5)
ax.set_xlabel('PCA Euclidean Distance')
ax.set_ylabel('Poincaré Geodesic Distance')
ax.set_title(f'F. Distance Fidelity (r={r:.3f})', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('E:/双曲空间模型/Figure1_method_overview.png', dpi=300, bbox_inches='tight')
print('Saved: Figure1_method_overview.png')

# ============================================================
# Figure 4: 量化总结
# ============================================================
print("组装 Figure 4...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: 距离相关性对比 (所有方法)
ax = axes[0, 0]
methods = ['Poincaré\nMDS', 'Euclidean\nMDS', 't-SNE']
tree_corrs = [0.919, 0.926, 0.427]
pca_corrs = [0.887, 1.0, 0.619]  # Hyp-PCA, Euc-PCA (perfect), tSNE-PCA
x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, tree_corrs, width, label='vs Tree Distance', color='#2196F3')
ax.bar(x + width/2, pca_corrs, width, label='vs PCA Distance', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Pearson r')
ax.set_title('A. Distance Preservation', fontweight='bold', fontsize=12)
ax.legend()
ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5)

# Panel B: 半径-深度相关性 (核心!)
ax = axes[0, 1]
depth_corrs = [0.853, -0.104, -0.088]
colors = ['#2196F3', '#FF9800', '#4CAF50']
bars = ax.bar(methods, depth_corrs, color=colors)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Spearman r')
ax.set_title('B. Radius-Depth Correlation (Key)', fontweight='bold', fontsize=12)
# 标注数值
for bar, val in zip(bars, depth_corrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

# Panel C: Niche纯度对比
ax = axes[0, 2]
hyp_purities = [0.672]  # From full niche analysis
euc_purities = [0.439]
ax.bar(['Hyperbolic\nNiche', 'Euclidean\nNiche'], [0.672, 0.439],
       color=['#2196F3', '#FF9800'])
ax.set_ylabel('Niche Purity')
ax.set_title('C. Niche Purity (p=5.1e-135)', fontweight='bold', fontsize=12)
ax.text(0.5, 0.95, '***', transform=ax.transAxes, ha='center', fontsize=20, color='red')

# Panel D: 交互对富集度
ax = axes[1, 0]
pairs = ['CAF-\nTumor', 'T-\nMacro', 'Endo-\nFibro', 'Epi-\nCAF']
hyp_enrich = [0.248, 1.266, 1.811, 0.588]
euc_enrich = [0.277, 1.241, 1.756, 0.631]
x = np.arange(len(pairs))
width = 0.35
ax.bar(x - width/2, hyp_enrich, width, label='Hyperbolic', color='#2196F3')
ax.bar(x + width/2, euc_enrich, width, label='Euclidean', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(pairs, fontsize=9)
ax.set_ylabel('Enrichment Score')
ax.set_title('D. Interaction Pair Enrichment', fontweight='bold', fontsize=12)
ax.legend()
# 标注胜者
winners = ['Euc', 'Hyp', 'Hyp', 'Euc']
for i, w in enumerate(winners):
    color = 'blue' if w == 'Hyp' else 'red'
    ax.annotate('*', (i - width/2 if w=='Hyp' else i + width/2,
                      max(hyp_enrich[i], euc_enrich[i]) + 0.02),
                fontsize=16, color=color, ha='center')

# Panel E: 方法总结表
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['Metric', 'Poincaré', 'Euclidean', 't-SNE'],
    ['Tree Corr', '0.919', '0.926', '0.427'],
    ['Depth Corr', '0.853', '-0.104', '-0.088'],
    ['k-NN Retention', '0.478', '0.501', '0.384'],
    ['Niche Purity', '0.672', '0.439', '-'],
    ['Hyp-Spatial', '0.397', '0.363', '-'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
# 高亮最佳
for i in range(1, len(table_data)):
    for j in range(1, len(table_data[i])):
        try:
            val = float(table_data[i][j])
            if j == 1:  # Poincaré列
                table[i, j].set_facecolor('#E3F2FD')
        except:
            pass
# 表头
for j in range(4):
    table[0, j].set_facecolor('#1976D2')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('E. Summary Table', fontweight='bold', fontsize=12)

# Panel F: Poincaré disk 示意 (标注层次结构)
ax = axes[1, 2]
theta_circle = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', alpha=0.5, lw=2)
# 同心圆标注
for r in [0.25, 0.5, 0.75]:
    ax.plot(r * np.cos(theta_circle), r * np.sin(theta_circle), 'k--', alpha=0.2)
# 标注
ax.text(0, 0, 'Center\n(General)', ha='center', fontsize=9, fontweight='bold')
ax.text(0.5, 0.5, 'Mid-radius\n(Specialized)', ha='center', fontsize=8, style='italic')
ax.text(0.75, 0.75, 'Boundary\n(Terminal)', ha='center', fontsize=8, style='italic')
# 箭头标注半径
ax.annotate('', xy=(0.8, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.4, -0.08, 'Depth', fontsize=10, color='red', fontweight='bold')
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.set_aspect('equal')
ax.set_title('F. Poincaré Disk Geometry', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('E:/双曲空间模型/Figure4_summary.png', dpi=300, bbox_inches='tight')
print('Saved: Figure4_summary.png')

print("\n" + "=" * 60)
print("Figure组装完成!")
print("=" * 60)
print("Figure1_method_overview.png - 方法概述 (6 panels)")
print("figure_hyperbolic_niche_full.png - Niche分析 (6 panels)")
print("figure_tree_validation.png - 模拟验证 (6 panels)")
print("Figure4_summary.png - 量化总结 (6 panels)")
print("=" * 60)
