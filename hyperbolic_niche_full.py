"""
CP3 Full: Hyperbolic Niche Analysis — 全量4252 spots
核心创新：用Poincaré disk上的测地线距离定义细胞邻域(niche)
与欧氏距离niche对比，验证双曲niche是否更好地捕捉细胞互作
"""

import numpy as np
import scanpy as sc
import torch
import geoopt
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, spearmanr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Step 0: 加载数据
# ============================================================
print("=" * 60)
print("Hyperbolic Niche Analysis (Full Dataset)")
print("=" * 60)

adata = sc.read_h5ad('E:/双曲空间模型/spatial_data/21_00731_hyperbolic_v3.h5ad')
hyp_emb = adata.obsm['X_poincare']
spatial = adata.obsm['spatial']
n = len(adata)

print(f"Spots: {n}")
print(f"Poincaré embedding shape: {hyp_emb.shape}")

# 计算cell type module scores
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
# Step 1: 计算距离矩阵
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 计算距离矩阵")
print("=" * 60)

# 欧氏距离
D_euclidean = cdist(spatial, spatial)
print(f"空间欧氏距离: {D_euclidean.min():.1f} - {D_euclidean.max():.1f}")

# 双曲距离 (分块)
ball = geoopt.PoincareBall(c=0.5)
hyp_tensor = torch.FloatTensor(hyp_emb)

print("计算双曲距离矩阵...")
D_hyperbolic = np.zeros((n, n))
chunk_size = 500
for i in range(0, n, chunk_size):
    end_i = min(i + chunk_size, n)
    for j in range(0, n, chunk_size):
        end_j = min(j + chunk_size, n)
        d = ball.dist(hyp_tensor[i:end_i].unsqueeze(1),
                      hyp_tensor[j:end_j].unsqueeze(0))
        D_hyperbolic[i:end_i, j:end_j] = d.detach().numpy()

print(f"双曲距离: {D_hyperbolic.min():.4f} - {D_hyperbolic.max():.4f}")

# PCA距离
pca_coords = adata.obsm['X_pca']
D_pca = cdist(pca_coords[:, :10], pca_coords[:, :10])

# ============================================================
# Step 2: 定义Niche函数
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 定义Niche")
print("=" * 60)

def get_niche(D_matrix, center_idx, radius):
    distances = D_matrix[center_idx]
    return np.where(distances < radius)[0]

def niche_composition(scores, niche_indices):
    return {ct: np.mean(score[niche_indices]) for ct, score in scores.items()}

def niche_purity(labels, niche_indices):
    niche_labels = labels[niche_indices]
    if len(niche_labels) == 0:
        return 0
    return max(np.bincount(niche_labels.astype(int))) / len(niche_labels)

# KMeans聚类作为ground truth
from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, random_state=42)
labels_km = km.fit_predict(pca_coords[:, :10])

# ============================================================
# Step 3: 双曲Niche vs 欧氏Niche 对比
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 双曲Niche vs 欧氏Niche 对比")
print("=" * 60)

interaction_pairs = [
    ('CAF_m', 'Tumor', 'CAF-Tumor'),
    ('T_cell', 'Macrophage', 'T-Macro'),
    ('Endothelial', 'Fibroblast', 'Endo-Fibro'),
    ('Epithelial', 'CAF_m', 'Epi-CAF'),
]

# 使用分位数定义半径
hyp_radius = np.percentile(D_hyperbolic[D_hyperbolic > 0], 10)
euc_radius = np.percentile(D_euclidean[D_euclidean > 0], 10)

print(f"双曲niche半径 (10th pct): {hyp_radius:.4f}")
print(f"欧氏niche半径 (10th pct): {euc_radius:.1f}")

# 对每个spot计算niche组成
results = []
for spot_idx in range(n):
    hyp_niche = get_niche(D_hyperbolic, spot_idx, hyp_radius)
    euc_niche = get_niche(D_euclidean, spot_idx, euc_radius)

    if len(hyp_niche) < 3 or len(euc_niche) < 3:
        continue

    center_scores = {ct: score[spot_idx] for ct, score in scores.items()}
    center_type = max(center_scores, key=center_scores.get)

    hyp_comp = niche_composition(scores, hyp_niche)
    euc_comp = niche_composition(scores, euc_niche)

    for ct1, ct2, pair_name in interaction_pairs:
        if center_scores[ct1] > np.mean(scores[ct1]):
            results.append({
                'spot': spot_idx,
                'center_type': center_type,
                'pair': pair_name,
                'ct1_score': center_scores[ct1],
                'hyp_ct2_enrichment': hyp_comp[ct2],
                'euc_ct2_enrichment': euc_comp[ct2],
                'hyp_niche_size': len(hyp_niche),
                'euc_niche_size': len(euc_niche),
            })

df = pd.DataFrame(results)
print(f"\n有效分析spots: {len(df)}")
print(f"互作对数量:")
print(df['pair'].value_counts())

# ============================================================
# Step 4: 统计检验
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 统计检验")
print("=" * 60)

comparison_results = []
for pair_name in df['pair'].unique():
    pair_df = df[df['pair'] == pair_name]

    hyp_vals = pair_df['hyp_ct2_enrichment'].values
    euc_vals = pair_df['euc_ct2_enrichment'].values

    stat, p_val = mannwhitneyu(hyp_vals, euc_vals, alternative='two-sided')

    hyp_mean = np.mean(hyp_vals)
    euc_mean = np.mean(euc_vals)
    effect = (hyp_mean - euc_mean) / (np.std(hyp_vals) + 1e-8)

    comparison_results.append({
        'pair': pair_name,
        'hyp_mean': hyp_mean,
        'euc_mean': euc_mean,
        'effect_size': effect,
        'p_value': p_val,
        'winner': 'Hyperbolic' if hyp_mean > euc_mean else 'Euclidean'
    })

    print(f"\n{pair_name}:")
    print(f"  双曲niche富集度: {hyp_mean:.4f}")
    print(f"  欧氏niche富集度: {euc_mean:.4f}")
    print(f"  效应量: {effect:.3f}")
    print(f"  p-value: {p_val:.2e}")
    print(f"  胜者: {'Hyperbolic' if hyp_mean > euc_mean else 'Euclidean'}")

comp_df = pd.DataFrame(comparison_results)
print(f"\n总结: 双曲niche在 {sum(comp_df['winner']=='Hyperbolic')}/{len(comp_df)} 个互作对上优于欧氏niche")

# ============================================================
# Step 5: Niche纯度对比
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Niche纯度对比")
print("=" * 60)

hyp_purities = []
euc_purities = []

for spot_idx in range(0, n, 5):  # 每5个spot采样
    hyp_niche = get_niche(D_hyperbolic, spot_idx, hyp_radius)
    euc_niche = get_niche(D_euclidean, spot_idx, euc_radius)

    if len(hyp_niche) >= 3:
        hyp_purities.append(niche_purity(labels_km, hyp_niche))
    if len(euc_niche) >= 3:
        euc_purities.append(niche_purity(labels_km, euc_niche))

print(f"双曲niche纯度: {np.mean(hyp_purities):.3f} +/- {np.std(hyp_purities):.3f}")
print(f"欧氏niche纯度: {np.mean(euc_purities):.3f} +/- {np.std(euc_purities):.3f}")
stat, p_val = mannwhitneyu(hyp_purities, euc_purities)
print(f"p-value: {p_val:.2e}")

# ============================================================
# Step 6: 可视化 (6 panels)
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: Poincaré disk + niche示例
ax = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, lw=1)
ax.scatter(hyp_emb[:, 0], hyp_emb[:, 1], c='lightgrey', s=3, alpha=0.4)

center = 100
hyp_niche_idx = get_niche(D_hyperbolic, center, hyp_radius)
ax.scatter(hyp_emb[center, 0], hyp_emb[center, 1], c='red', s=100, zorder=5, marker='*')
ax.scatter(hyp_emb[hyp_niche_idx, 0], hyp_emb[hyp_niche_idx, 1],
           c='blue', s=15, alpha=0.7, label=f'Hyperbolic niche (n={len(hyp_niche_idx)})')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.legend(fontsize=8)
ax.set_title('A. Hyperbolic Niche Definition', fontweight='bold')

# Panel B: 互作对富集度对比
ax = axes[0, 1]
pair_names = comp_df['pair'].values
x = np.arange(len(pair_names))
width = 0.35
ax.bar(x - width/2, comp_df['hyp_mean'], width, label='Hyperbolic', color='#2196F3')
ax.bar(x + width/2, comp_df['euc_mean'], width, label='Euclidean', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(pair_names, rotation=15, fontsize=9)
ax.set_ylabel('Enrichment Score')
ax.set_title('B. Interaction Pair Enrichment', fontweight='bold')
ax.legend()

for i, row in comp_df.iterrows():
    color = 'green' if row['winner'] == 'Hyperbolic' else 'red'
    ax.annotate('*', (i - width/2 if row['winner']=='Hyperbolic' else i + width/2,
                      max(row['hyp_mean'], row['euc_mean']) + 0.02),
                fontsize=16, color=color, ha='center')

# Panel C: Niche纯度对比
ax = axes[0, 2]
ax.boxplot([hyp_purities, euc_purities], labels=['Hyperbolic', 'Euclidean'])
ax.set_ylabel('Niche Purity')
ax.set_title('C. Niche Purity Comparison', fontweight='bold')
ax.text(0.5, 0.95, f'p={p_val:.2e}', transform=ax.transAxes, ha='center', fontsize=10)

# Panel D: 富集度散点图
ax = axes[1, 0]
for pair_name in df['pair'].unique():
    pair_df = df[df['pair'] == pair_name]
    ax.scatter(pair_df['euc_ct2_enrichment'], pair_df['hyp_ct2_enrichment'],
               s=3, alpha=0.2, label=pair_name)
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
ax.set_xlabel('Euclidean Niche Enrichment')
ax.set_ylabel('Hyperbolic Niche Enrichment')
ax.set_title('D. Hyperbolic vs Euclidean Enrichment', fontweight='bold')
ax.legend(fontsize=7)

# Panel E: 空间映射
ax = axes[1, 1]
ax.scatter(spatial[:, 0], spatial[:, 1], c='lightgrey', s=3, alpha=0.4)
for center_idx, color, label in [(100, 'red', 'Niche 1'),
                                   (1500, 'blue', 'Niche 2'),
                                   (3000, 'green', 'Niche 3')]:
    niche_idx = get_niche(D_hyperbolic, center_idx, hyp_radius)
    ax.scatter(spatial[center_idx, 0], spatial[center_idx, 1], c=color, s=100, zorder=5, marker='*')
    ax.scatter(spatial[niche_idx, 0], spatial[niche_idx, 1], c=color, s=10, alpha=0.5, label=label)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_title('E. Niche Spatial Mapping', fontweight='bold')
ax.legend(fontsize=8)

# Panel F: 半径 vs 细胞类型
ax = axes[1, 2]
norms = np.linalg.norm(hyp_emb, axis=1)
for ct in ['Epithelial', 'Fibroblast', 'CAF_m', 'Tumor']:
    r, p = spearmanr(norms, scores[ct])
    ax.scatter(norms, scores[ct], s=2, alpha=0.2, label=f'{ct} (r={r:.2f})')
ax.set_xlabel('Poincare Radius')
ax.set_ylabel('Module Score')
ax.set_title('F. Radius vs Cell Type Score', fontweight='bold')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('E:/双曲空间模型/figure_hyperbolic_niche_full.png', dpi=150, bbox_inches='tight')
print('Saved: figure_hyperbolic_niche_full.png')

comp_df.to_csv('E:/双曲空间模型/niche_comparison_results_full.csv', index=False)
print('Saved: niche_comparison_results_full.csv')

print("\n" + "=" * 60)
print("CP3 Full 检查清单:")
print("=" * 60)
n_hyperbolic_wins = sum(comp_df['winner'] == 'Hyperbolic')
print(f"[{'x' if n_hyperbolic_wins > 0 else ' '}] 双曲niche在至少1个互作对上优于欧氏")
print(f"[{'x' if np.mean(hyp_purities) > np.mean(euc_purities) else ' '}] 双曲niche纯度更高")
print(f"[{'x' if len(comp_df) >= 3 else ' '}] 至少3个互作对完成分析")
print(f"[{'x' if p_val < 0.05 else ' '}] 纯度差异显著 (p<0.05)")
print("=" * 60)
