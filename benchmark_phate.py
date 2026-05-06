"""
PHATE Benchmark: Poincaré MDS vs PHATE vs t-SNE vs Euclidean MDS
"""

import numpy as np
import scanpy as sc
import torch
import geoopt
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 加载数据
# ============================================================
print("=" * 60)
print("PHATE Benchmark")
print("=" * 60)

adata = sc.read_h5ad('E:/双曲空间模型/spatial_data/21_00731_processed.h5ad')
pca_coords = adata.obsm['X_pca'][:, :10]
spatial = adata.obsm['spatial']
n = len(adata)
print(f"Spots: {n}")

# ============================================================
# 1. Poincaré MDS (我们的方法)
# ============================================================
print("\n1. Poincaré MDS...")

# 用已有的v3嵌入
adata_v3 = sc.read_h5ad('E:/双曲空间模型/spatial_data/21_00731_hyperbolic_v3.h5ad')
emb_poincare = adata_v3.obsm['X_poincare']
print(f"  Loaded: {emb_poincare.shape}")

# ============================================================
# 2. PHATE
# ============================================================
print("\n2. PHATE...")
import phate

phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=1)
emb_phate = phate_op.fit_transform(pca_coords)
print(f"  Done: {emb_phate.shape}")

# ============================================================
# 3. Euclidean MDS
# ============================================================
print("\n3. Euclidean MDS...")
D_euclidean = squareform(pdist(pca_coords))
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
emb_mds = mds.fit_transform(D_euclidean)
print(f"  Done: {emb_mds.shape}")

# ============================================================
# 4. t-SNE
# ============================================================
print("\n4. t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_tsne = tsne.fit_transform(pca_coords)
print(f"  Done: {emb_tsne.shape}")

# ============================================================
# 评估
# ============================================================
print("\n" + "=" * 60)
print("评估指标")
print("=" * 60)

# 距离相关性
np.random.seed(42)
n_eval = 800
eval_idx = np.random.choice(n, n_eval, replace=False)

# 双曲距离
ball = geoopt.PoincareBall(c=0.5)
hyp_t = torch.FloatTensor(emb_poincare[eval_idx])
d_hyp = np.zeros((n_eval, n_eval))
for i in range(n_eval):
    for j in range(i+1, n_eval):
        d = ball.dist(hyp_t[i:i+1], hyp_t[j:j+1])
        d_hyp[i,j] = d.item()
        d_hyp[j,i] = d.item()

D_pca_eval = cdist(pca_coords[eval_idx], pca_coords[eval_idx])
D_spatial_eval = cdist(spatial[eval_idx], spatial[eval_idx])
D_phate_eval = squareform(pdist(emb_phate[eval_idx]))
D_mds_eval = squareform(pdist(emb_mds[eval_idx]))
D_tsne_eval = squareform(pdist(emb_tsne[eval_idx]))

mask = np.triu(np.ones((n_eval, n_eval), dtype=bool), k=1)

# k-NN保留率
k_nn = 15
nn_pca = NearestNeighbors(n_neighbors=k_nn+1).fit(pca_coords)
_, idx_pca = nn_pca.kneighbors(pca_coords)

def knn_retention(emb, k):
    nn = NearestNeighbors(n_neighbors=k+1).fit(emb)
    _, idx = nn.kneighbors(emb)
    return np.mean([
        len(set(idx_pca[i, 1:k+1]) & set(idx[i, 1:k+1])) / k
        for i in range(n)
    ])

# 半径-深度相关性 (用cluster作为proxy)
km = KMeans(n_clusters=8, random_state=42)
labels_km = km.fit_predict(pca_coords[:, :10])

def radius_depth_corr(emb):
    norms = np.linalg.norm(emb, axis=1)
    # 用PCA的cluster centroid距离作为"深度"
    centroid = pca_coords[:, :10].mean(axis=0)
    depths = np.linalg.norm(pca_coords[:, :10] - centroid, axis=1)
    r, p = spearmanr(norms, depths)
    return r

# Niche纯度
def niche_purity_score(emb, labels, c=0.5):
    n = len(emb)
    if hasattr(geoopt.PoincareBall(c=c), 'dist'):
        ball_local = geoopt.PoincareBall(c=c)
        tensor = torch.FloatTensor(emb)
        D = np.zeros((n, n))
        chunk = 300
        for i in range(0, n, chunk):
            ei = min(i + chunk, n)
            for j in range(0, n, chunk):
                ej = min(j + chunk, n)
                d = ball_local.dist(tensor[i:ei].unsqueeze(1), tensor[j:ej].unsqueeze(0))
                D[i:ei, j:ej] = d.detach().numpy()
    else:
        D = squareform(pdist(emb))

    radius = np.percentile(D[D > 0], 10)
    purities = []
    for i in range(0, n, 10):
        niche = np.where(D[i] < radius)[0]
        if len(niche) >= 3:
            niche_labels = labels[niche]
            purities.append(max(np.bincount(niche_labels)) / len(niche_labels))
    return np.mean(purities) if purities else 0

# 计算所有指标
results = []

for name, emb, is_hyp in [('Poincaré MDS', emb_poincare, True),
                            ('PHATE', emb_phate, False),
                            ('Euclidean MDS', emb_mds, False),
                            ('t-SNE', emb_tsne, False)]:
    print(f"\n{name}:")

    # 距离相关性
    if is_hyp:
        D_emb_eval = d_hyp
    elif name == 'PHATE':
        D_emb_eval = D_phate_eval
    elif name == 'Euclidean MDS':
        D_emb_eval = D_mds_eval
    else:
        D_emb_eval = D_tsne_eval

    r_pca, _ = pearsonr(D_pca_eval[mask], D_emb_eval[mask])
    r_spatial, _ = pearsonr(D_spatial_eval[mask], D_emb_eval[mask])
    print(f"  vs PCA distance: r={r_pca:.3f}")
    print(f"  vs Spatial distance: r={r_spatial:.3f}")

    # k-NN保留率
    knn = knn_retention(emb, k_nn)
    print(f"  k-NN retention: {knn:.3f}")

    # 半径-深度相关性
    r_depth = radius_depth_corr(emb)
    print(f"  Radius-depth correlation: {r_depth:.3f}")

    # 嵌入展开度
    norms = np.linalg.norm(emb, axis=1)
    print(f"  Norm: mean={norms.mean():.3f} std={norms.std():.3f} max={norms.max():.3f}")

    # Niche纯度 (只对Poincaré和空间距离)
    if is_hyp:
        purity = niche_purity_score(emb, labels_km, c=0.5)
    else:
        purity = niche_purity_score(emb, labels_km, c=0)  # Euclidean
    print(f"  Niche purity: {purity:.3f}")

    results.append({
        'method': name,
        'r_pca': r_pca,
        'r_spatial': r_spatial,
        'knn_retention': knn,
        'radius_depth': r_depth,
        'norm_std': norms.std(),
        'niche_purity': purity,
    })

df = pd.DataFrame(results)

# ============================================================
# 可视化
# ============================================================
print("\n" + "=" * 60)
print("可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: Poincaré MDS
ax = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
ax.scatter(emb_poincare[:, 0], emb_poincare[:, 1], c='steelblue', s=5, alpha=0.5)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_title('A. Poincaré MDS (Ours)', fontweight='bold')

# Panel B: PHATE
ax = axes[0, 1]
ax.scatter(emb_phate[:, 0], emb_phate[:, 1], c='steelblue', s=5, alpha=0.5)
ax.set_title('B. PHATE', fontweight='bold')

# Panel C: Euclidean MDS
ax = axes[0, 2]
ax.scatter(emb_mds[:, 0], emb_mds[:, 1], c='steelblue', s=5, alpha=0.5)
ax.set_title('C. Euclidean MDS', fontweight='bold')

# Panel D: t-SNE
ax = axes[1, 0]
ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c='steelblue', s=5, alpha=0.5)
ax.set_title('D. t-SNE', fontweight='bold')

# Panel E: 指标对比
ax = axes[1, 1]
metrics = ['r_pca', 'r_spatial', 'knn_retention', 'radius_depth', 'niche_purity']
metric_labels = ['PCA\nCorr', 'Spatial\nCorr', 'k-NN\nRetention', 'Depth\nCorr', 'Niche\nPurity']
x = np.arange(len(metrics))
width = 0.2
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
for i, row in df.iterrows():
    vals = [row[m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=row['method'], color=colors[i])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylabel('Score')
ax.set_title('E. Method Comparison', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')

# Panel F: 总结表
ax = axes[1, 2]
ax.axis('off')
table_data = [['Metric', 'Poincaré', 'PHATE', 'MDS', 't-SNE']]
for metric, label in [('r_pca', 'PCA Corr'), ('r_spatial', 'Spatial Corr'),
                       ('knn_retention', 'k-NN'), ('radius_depth', 'Depth Corr'),
                       ('niche_purity', 'Niche Purity')]:
    row = [label] + [f'{r[metric]:.3f}' for _, r in df.iterrows()]
    table_data.append(row)

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
for j in range(5):
    table[0, j].set_facecolor('#1976D2')
    table[0, j].set_text_props(color='white', fontweight='bold')
# 高亮Poincaré列
for i in range(1, 6):
    table[i, 1].set_facecolor('#E3F2FD')
ax.set_title('F. Summary Table', fontweight='bold')

plt.tight_layout()
plt.savefig('E:/双曲空间模型/Figure_phate_benchmark.png', dpi=150, bbox_inches='tight')
print('Saved: Figure_phate_benchmark.png')

df.to_csv('E:/双曲空间模型/phate_benchmark.csv', index=False)
print('Saved: phate_benchmark.csv')

# 更新Results.md
print("\n" + "=" * 60)
print("PHATE Benchmark 结果:")
print("=" * 60)
print(df.to_string(index=False))
