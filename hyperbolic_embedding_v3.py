"""
CP2 v3: Poincaré MDS — 改进版
核心改进：
1. 用k-NN graph distance替代欧氏distance (保留局部结构)
2. 降低曲率c=0.1 (让disk更"平坦"，点能展开)
3. 更大的初始散布
4. 加repulsion loss防止collapse
"""

import torch
import geoopt
import numpy as np
import scanpy as sc
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Step 0: 加载数据
# ============================================================
print("=" * 60)
print("Step 0: 加载数据")
print("=" * 60)

adata = sc.read_h5ad('E:/双曲空间模型/spatial_data/21_00731_processed.h5ad')
print(f"Spots: {adata.n_obs}, Genes: {adata.n_vars}")

pca_coords = adata.obsm['X_pca']
spatial_coords = adata.obsm['spatial']
n = adata.n_obs

# ============================================================
# Step 1: 计算target距离 (k-NN graph distance)
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 计算target距离 (k-NN graph shortest path)")
print("=" * 60)

pca_10d = pca_coords[:, :10]

# 构建k-NN graph
k = 30
nn = NearestNeighbors(n_neighbors=k+1).fit(pca_10d)
distances, indices = nn.kneighbors(pca_10d)

# 构建稀疏邻接矩阵
rows = np.repeat(np.arange(n), k)
cols = indices[:, 1:].flatten()  # 排除自身
weights = distances[:, 1:].flatten()

# 对称化
adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
adj = (adj + adj.T) / 2  # 对称

# 最短路径距离
print("计算最短路径距离...")
D_graph = shortest_path(adj, directed=False, method='D')
D_graph[D_graph == np.inf] = np.nanmax(D_graph[D_graph != np.inf]) * 2
print(f"Graph距离范围: {np.nanmin(D_graph):.2f} - {np.nanmax(D_graph):.2f}")

# 也计算欧氏距离用于对比
D_euclidean = squareform(pdist(pca_10d))

# 归一化到 [0, 0.95]
D_target = D_graph / np.nanmax(D_graph) * 0.95
D_target = np.nan_to_num(D_target, nan=0.95)

# ============================================================
# Step 2: 初始化Poincaré disk嵌入
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 初始化嵌入")
print("=" * 60)

# 用更小的曲率，让disk更平坦
ball = geoopt.PoincareBall(c=0.5)

# Torgerson MDS 初始化
from scipy.linalg import eigh
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ (D_target ** 2) @ H
eigenvalues, eigenvectors = eigh(B)
idx = np.argsort(eigenvalues)[::-1][:2]
L = np.diag(np.sqrt(np.maximum(eigenvalues[idx], 0)))
V = eigenvectors[:, idx]
init_coords = V @ L

# 缩放到更大范围
init_coords = init_coords / (np.abs(init_coords).max() + 1e-8) * 0.8
init_tensor = torch.FloatTensor(init_coords)
init_tensor = ball.projx(init_tensor)

embeddings = geoopt.ManifoldParameter(init_tensor, manifold=ball)
print(f"嵌入shape: {embeddings.shape}")
print(f"初始norm: min={embeddings.data.norm(dim=1).min():.4f} max={embeddings.data.norm(dim=1).max():.4f}")

# ============================================================
# Step 3: Mini-batch MDS + Repulsion 训练
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 训练 (Graph MDS + Repulsion)")
print("=" * 60)

D_target_tensor = torch.FloatTensor(D_target)
D_euc_tensor = torch.FloatTensor(D_euclidean)

batch_size = 100000

def combined_loss(embeddings, D_target, D_euc, ball, batch_size):
    """MDS loss + repulsion loss"""
    n = len(embeddings)

    # MDS loss (mini-batch)
    i_idx = torch.randint(0, n, (batch_size,))
    j_idx = torch.randint(0, n, (batch_size,))
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    d_hyp = ball.dist(embeddings[i_idx], embeddings[j_idx])
    d_tgt = D_target[i_idx, j_idx]
    mds_loss = ((d_hyp - d_tgt) ** 2).mean()

    # Repulsion loss: 防止所有点collapse到中心
    # 鼓励点的平均半径不要太小
    norms = embeddings.norm(dim=1)
    target_radius = 0.4  # 目标平均半径
    radius_loss = torch.relu(target_radius - norms.mean()) ** 2

    return mds_loss + 0.5 * radius_loss

optimizer = geoopt.optim.RiemannianAdam([embeddings], lr=0.05, weight_decay=1e-5)

n_epochs = 2000
losses = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = combined_loss(embeddings, D_target_tensor, D_euc_tensor, ball, batch_size)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        embeddings.data = ball.projx(embeddings.data)

    losses.append(loss.item())

    if (epoch + 1) % 400 == 0:
        norms = embeddings.data.norm(dim=1)
        print(f"  Epoch {epoch+1:4d}/{n_epochs} | Loss: {loss.item():.6f} | "
              f"Norm: min={norms.min():.4f} max={norms.max():.4f} std={norms.std():.4f}")

print(f"\n训练完成! 最终loss: {losses[-1]:.6f}")

# ============================================================
# Step 4: 检查嵌入结果
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 检查嵌入结果")
print("=" * 60)

hyp_emb = embeddings.detach().numpy()
norms = np.linalg.norm(hyp_emb, axis=1)

print(f"嵌入坐标范围:")
print(f"  X: {hyp_emb[:,0].min():.4f} - {hyp_emb[:,0].max():.4f}")
print(f"  Y: {hyp_emb[:,1].min():.4f} - {hyp_emb[:,1].max():.4f}")
print(f"  半径(norm): min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f} std={norms.std():.4f}")

inside_disk = (norms < 1.0).all()
print(f"所有点在disk内: {inside_disk}")

# ============================================================
# Step 5: 评估指标
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 评估指标")
print("=" * 60)

# 采样计算相关性
np.random.seed(42)
n_eval = min(1000, n)
eval_idx = np.random.choice(n, n_eval, replace=False)

d_hyp_eval = np.zeros((n_eval, n_eval))
for i in range(n_eval):
    for j in range(i+1, n_eval):
        d = ball.dist(torch.FloatTensor(hyp_emb[eval_idx[i]:eval_idx[i]+1]),
                      torch.FloatTensor(hyp_emb[eval_idx[j]:eval_idx[j]+1]))
        d_hyp_eval[i,j] = d.item()
        d_hyp_eval[j,i] = d.item()

mask_upper = np.triu(np.ones((n_eval, n_eval), dtype=bool), k=1)
r_hyp_pca, _ = pearsonr(d_hyp_eval[mask_upper], D_euclidean[np.ix_(eval_idx, eval_idx)][mask_upper])
r_hyp_graph, _ = pearsonr(d_hyp_eval[mask_upper], D_graph[np.ix_(eval_idx, eval_idx)][mask_upper])
print(f"双曲距离 vs PCA欧氏距离 相关性: r={r_hyp_pca:.3f}")
print(f"双曲距离 vs Graph距离 相关性:   r={r_hyp_graph:.3f}")

D_spatial_eval = squareform(pdist(spatial_coords[eval_idx]))
r_hyp_spatial, _ = pearsonr(d_hyp_eval[mask_upper], D_spatial_eval[mask_upper])
r_pca_spatial, _ = pearsonr(D_euclidean[np.ix_(eval_idx, eval_idx)][mask_upper], D_spatial_eval[mask_upper])
print(f"双曲距离 vs 空间距离 相关性: r={r_hyp_spatial:.3f}")
print(f"PCA距离 vs 空间距离 相关性:   r={r_pca_spatial:.3f}")

# k-NN保留率 (vs PCA)
k_nn = 15
nn_pca = NearestNeighbors(n_neighbors=k_nn+1).fit(pca_10d)
_, idx_pca = nn_pca.kneighbors(pca_10d)

nn_hyp = NearestNeighbors(n_neighbors=k_nn+1).fit(hyp_emb)
_, idx_hyp = nn_hyp.kneighbors(hyp_emb)

retention = np.mean([
    len(set(idx_pca[i, 1:k_nn+1]) & set(idx_hyp[i, 1:k_nn+1])) / k_nn
    for i in range(n)
])
print(f"k-NN保留率 (k={k_nn}): {retention:.3f}")

# ============================================================
# Step 6: 保存
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 保存")
print("=" * 60)

adata.obsm['X_poincare'] = hyp_emb
adata.obs['poincare_radius'] = norms

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
adata.obs['cluster'] = kmeans.fit_predict(hyp_emb).astype(str)

adata.write('E:/双曲空间模型/spatial_data/21_00731_hyperbolic_v3.h5ad')
print("已保存: 21_00731_hyperbolic_v3.h5ad")

print("\n" + "=" * 60)
print("CP2 v3 检查清单:")
print("=" * 60)
print(f"[{'x' if inside_disk else ' '}] 所有点在Poincaré disk内")
print(f"[{'x' if retention > 0.15 else ' '}] k-NN保留率 > 0.15")
print(f"[{'x' if norms.max() < 0.95 else ' '}] 没有点在边界")
print(f"[{'x' if r_hyp_pca > 0.5 else ' '}] 双曲-PCA相关性 > 0.5")
print(f"[{'x' if norms.std() > 0.05 else ' '}] 嵌入展开度足够 (std > 0.05)")
print("=" * 60)
