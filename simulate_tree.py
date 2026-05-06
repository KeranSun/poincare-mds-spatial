"""
CP4 v3: 二叉树层次结构验证
核心：生成二叉树结构数据，验证Poincaré MDS能否恢复层次
关键指标：半径-深度相关性（树深度越大→Poincaré半径越大）
"""

import numpy as np
import torch
import geoopt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# Step 1: 生成二叉树层次数据
# ============================================================
print("=" * 60)
print("Step 1: 生成二叉树层次数据")
print("=" * 60)

def generate_binary_tree_data(depth=4, n_per_leaf=30, noise=0.3):
    """
    生成二叉树结构数据
    每个叶节点代表一种"细胞类型"
    树深度 = 层次数
    距离 = 树上路径长度 (真正的层次距离)
    """
    n_leaves = 2 ** depth  # 16个叶节点
    n_total = n_leaves * n_per_leaf

    # 为每个叶节点分配坐标
    # 用树的结构来定义：左=0, 右=1
    leaf_codes = []
    for i in range(n_leaves):
        code = format(i, f'0{depth}b')  # 如 '0000', '0001', ...
        leaf_codes.append(code)

    # 计算树上距离 (两个叶节点之间的路径长度)
    def tree_distance(code1, code2):
        """树上距离 = 最近公共祖先的深度"""
        # 找最长公共前缀
        common = 0
        for a, b in zip(code1, code2):
            if a == b:
                common += 1
            else:
                break
        # 距离 = 2 * (depth - common_ancestor_depth)
        return 2 * (depth - common)

    # 计算树距离矩阵
    D_tree = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        for j in range(n_leaves):
            D_tree[i, j] = tree_distance(leaf_codes[i], leaf_codes[j])

    # 生成数据点
    # 每个叶节点的"表达谱" = 基于树路径的one-hot编码 + 噪声
    # 这样距离矩阵自然反映树结构
    n_features = 50

    # 为每个叶节点定义特征向量
    # 路径上每个分叉点贡献一个特征
    leaf_features = np.zeros((n_leaves, n_features))
    for i, code in enumerate(leaf_codes):
        for d in range(depth):
            # 路径上的每个决策 (0或1) 对应一个特征
            feature_idx = d * 2 + int(code[d])
            leaf_features[i, feature_idx] = 1.0

    # 生成数据点
    X = np.zeros((n_total, n_features))
    labels_leaf = []
    labels_depth = []  # 树深度 (用公共祖先深度表示)

    idx = 0
    for i in range(n_leaves):
        pts = leaf_features[i] + np.random.randn(n_per_leaf, n_features) * noise
        X[idx:idx+n_per_leaf] = pts
        labels_leaf.extend([leaf_codes[i]] * n_per_leaf)
        # 深度 = 从根到叶的距离
        labels_depth.extend([depth] * n_per_leaf)
        idx += n_per_leaf

    # 计算欧氏距离矩阵
    D_euclidean = squareform(pdist(X))

    return X, D_euclidean, D_tree, leaf_codes, labels_leaf, np.array(labels_depth)

X, D_euclidean, D_tree, leaf_codes, labels_leaf, labels_depth = generate_binary_tree_data(
    depth=4, n_per_leaf=30, noise=0.3
)
n = len(X)
n_leaves = len(leaf_codes)

print(f"样本数: {n}")
print(f"叶节点数: {n_leaves}")
print(f"树深度: 4")
print(f"特征维度: {X.shape[1]}")
print(f"树距离范围: {D_tree.min():.0f} - {D_tree.max():.0f}")

# 扩展树距离矩阵到全样本
D_tree_full = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D_tree_full[i, j] = D_tree[leaf_codes.index(labels_leaf[i]),
                                     leaf_codes.index(labels_leaf[j])]

# ============================================================
# Step 2: Poincaré MDS
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Poincaré MDS嵌入")
print("=" * 60)

# 用树距离作为target (关键！)
D_target = D_tree_full / D_tree_full.max() * 0.95

ball = geoopt.PoincareBall(c=1.0)

from scipy.linalg import eigh
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ (D_target ** 2) @ H
eigenvalues, eigenvectors = eigh(B)
idx = np.argsort(eigenvalues)[::-1][:2]
L = np.diag(np.sqrt(np.maximum(eigenvalues[idx], 0)))
V = eigenvectors[:, idx]
init_coords = V @ L
init_coords = init_coords / (np.abs(init_coords).max() + 1e-8) * 0.5
init_tensor = torch.FloatTensor(init_coords)
init_tensor = ball.projx(init_tensor)

embeddings = geoopt.ManifoldParameter(init_tensor, manifold=ball)
D_target_tensor = torch.FloatTensor(D_target)

batch_size = 30000
optimizer = geoopt.optim.RiemannianAdam([embeddings], lr=0.05, weight_decay=1e-5)

for epoch in range(2000):
    i_idx = torch.randint(0, n, (batch_size,))
    j_idx = torch.randint(0, n, (batch_size,))
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    d_hyp = ball.dist(embeddings[i_idx], embeddings[j_idx])
    d_tgt = D_target_tensor[i_idx, j_idx]
    mds_loss = ((d_hyp - d_tgt) ** 2).mean()

    norms_t = embeddings.norm(dim=1)
    radius_loss = torch.relu(0.3 - norms_t.mean()) ** 2
    loss = mds_loss + 0.3 * radius_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        embeddings.data = ball.projx(embeddings.data)

    if (epoch + 1) % 500 == 0:
        norms_print = embeddings.data.norm(dim=1)
        print(f"  Epoch {epoch+1}: loss={loss.item():.6f} norm={norms_print.min():.4f}-{norms_print.max():.4f}")

emb_poincare = embeddings.detach().numpy()
print(f"嵌入完成")

# ============================================================
# Step 3: 欧氏MDS + t-SNE
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 对比方法")
print("=" * 60)

# 欧氏MDS (用树距离)
print("Euclidean MDS (tree distance)...")
mds_tree = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=500)
emb_mds_tree = mds_tree.fit_transform(D_tree_full)

# 欧氏MDS (用欧氏距离)
print("Euclidean MDS (Euclidean distance)...")
mds_euc = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=500)
emb_mds_euc = mds_euc.fit_transform(D_euclidean)

# t-SNE
print("t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_tsne = tsne.fit_transform(X)

print("完成")

# ============================================================
# Step 4: 评估
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 评估指标")
print("=" * 60)

def evaluate(emb, name, D_tree_full, labels_leaf):
    n = len(emb)
    D_emb = squareform(pdist(emb))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # 1. 树距离相关性
    r_tree, _ = pearsonr(D_tree_full[mask], D_emb[mask])

    # 2. 半径 vs 树深度相关性 (核心指标！)
    norms = np.linalg.norm(emb, axis=1) if emb.shape[1] == 2 else None
    if norms is not None:
        # 计算每个点到根的距离 (树深度)
        # 根 = 全0路径 '0000'
        root_idx = leaf_codes.index('0000')
        tree_depths = D_tree_full[root_idx]
        r_depth, _ = spearmanr(norms, tree_depths)
    else:
        r_depth = float('nan')

    # 3. k-NN保留率 (树距离的k近邻)
    k = 15
    nn_tree = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
    nn_tree.fit(D_tree_full)
    _, idx_tree = nn_tree.kneighbors(D_tree_full)
    nn_emb = NearestNeighbors(n_neighbors=k+1).fit(emb)
    _, idx_emb = nn_emb.kneighbors(emb)

    retention = np.mean([
        len(set(idx_tree[i, 1:k+1]) & set(idx_emb[i, 1:k+1])) / k
        for i in range(n)
    ])

    # 4. 层次保真度
    same_leaf = []
    diff_leaf_same_parent = []
    diff_parent = []
    sample_idx = np.random.choice(n, min(300, n), replace=False)
    for ii in range(len(sample_idx)):
        for jj in range(ii+1, len(sample_idx)):
            i, j = sample_idx[ii], sample_idx[jj]
            d = D_emb[i, j]
            li, lj = labels_leaf[i], labels_leaf[j]
            if li == lj:
                same_leaf.append(d)
            elif li[:3] == lj[:3]:  # 同父节点 (前3位相同)
                diff_leaf_same_parent.append(d)
            else:
                diff_parent.append(d)

    sep = np.mean(diff_parent) / (np.mean(same_leaf) + 1e-8)

    print(f"\n{name}:")
    print(f"  树距离相关性: r={r_tree:.3f}")
    print(f"  半径-深度相关性: r={r_depth:.3f}")
    print(f"  k-NN保留率: {retention:.3f}")
    print(f"  层次保真度: {sep:.3f}")

    return {
        'method': name,
        'tree_corr': r_tree,
        'radius_depth_corr': r_depth,
        'knn_retention': retention,
        'hierarchy_fidelity': sep,
    }

results = []
results.append(evaluate(emb_poincare, 'Poincaré MDS', D_tree_full, labels_leaf))
results.append(evaluate(emb_mds_tree, 'Euclidean MDS (tree)', D_tree_full, labels_leaf))
results.append(evaluate(emb_mds_euc, 'Euclidean MDS (euc)', D_tree_full, labels_leaf))
results.append(evaluate(emb_tsne, 't-SNE', D_tree_full, labels_leaf))

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("总结:")
print("=" * 60)
print(results_df.to_string(index=False))

# ============================================================
# Step 5: 可视化
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Color by leaf code
cmap = plt.cm.tab20
leaf_colors = {code: cmap(i/n_leaves) for i, code in enumerate(leaf_codes)}

# Panel A: 树距离矩阵
ax = axes[0, 0]
im = ax.imshow(D_tree[:n_leaves, :n_leaves], cmap='viridis')
ax.set_xlabel('Leaf Node')
ax.set_ylabel('Leaf Node')
ax.set_title('A. Tree Distance Matrix', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel B: Poincaré MDS
ax = axes[0, 1]
theta_circle = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3)
for code in leaf_codes:
    mask = np.array(labels_leaf) == code
    ax.scatter(emb_poincare[mask, 0], emb_poincare[mask, 1],
               c=[leaf_colors[code]], s=10, alpha=0.6)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_title('B. Poincaré MDS (Ours)', fontweight='bold')

# Panel C: Euclidean MDS (tree distance)
ax = axes[0, 2]
for code in leaf_codes:
    mask = np.array(labels_leaf) == code
    ax.scatter(emb_mds_tree[mask, 0], emb_mds_tree[mask, 1],
               c=[leaf_colors[code]], s=10, alpha=0.6)
ax.set_title('C. Euclidean MDS (tree dist)', fontweight='bold')

# Panel D: t-SNE
ax = axes[1, 0]
for code in leaf_codes:
    mask = np.array(labels_leaf) == code
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=[leaf_colors[code]], s=10, alpha=0.6)
ax.set_title('D. t-SNE', fontweight='bold')

# Panel E: 半径 vs 树深度散点图
ax = axes[1, 1]
norms = np.linalg.norm(emb_poincare, axis=1)
root_idx = leaf_codes.index('0000')
tree_depths = D_tree_full[root_idx]
r, p = spearmanr(norms, tree_depths)
ax.scatter(tree_depths, norms, s=10, alpha=0.5, c='steelblue')
ax.set_xlabel('Tree Depth (distance from root)')
ax.set_ylabel('Poincaré Radius')
ax.set_title(f'E. Radius vs Tree Depth (r={r:.3f})', fontweight='bold')

# Panel F: 方法对比
ax = axes[1, 2]
metrics = ['tree_corr', 'radius_depth_corr', 'knn_retention', 'hierarchy_fidelity']
metric_labels = ['Tree\nCorr', 'Depth\nCorr', 'k-NN\nRetention', 'Hierarchy\nFidelity']
x = np.arange(len(metrics))
width = 0.2
for i, row in results_df.iterrows():
    vals = [row[m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=row['method'])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels, fontsize=8)
ax.set_ylabel('Score')
ax.set_title('F. Method Comparison', fontweight='bold')
ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('E:/双曲空间模型/figure_tree_validation.png', dpi=150, bbox_inches='tight')
print('Saved: figure_tree_validation.png')

results_df.to_csv('E:/双曲空间模型/tree_validation_results.csv', index=False)
print('Saved: tree_validation_results.csv')

print("\n" + "=" * 60)
print("CP4 v3 检查清单:")
print("=" * 60)
po = results_df[results_df['method'] == 'Poincaré MDS'].iloc[0]
print(f"[{'x' if po['tree_corr'] > 0.5 else ' '}] 树距离相关性 > 0.5")
print(f"[{'x' if po['radius_depth_corr'] > 0.5 else ' '}] 半径-深度相关性 > 0.5")
print(f"[{'x' if po['knn_retention'] > 0.15 else ' '}] k-NN保留率 > 0.15")
print("=" * 60)
