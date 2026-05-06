"""
A: 多样本验证 — 对全部10个Visium样本运行Poincaré MDS + Niche分析
验证结果可重复性
"""

import numpy as np
import scanpy as sc
import torch
import geoopt
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 核心函数
# ============================================================

def preprocess_sample(h5_path, n_pcs=10):
    """读取并预处理一个Visium样本"""
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # 空间坐标
    sample_id = os.path.basename(os.path.dirname(h5_path))
    spatial_path = os.path.join(os.path.dirname(h5_path), 'spatial', 'tissue_positions_list.csv')
    if os.path.exists(spatial_path):
        spatial_df = pd.read_csv(spatial_path, header=None,
                                  names=['barcode', 'in_tissue', 'row', 'col', 'px_row', 'px_col'])
        spatial_df = spatial_df.set_index('barcode')
        common = adata.obs_names.intersection(spatial_df.index)
        adata = adata[common].copy()
        adata.obsm['spatial'] = spatial_df.loc[common, ['px_col', 'px_row']].values.astype(float)
    else:
        # 尝试其他格式
        spatial_path2 = os.path.join(os.path.dirname(h5_path), 'spatial')
        for f in os.listdir(spatial_path2):
            if 'positions' in f.lower():
                print(f"  Found spatial file: {f}")
                break

    # 过滤
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=10)

    # 标准化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 高变基因
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()

    # PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')

    return adata


def poincare_mds(pca_coords, c=0.5, n_epochs=1500, lr=0.05, batch_size=50000):
    """Poincaré MDS嵌入"""
    n = len(pca_coords)
    D_euclidean = squareform(pdist(pca_coords))

    # k-NN graph distance
    k = min(30, n // 5)
    nn = NearestNeighbors(n_neighbors=k+1).fit(pca_coords)
    distances, indices = nn.kneighbors(pca_coords)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].flatten()
    weights = distances[:, 1:].flatten()
    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
    adj = (adj + adj.T) / 2
    D_graph = shortest_path(adj, directed=False, method='D')
    D_graph[D_graph == np.inf] = np.nanmax(D_graph[D_graph != np.inf]) * 2
    D_target = D_graph / np.nanmax(D_graph) * 0.95
    D_target = np.nan_to_num(D_target, nan=0.95)

    # 初始化
    ball = geoopt.PoincareBall(c=c)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D_target ** 2) @ H
    eigenvalues, eigenvectors = eigh(B)
    idx = np.argsort(eigenvalues)[::-1][:2]
    L = np.diag(np.sqrt(np.maximum(eigenvalues[idx], 0)))
    V = eigenvectors[:, idx]
    init_coords = V @ L
    init_coords = init_coords / (np.abs(init_coords).max() + 1e-8) * 0.8
    init_tensor = torch.FloatTensor(init_coords)
    init_tensor = ball.projx(init_tensor)

    embeddings = geoopt.ManifoldParameter(init_tensor, manifold=ball)
    D_target_tensor = torch.FloatTensor(D_target)
    optimizer = geoopt.optim.RiemannianAdam([embeddings], lr=lr, weight_decay=1e-5)

    for epoch in range(n_epochs):
        i_idx = torch.randint(0, n, (batch_size,))
        j_idx = torch.randint(0, n, (batch_size,))
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]

        d_hyp = ball.dist(embeddings[i_idx], embeddings[j_idx])
        d_tgt = D_target_tensor[i_idx, j_idx]
        mds_loss = ((d_hyp - d_tgt) ** 2).mean()
        radius_loss = torch.relu(0.4 - embeddings.norm(dim=1).mean()) ** 2
        loss = mds_loss + 0.5 * radius_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            embeddings.data = ball.projx(embeddings.data)

    return embeddings.detach().numpy(), D_euclidean, D_graph


def compute_niche_metrics(hyp_emb, D_euclidean, scores, ball_c=0.5):
    """计算niche指标"""
    n = len(hyp_emb)
    ball = geoopt.PoincareBall(c=ball_c)
    hyp_tensor = torch.FloatTensor(hyp_emb)

    # 双曲距离矩阵
    D_hyperbolic = np.zeros((n, n))
    chunk = 300
    for i in range(0, n, chunk):
        ei = min(i + chunk, n)
        for j in range(0, n, chunk):
            ej = min(j + chunk, n)
            d = ball.dist(hyp_tensor[i:ei].unsqueeze(1), hyp_tensor[j:ej].unsqueeze(0))
            D_hyperbolic[i:ei, j:ej] = d.detach().numpy()

    # Niche半径
    hyp_radius = np.percentile(D_hyperbolic[D_hyperbolic > 0], 10)
    euc_radius = np.percentile(D_euclidean[D_euclidean > 0], 10)

    # Niche纯度
    km = KMeans(n_clusters=min(8, n // 10), random_state=42)
    labels_km = km.fit_predict(hyp_emb)

    hyp_purities = []
    euc_purities = []
    step = max(1, n // 200)
    for i in range(0, n, step):
        hyp_niche = np.where(D_hyperbolic[i] < hyp_radius)[0]
        euc_niche = np.where(D_euclidean[i] < euc_radius)[0]
        if len(hyp_niche) >= 3:
            niche_labels = labels_km[hyp_niche]
            hyp_purities.append(max(np.bincount(niche_labels)) / len(niche_labels))
        if len(euc_niche) >= 3:
            niche_labels = labels_km[euc_niche]
            euc_purities.append(max(np.bincount(niche_labels)) / len(niche_labels))

    # 统计检验
    stat, p_val = mannwhitneyu(hyp_purities, euc_purities) if len(hyp_purities) > 0 and len(euc_purities) > 0 else (0, 1)

    return {
        'hyp_purity_mean': np.mean(hyp_purities) if hyp_purities else 0,
        'hyp_purity_std': np.std(hyp_purities) if hyp_purities else 0,
        'euc_purity_mean': np.mean(euc_purities) if euc_purities else 0,
        'euc_purity_std': np.std(euc_purities) if euc_purities else 0,
        'purity_pval': p_val,
        'hyp_radius': hyp_radius,
        'euc_radius': euc_radius,
    }


# ============================================================
# 主流程
# ============================================================
print("=" * 60)
print("Multi-Sample Validation")
print("=" * 60)

data_dir = 'E:/双曲空间模型/spatial_data'
samples = sorted([d for d in os.listdir(data_dir) if d.endswith('_LI_SING')])
print(f"找到 {len(samples)} 个样本")

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

all_results = []

for sample in samples:
    print(f"\n{'='*60}")
    print(f"Processing: {sample}")
    print(f"{'='*60}")

    h5_path = os.path.join(data_dir, sample, f'{sample}_filtered_feature_bc_matrix.h5')

    try:
        # 预处理
        adata = preprocess_sample(h5_path, n_pcs=10)
        n = len(adata)
        print(f"  Spots: {n}")

        if n < 100:
            print(f"  SKIP: too few spots")
            continue

        # Poincaré MDS
        pca_coords = adata.obsm['X_pca'][:, :10]
        hyp_emb, D_euclidean, D_graph = poincare_mds(pca_coords, n_epochs=1000, batch_size=min(50000, n*10))

        norms = np.linalg.norm(hyp_emb, axis=1)
        inside_disk = (norms < 1.0).all()

        # 距离相关性
        np.random.seed(42)
        n_eval = min(500, n)
        eval_idx = np.random.choice(n, n_eval, replace=False)
        d_hyp_eval = np.zeros((n_eval, n_eval))
        ball = geoopt.PoincareBall(c=0.5)
        hyp_eval = hyp_emb[eval_idx]
        hyp_t = torch.FloatTensor(hyp_eval)
        for i in range(n_eval):
            for j in range(i+1, n_eval):
                d = ball.dist(hyp_t[i:i+1], hyp_t[j:j+1])
                d_hyp_eval[i,j] = d.item()
                d_hyp_eval[j,i] = d.item()

        mask_upper = np.triu(np.ones((n_eval, n_eval), dtype=bool), k=1)
        D_euc_eval = cdist(pca_coords[eval_idx], pca_coords[eval_idx])
        D_graph_eval = D_graph[np.ix_(eval_idx, eval_idx)]

        r_hyp_pca, _ = pearsonr(d_hyp_eval[mask_upper], D_euc_eval[mask_upper])
        r_hyp_graph, _ = pearsonr(d_hyp_eval[mask_upper], D_graph_eval[mask_upper])

        # 空间距离
        if 'spatial' in adata.obsm:
            spatial = adata.obsm['spatial']
            D_spatial = cdist(spatial[eval_idx], spatial[eval_idx])
            r_hyp_spatial, _ = pearsonr(d_hyp_eval[mask_upper], D_spatial[mask_upper])
            r_pca_spatial, _ = pearsonr(D_euc_eval[mask_upper], D_spatial[mask_upper])
        else:
            r_hyp_spatial = r_pca_spatial = 0

        # k-NN保留率
        k_nn = 15
        nn_pca = NearestNeighbors(n_neighbors=k_nn+1).fit(pca_coords)
        _, idx_pca = nn_pca.kneighbors(pca_coords)
        nn_hyp = NearestNeighbors(n_neighbors=k_nn+1).fit(hyp_emb)
        _, idx_hyp = nn_hyp.kneighbors(hyp_emb)
        retention = np.mean([
            len(set(idx_pca[i, 1:k_nn+1]) & set(idx_hyp[i, 1:k_nn+1])) / k_nn
            for i in range(n)
        ])

        # Module scores
        def module_score(adata, genes):
            avail = [g for g in genes if g in adata.var_names]
            if not avail: return np.zeros(adata.n_obs)
            idx = [list(adata.var_names).index(g) for g in avail]
            expr = adata.X[:, idx]
            if hasattr(expr, 'toarray'): expr = expr.toarray()
            return np.array(expr.mean(axis=1)).flatten()

        scores = {name: module_score(adata, genes) for name, genes in signatures.items()}

        # Niche指标
        niche_metrics = compute_niche_metrics(hyp_emb, D_euclidean, scores)

        # 半径-细胞类型相关性
        radius_corr = {}
        for ct in ['Tumor', 'CAF_m', 'Fibroblast', 'Epithelial']:
            if ct in scores:
                r, p = spearmanr(norms, scores[ct])
                radius_corr[f'r_{ct}'] = r

        result = {
            'sample': sample,
            'n_spots': n,
            'inside_disk': inside_disk,
            'norm_mean': norms.mean(),
            'norm_std': norms.std(),
            'norm_max': norms.max(),
            'r_hyp_pca': r_hyp_pca,
            'r_hyp_graph': r_hyp_graph,
            'r_hyp_spatial': r_hyp_spatial,
            'r_pca_spatial': r_pca_spatial,
            'knn_retention': retention,
            **niche_metrics,
            **radius_corr,
        }
        all_results.append(result)

        print(f"  Spots: {n}")
        print(f"  Hyp-PCA r: {r_hyp_pca:.3f}")
        print(f"  Hyp-Graph r: {r_hyp_graph:.3f}")
        print(f"  Hyp-Spatial r: {r_hyp_spatial:.3f} (PCA: {r_pca_spatial:.3f})")
        print(f"  k-NN retention: {retention:.3f}")
        print(f"  Niche purity: Hyp={niche_metrics['hyp_purity_mean']:.3f} vs Euc={niche_metrics['euc_purity_mean']:.3f} (p={niche_metrics['purity_pval']:.2e})")
        print(f"  All inside disk: {inside_disk}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================
# 汇总结果
# ============================================================
print("\n" + "=" * 60)
print("汇总结果")
print("=" * 60)

df = pd.DataFrame(all_results)
df.to_csv('E:/双曲空间模型/multi_sample_results.csv', index=False)
print(f"\n成功处理 {len(df)} 个样本")
print(f"\n关键指标统计:")
print(f"  Hyp-PCA相关性: {df['r_hyp_pca'].mean():.3f} +/- {df['r_hyp_pca'].std():.3f}")
print(f"  Hyp-Graph相关性: {df['r_hyp_graph'].mean():.3f} +/- {df['r_hyp_graph'].std():.3f}")
print(f"  Hyp-Spatial相关性: {df['r_hyp_spatial'].mean():.3f} +/- {df['r_hyp_spatial'].std():.3f}")
print(f"  PCA-Spatial相关性: {df['r_pca_spatial'].mean():.3f} +/- {df['r_pca_spatial'].std():.3f}")
print(f"  k-NN保留率: {df['knn_retention'].mean():.3f} +/- {df['knn_retention'].std():.3f}")
print(f"  Niche纯度(Hyp): {df['hyp_purity_mean'].mean():.3f} +/- {df['hyp_purity_mean'].std():.3f}")
print(f"  Niche纯度(Euc): {df['euc_purity_mean'].mean():.3f} +/- {df['euc_purity_mean'].std():.3f}")

# 检查一致性
hyp_wins_spatial = (df['r_hyp_spatial'] > df['r_pca_spatial']).sum()
hyp_wins_purity = (df['hyp_purity_mean'] > df['euc_purity_mean']).sum()
print(f"\n一致性检查:")
print(f"  Hyp-Spatial > PCA-Spatial: {hyp_wins_spatial}/{len(df)} 样本")
print(f"  Hyp纯度 > Euc纯度: {hyp_wins_purity}/{len(df)} 样本")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: Hyp-Spatial vs PCA-Spatial
ax = axes[0, 0]
ax.scatter(df['r_pca_spatial'], df['r_hyp_spatial'], s=80, c='steelblue', edgecolors='black')
ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
ax.set_xlabel('PCA-Spatial Correlation')
ax.set_ylabel('Hyp-Spatial Correlation')
ax.set_title('A. Spatial Correlation (per sample)', fontweight='bold')
for _, row in df.iterrows():
    ax.annotate(row['sample'][:6], (row['r_pca_spatial'], row['r_hyp_spatial']),
                fontsize=6, ha='center', va='bottom')

# Panel B: Niche purity comparison
ax = axes[0, 1]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['hyp_purity_mean'], width, label='Hyperbolic', color='#2196F3')
ax.bar(x + width/2, df['euc_purity_mean'], width, label='Euclidean', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(df['sample'].str[:6], rotation=45, fontsize=7)
ax.set_ylabel('Niche Purity')
ax.set_title('B. Niche Purity (all samples)', fontweight='bold')
ax.legend()

# Panel C: k-NN retention
ax = axes[0, 2]
ax.bar(df['sample'].str[:6], df['knn_retention'], color='steelblue')
ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Threshold')
ax.set_xticklabels(df['sample'].str[:6], rotation=45, fontsize=7)
ax.set_ylabel('k-NN Retention')
ax.set_title('C. k-NN Retention (all samples)', fontweight='bold')
ax.legend()

# Panel D: 距离相关性箱线图
ax = axes[1, 0]
data_box = [df['r_hyp_pca'].values, df['r_hyp_graph'].values]
bp = ax.boxplot(data_box, labels=['Hyp-PCA', 'Hyp-Graph'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2196F3')
bp['boxes'][1].set_facecolor('#4CAF50')
ax.set_ylabel('Pearson r')
ax.set_title('D. Distance Correlation Distribution', fontweight='bold')

# Panel E: 嵌入展开度
ax = axes[1, 1]
ax.bar(df['sample'].str[:6], df['norm_std'], color='steelblue')
ax.set_xticklabels(df['sample'].str[:6], rotation=45, fontsize=7)
ax.set_ylabel('Norm Std')
ax.set_title('E. Embedding Spread', fontweight='bold')

# Panel F: 总结统计
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
Multi-Sample Validation Summary
================================
Samples: {len(df)}

Hyp-Spatial r: {df['r_hyp_spatial'].mean():.3f} +/- {df['r_hyp_spatial'].std():.3f}
PCA-Spatial r: {df['r_pca_spatial'].mean():.3f} +/- {df['r_pca_spatial'].std():.3f}
Hyp > PCA: {hyp_wins_spatial}/{len(df)} samples

Niche Purity (Hyp): {df['hyp_purity_mean'].mean():.3f}
Niche Purity (Euc): {df['euc_purity_mean'].mean():.3f}
Hyp > Euc: {hyp_wins_purity}/{len(df)} samples

k-NN Retention: {df['knn_retention'].mean():.3f}
All inside disk: {df['inside_disk'].all()}
"""
ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.set_title('F. Summary', fontweight='bold')

plt.tight_layout()
plt.savefig('E:/双曲空间模型/Figure_multi_sample.png', dpi=150, bbox_inches='tight')
print('\nSaved: Figure_multi_sample.png')
print('Saved: multi_sample_results.csv')
