"""
Poincaré MDS: Core embedding algorithm
"""

import numpy as np
import torch
import geoopt
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class PoincareMDS:
    """
    Poincaré Multidimensional Scaling for spatial transcriptomics.

    Embeds high-dimensional data into the Poincaré disk model of hyperbolic space,
    preserving local neighborhood structure via k-NN graph shortest-path distances.

    Parameters
    ----------
    curvature : float, default=0.5
        Curvature of the Poincaré ball. Smaller values produce flatter geometry.
    n_components : int, default=2
        Dimensionality of the embedding (2 for Poincaré disk).
    n_epochs : int, default=2000
        Number of optimization epochs.
    learning_rate : float, default=0.05
        Learning rate for Riemannian Adam optimizer.
    batch_size : int, default=100000
        Number of point pairs sampled per epoch for mini-batch MDS.
    k_neighbors : int, default=30
        Number of neighbors for k-NN graph construction.
    target_radius : float, default=0.4
        Target mean radius to prevent embedding collapse.
    repulsion_weight : float, default=0.5
        Weight of the repulsion loss.
    random_state : int, default=42
        Random seed for reproducibility.

    Examples
    --------
    >>> from poincare_mds import PoincareMDS
    >>> model = PoincareMDS(curvature=0.5, n_epochs=1500)
    >>> embedding = model.fit_transform(X_pca)
    """

    def __init__(self, curvature=0.5, n_components=2, n_epochs=2000,
                 learning_rate=0.05, batch_size=100000, k_neighbors=30,
                 target_radius=0.4, repulsion_weight=0.5, random_state=42):
        self.curvature = curvature
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.k_neighbors = k_neighbors
        self.target_radius = target_radius
        self.repulsion_weight = repulsion_weight
        self.random_state = random_state
        self.embedding_ = None
        self.ball_ = None

    def _compute_target_distance(self, X):
        """Compute k-NN graph shortest-path distance as target."""
        n = len(X)
        k = min(self.k_neighbors, n // 5)

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, indices = nn.kneighbors(X)

        rows = np.repeat(np.arange(n), k)
        cols = indices[:, 1:].flatten()
        weights = distances[:, 1:].flatten()

        adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
        adj = (adj + adj.T) / 2
        D_graph = shortest_path(adj, directed=False, method='D')
        D_graph[D_graph == np.inf] = np.nanmax(D_graph[D_graph != np.inf]) * 2

        D_target = D_graph / np.nanmax(D_graph) * 0.95
        return np.nan_to_num(D_target, nan=0.95)

    def _initialize(self, D_target, n):
        """Initialize embedding via Torgerson scaling."""
        self.ball_ = geoopt.PoincareBall(c=self.curvature)

        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D_target ** 2) @ H
        eigenvalues, eigenvectors = eigh(B)
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        L = np.diag(np.sqrt(np.maximum(eigenvalues[idx], 0)))
        V = eigenvectors[:, idx]
        init_coords = V @ L

        init_coords = init_coords / (np.abs(init_coords).max() + 1e-8) * 0.8
        init_tensor = torch.FloatTensor(init_coords)
        init_tensor = self.ball_.projx(init_tensor)

        return geoopt.ManifoldParameter(init_tensor, manifold=self.ball_)

    def fit_transform(self, X, verbose=True):
        """
        Fit the Poincaré MDS embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data (e.g., PCA coordinates).
        verbose : bool, default=True
            Whether to print progress.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Poincaré disk coordinates.
        """
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        n = len(X)
        D_target = self._compute_target_distance(X)
        embeddings = self._initialize(D_target, n)

        D_target_tensor = torch.FloatTensor(D_target)
        optimizer = geoopt.optim.RiemannianAdam(
            [embeddings], lr=self.learning_rate, weight_decay=1e-5
        )

        for epoch in range(self.n_epochs):
            i_idx = torch.randint(0, n, (self.batch_size,))
            j_idx = torch.randint(0, n, (self.batch_size,))
            mask = i_idx != j_idx
            i_idx, j_idx = i_idx[mask], j_idx[mask]

            d_hyp = self.ball_.dist(embeddings[i_idx], embeddings[j_idx])
            d_tgt = D_target_tensor[i_idx, j_idx]
            mds_loss = ((d_hyp - d_tgt) ** 2).mean()

            norms = embeddings.norm(dim=1)
            radius_loss = torch.relu(self.target_radius - norms.mean()) ** 2
            loss = mds_loss + self.repulsion_weight * radius_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                embeddings.data = self.ball_.projx(embeddings.data)

            if verbose and (epoch + 1) % 500 == 0:
                norms_np = embeddings.data.norm(dim=1)
                print(f"  Epoch {epoch+1:4d}/{self.n_epochs} | Loss: {loss.item():.6f} | "
                      f"Norm: {norms_np.min():.4f}-{norms_np.max():.4f}")

        self.embedding_ = embeddings.detach().numpy()
        return self.embedding_

    def get_distances(self, sample_indices=None):
        """
        Compute pairwise geodesic distances in the embedding.

        Parameters
        ----------
        sample_indices : array-like or None
            If provided, compute distances only for these indices.

        Returns
        -------
        distances : ndarray
            Pairwise geodesic distance matrix.
        """
        if self.embedding_ is None:
            raise ValueError("Model not fitted. Call fit_transform first.")

        emb = self.embedding_
        if sample_indices is not None:
            emb = emb[sample_indices]

        n = len(emb)
        tensor = torch.FloatTensor(emb)
        D = np.zeros((n, n))
        chunk = 300
        for i in range(0, n, chunk):
            ei = min(i + chunk, n)
            for j in range(0, n, chunk):
                ej = min(j + chunk, n)
                d = self.ball_.dist(tensor[i:ei].unsqueeze(1), tensor[j:ej].unsqueeze(0))
                D[i:ei, j:ej] = d.detach().numpy()
        return D

    def get_norms(self):
        """Get the Poincaré radius (norm) of each point."""
        if self.embedding_ is None:
            raise ValueError("Model not fitted. Call fit_transform first.")
        return np.linalg.norm(self.embedding_, axis=1)
