"""
Hyperbolic Niche: Geodesic-distance neighborhoods for cell-cell interaction analysis
"""

import numpy as np
import geoopt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans


class HyperbolicNiche:
    """
    Hyperbolic Niche analysis for spatial transcriptomics.

    Defines cell neighborhoods using geodesic distance on the Poincaré disk
    and compares them against Euclidean neighborhoods.

    Parameters
    ----------
    curvature : float, default=0.5
        Curvature of the Poincaré ball (must match embedding curvature).
    percentile : float, default=10
        Percentile of pairwise distances to use as niche radius.
    min_niche_size : int, default=3
        Minimum number of spots required for a valid niche.
    """

    def __init__(self, curvature=0.5, percentile=10, min_niche_size=3):
        self.curvature = curvature
        self.percentile = percentile
        self.min_niche_size = min_niche_size
        self.ball_ = geoopt.PoincareBall(c=curvature)

    def compute_distances(self, embedding):
        """
        Compute pairwise hyperbolic distance matrix.

        Parameters
        ----------
        embedding : ndarray of shape (n_samples, 2)
            Poincaré disk coordinates.

        Returns
        -------
        D_hyperbolic : ndarray of shape (n_samples, n_samples)
            Pairwise geodesic distance matrix.
        """
        n = len(embedding)
        tensor = torch.FloatTensor(embedding)
        D = np.zeros((n, n))
        chunk = 300
        for i in range(0, n, chunk):
            ei = min(i + chunk, n)
            for j in range(0, n, chunk):
                ej = min(j + chunk, n)
                d = self.ball_.dist(tensor[i:ei].unsqueeze(1), tensor[j:ej].unsqueeze(0))
                D[i:ei, j:ej] = d.detach().numpy()
        return D

    def get_niche(self, D_matrix, center_idx, radius):
        """Return indices of spots within radius of center."""
        return np.where(D_matrix[center_idx] < radius)[0]

    def niche_composition(self, scores, niche_indices):
        """Compute mean module score for each cell type in niche."""
        return {ct: np.mean(score[niche_indices]) for ct, score in scores.items()}

    def niche_purity(self, labels, niche_indices):
        """Compute purity (proportion of most abundant class)."""
        niche_labels = labels[niche_indices]
        if len(niche_labels) == 0:
            return 0
        return max(np.bincount(niche_labels.astype(int))) / len(niche_labels)

    def analyze(self, hyp_embedding, euclidean_distances, scores, n_clusters=8):
        """
        Run full Hyperbolic Niche analysis.

        Parameters
        ----------
        hyp_embedding : ndarray of shape (n_samples, 2)
            Poincaré disk coordinates.
        euclidean_distances : ndarray of shape (n_samples, n_samples)
            Euclidean distance matrix (e.g., spatial distances).
        scores : dict
            Cell type module scores {name: array}.
        n_clusters : int
            Number of clusters for purity computation.

        Returns
        -------
        results : dict
            Dictionary with purity statistics and interaction enrichment.
        """
        import torch

        n = len(hyp_embedding)
        D_hyperbolic = self.compute_distances(hyp_embedding)

        # Niche radii
        hyp_radius = np.percentile(D_hyperbolic[D_hyperbolic > 0], self.percentile)
        euc_radius = np.percentile(euclidean_distances[euclidean_distances > 0], self.percentile)

        # Cluster labels
        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels_km = km.fit_predict(hyp_embedding)

        # Niche purity comparison
        hyp_purities = []
        euc_purities = []
        step = max(1, n // 200)
        for i in range(0, n, step):
            hyp_niche = self.get_niche(D_hyperbolic, i, hyp_radius)
            euc_niche = self.get_niche(euclidean_distances, i, euc_radius)

            if len(hyp_niche) >= self.min_niche_size:
                hyp_purities.append(self.niche_purity(labels_km, hyp_niche))
            if len(euc_niche) >= self.min_niche_size:
                euc_purities.append(self.niche_purity(labels_km, euc_niche))

        stat, p_val = mannwhitneyu(hyp_purities, euc_purities)

        return {
            'D_hyperbolic': D_hyperbolic,
            'hyp_radius': hyp_radius,
            'euc_radius': euc_radius,
            'hyp_purities': hyp_purities,
            'euc_purities': euc_purities,
            'hyp_purity_mean': np.mean(hyp_purities),
            'euc_purity_mean': np.mean(euc_purities),
            'purity_pval': p_val,
            'labels_km': labels_km,
        }


import torch  # needed for compute_distances
