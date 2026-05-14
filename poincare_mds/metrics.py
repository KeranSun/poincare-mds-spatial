"""
Evaluation metrics for spatial transcriptomics embeddings.
"""

import numpy as np
from scipy.stats import spearmanr


def global_distance_preservation(D_original, D_embedding):
    """Spearman correlation between original and embedding distance matrices.

    Parameters
    ----------
    D_original : ndarray (n, n)
        Distance matrix in the original space (e.g., PCA or spatial).
    D_embedding : ndarray (n, n)
        Distance matrix in the embedding space.

    Returns
    -------
    rho : float
        Spearman correlation coefficient.
    pvalue : float
        p-value for the correlation.
    """
    mask = np.triu(np.ones(D_original.shape, dtype=bool), k=1)
    return spearmanr(D_original[mask], D_embedding[mask])


def hierarchical_gold_standard_score(embedding, ground_truth_labels):
    """Compute NMI and ARI against ground-truth labels.

    Parameters
    ----------
    embedding : ndarray (n, 2)
        2D embedding coordinates.
    ground_truth_labels : array-like (n,)
        Ground-truth cluster/zone labels.

    Returns
    -------
    dict with 'nmi' and 'ari' keys.
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans

    n_clusters = len(np.unique(ground_truth_labels))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(embedding)
    return {
        'nmi': normalized_mutual_info_score(ground_truth_labels, pred),
        'ari': adjusted_rand_score(ground_truth_labels, pred),
    }


def trustworthiness_score(D_original, embedding, k=15):
    """Trustworthiness: measures k-NN preservation quality.

    Parameters
    ----------
    D_original : ndarray (n, n)
        Original distance matrix.
    embedding : ndarray (n, 2)
        Embedding coordinates.
    k : int
        Number of neighbors.

    Returns
    -------
    float
        Trustworthiness score in [0, 1], higher is better.
    """
    from sklearn.manifold import trustworthiness
    return trustworthiness(D_original, embedding, n_neighbors=k)


def radius_label_correlation(norms, labels):
    """Spearman correlation between Poincaré radius and ordinal labels.

    Parameters
    ----------
    norms : ndarray (n,)
        Poincaré radii (Euclidean norms in disk).
    labels : array-like (n,)
        Ordinal labels (e.g., differentiation stage 0, 1, 2, ...).

    Returns
    -------
    rho : float
        Spearman correlation.
    pvalue : float
        p-value.
    """
    return spearmanr(norms, labels)


def all_metrics(D_original, D_embedding, embedding, labels=None, k=15):
    """Compute all benchmark metrics in one call.

    Parameters
    ----------
    D_original : ndarray (n, n)
        Original distance matrix.
    D_embedding : ndarray (n, n)
        Embedding distance matrix.
    embedding : ndarray (n, 2)
        Embedding coordinates.
    labels : array-like (n,), optional
        Ground-truth labels for NMI/ARI.
    k : int
        Number of neighbors for trustworthiness.

    Returns
    -------
    dict with all computed metrics.
    """
    results = {}
    rho, pval = global_distance_preservation(D_original, D_embedding)
    results['spearman_rho'] = rho
    results['spearman_p'] = pval
    results['trustworthiness'] = trustworthiness_score(D_original, embedding, k=k)
    if labels is not None:
        results.update(hierarchical_gold_standard_score(embedding, labels))
    return results
