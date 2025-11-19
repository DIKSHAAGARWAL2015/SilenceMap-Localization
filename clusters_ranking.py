# clusters_ranking.py
# Extract connected components from mask + rank clusters.

from itertools import combinations

import numpy as np
from scipy.sparse.csgraph import connected_components


def extract_clusters_from_mask_sparse(mask, W):
    """
    Extract connected components from subgraph induced by mask.

    mask : (p,) bool
    W    : (p,p) sparse adjacency
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]

    if idx.size == 0:
        return []

    W_csr = W.tocsr()
    W_sub = W_csr[mask][:, mask]

    n_comp, labels = connected_components(
        csgraph=W_sub, directed=False, connection="weak"
    )

    clusters = []
    for c in range(n_comp):
        members = idx[labels == c]
        if members.size > 0:
            clusters.append(members.tolist())
    return clusters


def rank_clusters(clusters, coords):
    """
    Rank clusters by size and compactness.

    Returns list of dicts sorted by descending score.
    """
    coords = np.asarray(coords)
    cluster_info = []

    for c_id, cluster in enumerate(clusters):
        nodes = np.array(cluster, dtype=int)
        points = coords[nodes]
        size = len(nodes)

        if size > 1:
            dists = [
                np.linalg.norm(points[i] - points[j])
                for i, j in combinations(range(size), 2)
            ]
            mean_dist = float(np.mean(dists))
            max_dist = float(np.max(dists))
        else:
            mean_dist = 0.0
            max_dist = 0.0

        score = size - mean_dist

        cluster_info.append(
            {
                "cluster_id": c_id,
                "nodes": nodes.tolist(),
                "size": size,
                "mean_internal_distance": mean_dist,
                "max_internal_distance": max_dist,
                "score": score,
            }
        )

    return sorted(cluster_info, key=lambda x: x["score"], reverse=True)
