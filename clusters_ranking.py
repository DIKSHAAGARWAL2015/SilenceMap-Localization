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
    Rank clusters using a simple score based on size and compactness.

    - size  = number of nodes in the cluster
    - radius = mean distance of nodes to the cluster centroid
    - score = size / radius  (larger & more compact clusters get higher score)

    clusters: list of lists of node indices
    coords: (p,3) numpy array of 3D coordinates

    Returns: ranked list of dicts
    """
    cluster_info = []

    for cid, nodes in enumerate(clusters):
        nodes = np.asarray(nodes, dtype=int)

        if nodes.size == 0:
            cluster_info.append({
                "cluster_id": cid,
                "nodes": [],
                "size": 0,
                "radius": np.inf,
                "centroid": None,
                "score": 0.0,      # no nodes → score 0
            })
            continue

        # Coordinates of this cluster
        coords_S = coords[nodes]              # (k,3)

        # Centroid
        centroid = coords_S.mean(axis=0)      # (3,)

        # Distances to centroid
        dists = np.linalg.norm(coords_S - centroid, axis=1)

        # Compactness radius (mean distance)
        #radius = float(dists.mean())
        radius = float(np.median(dists))
        size = int(nodes.size)

        # Score: size / radius (avoid division by zero)
        if radius > 0:
            score = size / radius
        else:
            # degenerate case: all nodes at exactly same point
            score = float(size)

        cluster_info.append({
            "cluster_id": cid,
            "nodes": nodes.tolist(),
            "size": size,
            "radius": radius,
            "centroid": centroid,
            "score": score,
        })

    # Sort by score (higher is better)
    ranked = sorted(
        cluster_info,
        key=lambda c: c["score"],
        reverse=True,
    )

    return ranked
