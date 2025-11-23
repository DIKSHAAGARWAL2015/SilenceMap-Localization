import numpy as np
from scipy.spatial.distance import cdist


def _cluster_metrics(gt_nodes, pred_nodes, coords):
    """
    Compute metrics for a single (GT cluster, predicted cluster) pair.

    gt_nodes, pred_nodes: iterables of node indices
    coords: (p,3) array of source coordinates

    Returns dict with Jaccard, ΔCOM, size stats.
    """
    gt_nodes = set(gt_nodes)
    pred_nodes = set(pred_nodes)

    # Jaccard index
    inter = len(gt_nodes & pred_nodes)
    union = len(gt_nodes | pred_nodes)
    jaccard = inter / (union + 1e-12)

    # Sizes
    size_gt = len(gt_nodes)
    size_pred = len(pred_nodes)
    size_rel_error = abs(size_pred - size_gt) / (size_gt + 1e-12)

    # ΔCOM (centroid distance)
    if size_gt > 0 and size_pred > 0:
        gt_coords = coords[list(gt_nodes)]
        pred_coords = coords[list(pred_nodes)]
        gt_com = gt_coords.mean(axis=0)
        pred_com = pred_coords.mean(axis=0)
        delta_com = float(np.linalg.norm(gt_com - pred_com))
    else:
        gt_com = np.full(3, np.nan, dtype=float)
        pred_com = np.full(3, np.nan, dtype=float)
        delta_com = np.nan

    return {
        "jaccard": jaccard,
        "delta_com": delta_com,
        "size_gt": size_gt,
        "size_pred": size_pred,
        "size_rel_error": size_rel_error,
        "gt_com": gt_com,
        "pred_com": pred_com,
    }


def _match_clusters_by_centroid(gt_ranked, pred_ranked):
    """
    Match GT clusters to predicted clusters using centroid distance.

    IMPORTANT:
    - Does NOT assume cluster_id matches.
    - Uses centroid distance matrix and a greedy assignment.

    gt_ranked, pred_ranked: outputs of your rank_clusters(...)
      each is a list of dicts with key "centroid".

    Returns list of (gt_idx, pred_idx) pairs (indices into gt_ranked/pred_ranked).
    """
    if len(gt_ranked) == 0 or len(pred_ranked) == 0:
        return []

    gt_centroids = []
    pred_centroids = []

    for c in gt_ranked:
        if c["centroid"] is None:
            gt_centroids.append([np.nan, np.nan, np.nan])
        else:
            gt_centroids.append(c["centroid"])

    for c in pred_ranked:
        if c["centroid"] is None:
            pred_centroids.append([np.nan, np.nan, np.nan])
        else:
            pred_centroids.append(c["centroid"])

    gt_centroids = np.asarray(gt_centroids, dtype=float)
    pred_centroids = np.asarray(pred_centroids, dtype=float)

    # Distance matrix (gt x pred)
    D = cdist(gt_centroids, pred_centroids)  # shape (G, P)

    matches = []
    used_pred = set()

    for gi in range(len(gt_ranked)):
        # find closest predicted cluster not used yet
        pj = int(np.argmin(D[gi]))
        while pj in used_pred and np.isfinite(D[gi]).any():
            D[gi, pj] = np.inf
            pj = int(np.argmin(D[gi]))

        # if everything is inf (no valid), we still pair with that min (degenerate)
        used_pred.add(pj)
        matches.append((gi, pj))

    return matches


def evaluate_silence_localization_multi_region(
    gt_ranked, pred_ranked, coords, top_k_gt=None, top_k_pred=None
):
    """
    Evaluate localization performance for multiple regions of silence.

    gt_ranked, pred_ranked:
        Output of your rank_clusters(clusters, coords) for:
        - ground truth clusters
        - predicted clusters

        Each element is a dict with:
        - "nodes": list of node indices
        - "centroid": np.array(3,)
        - "size", "radius", "score", "cluster_id", etc.

    coords:
        (p,3) numpy array of 3D coordinates.

    top_k_gt, top_k_pred:
        Optionally restrict evaluation to top-k clusters in each list (by ranking order).
        If None, use all.

    Returns:
        {
          "per_pair": [ { ... metrics ... }, ... ],
          "mean_jaccard": float,
          "mean_delta_com": float,
          "mean_size_rel_error": float,
        }
    """

    # Optionally trim to top-k clusters (most relevant ones)
    if top_k_gt is not None:
        gt_ranked = gt_ranked[:top_k_gt]
    if top_k_pred is not None:
        pred_ranked = pred_ranked[:top_k_pred]

    # Match GT and predicted clusters by centroid distance
    matches = _match_clusters_by_centroid(gt_ranked, pred_ranked)

    per_pair = []

    for gi, pj in matches:
        gt_cluster = gt_ranked[gi]
        pred_cluster = pred_ranked[pj]

        metrics = _cluster_metrics(gt_cluster["nodes"], pred_cluster["nodes"], coords)
        metrics["gt_index"] = gi
        metrics["pred_index"] = pj
        metrics["gt_cluster_id"] = gt_cluster["cluster_id"]
        metrics["pred_cluster_id"] = pred_cluster["cluster_id"]

        per_pair.append(metrics)

    if len(per_pair) == 0:
        return {
            "per_pair": [],
            "mean_jaccard": np.nan,
            "mean_delta_com": np.nan,
            "mean_size_rel_error": np.nan,
        }

    mean_jaccard = float(np.mean([m["jaccard"] for m in per_pair]))
    mean_delta_com = float(np.mean([m["delta_com"] for m in per_pair]))
    mean_size_rel_error = float(np.mean([m["size_rel_error"] for m in per_pair]))

    return {
        "per_pair": per_pair,
        "mean_jaccard": mean_jaccard,
        "mean_delta_com": mean_delta_com,
        "mean_size_rel_error": mean_size_rel_error,
    }
