# graph.py
# Graph construction + Laplacian + torch sparse adjacency.

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags
import torch
from gnn import device


def knn_graph_gauss(coords, k=12, sigma=12.0):
    """
    Build a k-NN graph with Gaussian weights.

    Returns
    -------
    W : (p,p) coo_matrix
    deg : (p,) float32
    """
    coords = np.asarray(coords, dtype=np.float32)
    p = coords.shape[0]

    nn = NearestNeighbors(n_neighbors=min(k + 1, p)).fit(coords)
    dists, idxs = nn.kneighbors(coords)

    rows, cols, vals = [], [], []
    for i in range(p):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):  # skip self
            w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
            rows.append(i)
            cols.append(j)
            vals.append(w)

    # Symmetrize
    d = {}
    for r, c, v in zip(rows, cols, vals):
        if (r, c) not in d or d[(r, c)] < v:
            d[(r, c)] = v
        if (c, r) not in d or d[(c, r)] < v:
            d[(c, r)] = v

    if not d:
        return coo_matrix(([], ([], [])), shape=(p, p)), np.zeros(p, dtype=np.float32)

    rr, cc, vv = zip(*[(r, c, v) for (r, c), v in d.items()])
    W = coo_matrix((vv, (rr, cc)), shape=(p, p))
    deg = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    return W.tocoo(), deg


def laplacian_from_W(W):
    """
    Combinatorial Laplacian L = D - W.
    """
    deg = np.asarray(W.sum(axis=1)).ravel()
    return (diags(deg) - W).tocoo(), deg


def build_torch_graph(W, deg):
    """
    Convert scipy sparse adjacency to normalized torch sparse tensor:

        A_hat = D^{-1/2} W D^{-1/2}
    """
    W = W.tocoo()
    idx = torch.tensor(
        np.vstack([W.row, W.col]).astype(np.int64),
        device=device,
    )
    di = 1.0 / torch.sqrt(
        torch.clamp(torch.tensor(deg, dtype=torch.float32, device=device), min=1e-8)
    )
    v = torch.tensor(W.data.astype(np.float32), device=device)
    v = v * di[idx[0]] * di[idx[1]]
    Ahat = torch.sparse_coo_tensor(idx, v, W.shape, device=device).coalesce()
    return Ahat
