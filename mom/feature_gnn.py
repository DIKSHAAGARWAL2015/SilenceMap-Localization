#!/usr/bin/env python3
import os, sys, math, time, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # ensures 3D plotting works
import numpy as np
import mat73
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.signal import butter, filtfilt, welch
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from scipy.signal import welch

def _walk_py(obj, prefix=""):
    """Recursively walk Python objects (dicts/lists/arrays) from mat73."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _walk_py(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += _walk_py(v, f"{prefix}[{i}]")
    elif isinstance(obj, np.ndarray):
        found.append((prefix, obj))
    return found

def _pick_leadfield_from_py(items):
    """Pick a large 2D array as leadfield (prefer names containing 'L'/'leadfield')."""
    best = None
    for path, arr in items:
        if arr.ndim == 2 and arr.size >= 32*256:
            name = path.lower()
            score = arr.size
            if path.split(".")[-1].lower() in ("l",): score += 10_000_000
            if "leadfield" in name: score += 5_000_000
            if (best is None) or (score > best[0]):
                best = (score, path, arr)
    return None if best is None else best[2]

def _pick_vertices_from_py(items):
    """Pick a (p,3) or (3,p) array for cortex vertices (prefer names with 'vertices'/'pial'/'cortex')."""
    best = None
    for path, arr in items:
        if arr.ndim == 2 and 3 in arr.shape:
            name = path.lower()
            score = max(arr.shape)
            if "vertices" in name: score += 1_000_000
            if "pial" in name:     score += 500_000
            if "cortex" in name:   score += 100_000
            if (best is None) or (score > best[0]):
                best = (score, path, arr)
    return None if best is None else best[2]

def load_leadfield_any(leadfield_path, headmodel_path, leadfield_var=None, vertices_var=None):
    """
    Robust loader for MATLAB v7.3 .mat via mat73 (and works for <=v7 too).
    If variable names are known, pass leadfield_var (e.g., 'L')
    and vertices_var (e.g., 'Cortex.Pial.vertices').
    """
    # Leadfield
    lead = mat73.loadmat(leadfield_path)  # dict-like
    if leadfield_var:
        # resolve dotted path like 'foo.bar.baz'
        L = lead
        for part in leadfield_var.split("."):
            L = L[part]
    else:
        items = _walk_py(lead)
        L = _pick_leadfield_from_py(items)
        if L is None:
            raise RuntimeError("Could not find leadfield matrix L in file (mat73).")

    # Vertices
    head = mat73.loadmat(headmodel_path)
    if vertices_var:
        V = head
        for part in vertices_var.split("."):
            V = V[part]
    else:
        items = _walk_py(head)
        V = _pick_vertices_from_py(items)
        if V is None:
            raise RuntimeError("Could not find cortex vertices (Pial) in file (mat73).")

    # Normalize shapes
    L = np.array(L).astype(np.float32)
    V = np.array(V).astype(np.float32)
    if V.ndim != 2:
        V = V.reshape(V.shape[0], -1)
    if V.shape[1] == 3:
        src_xyz = V
    elif V.shape[0] == 3:
        src_xyz = V.T
    else:
        raise RuntimeError(f"Vertices shape {V.shape} not (p,3) or (3,p).")

    # Ensure L is (n,p)
    if L.ndim != 2:
        L = L.reshape(L.shape[0], -1)
    nL, pL = L.shape
    pV = src_xyz.shape[0]
    if pL != pV:
        if nL == pV:
            L = L.T
        else:
            raise RuntimeError(f"Dimension mismatch: L {L.shape}, vertices {src_xyz.shape}")

    return L.astype(np.float32), src_xyz.astype(np.float32)


def show_or_save(fig, name, save=False, outdir="/content/sample_data/figs/"):
    """
    Saves or displays the matplotlib figure.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if save:
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved figure → {path}]")
    else:
        plt.show()

# ---- Light deps; install hints if missing ----
def _need(pkg):
    try:
        __import__(pkg)
        return False
    except Exception:
        return True

_missing = []
for pkg in ["numpy","matplotlib","scipy","sklearn","torch","tqdm"]:
    if _need(pkg): _missing.append(pkg)

if _missing:
    print(">> Missing packages:", _missing)
    print(">> In Colab, run: !pip -q install " + " ".join(_missing))
    sys.exit(1)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def mask_connected_components(W, mask):
    """
    W: sparse adjacency (p,p) (your kNN graph W)
    mask: (p,) bool
    Returns: list of components, each component is a numpy array of vertex indices.
    """
    mask = np.asarray(mask).astype(bool).reshape(-1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    W = W.tocsr()
    sub = W[idx][:, idx]                  # induced subgraph on masked nodes
    sub = ((sub + sub.T) > 0).astype(int) # undirected, binary

    n_comp, labels = connected_components(sub, directed=False, connection='weak')
    comps = []
    for c in range(n_comp):
        comps.append(idx[labels == c])
    # sort by size descending
    comps.sort(key=lambda a: -a.size)
    return comps

def jaccard_set(a, b):
    a = set(a); b = set(b)
    if len(a) == 0 and len(b) == 0: return 1.0
    if len(a) == 0 or len(b) == 0:  return 0.0
    return len(a & b) / (len(a | b) + 1e-12)

def component_metrics(W, gt_mask, pred_mask):
    """
    Returns:
      - set_jaccard (vertex-level)
      - ncomp_gt, ncomp_pred
      - matched component IoUs (greedy match)
      - size errors (abs and relative) for matched comps
      - extras: unmatched components sizes
    """
    gt_mask = np.asarray(gt_mask).astype(bool).reshape(-1)
    pred_mask = np.asarray(pred_mask).astype(bool).reshape(-1)

    # Vertex-set Jaccard
    gt_idx = np.where(gt_mask)[0]
    pr_idx = np.where(pred_mask)[0]
    set_iou = jaccard_set(gt_idx, pr_idx)

    # Components
    gt_comps = mask_connected_components(W, gt_mask)
    pr_comps = mask_connected_components(W, pred_mask)

    # Greedy match components by IoU
    used = np.zeros(len(pr_comps), dtype=bool)
    matches = []  # (gt_i, pr_j, iou, gt_size, pr_size)

    for i, g in enumerate(gt_comps):
        best_j, best_iou = -1, -1.0
        gset = set(g.tolist())
        for j, p in enumerate(pr_comps):
            if used[j]: 
                continue
            iou = jaccard_set(gset, p.tolist())
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j != -1:
            used[best_j] = True
            p = pr_comps[best_j]
            matches.append((i, best_j, best_iou, g.size, p.size))

    # Size errors for matched comps
    abs_size_err = [abs(gs - ps) for (_,_,_,gs,ps) in matches]
    rel_size_err = [abs(gs - ps) / (gs + 1e-12) for (_,_,_,gs,ps) in matches]
    ious = [m[2] for m in matches]

    unmatched_pred_sizes = [pr_comps[j].size for j in range(len(pr_comps)) if not used[j]]
    unmatched_gt_sizes = []  # (if you want, those with no match: rare with greedy, but possible)
    matched_gt = set(m[0] for m in matches)
    for i, g in enumerate(gt_comps):
        if i not in matched_gt:
            unmatched_gt_sizes.append(g.size)

    out = {
        "set_jaccard": set_iou,
        "ncomp_gt": len(gt_comps),
        "ncomp_pred": len(pr_comps),
        "matched_ious": ious,
        "abs_size_err": abs_size_err,
        "rel_size_err": rel_size_err,
        "unmatched_pred_sizes": unmatched_pred_sizes,
        "unmatched_gt_sizes": unmatched_gt_sizes,
        "gt_comp_sizes": [c.size for c in gt_comps],
        "pred_comp_sizes": [c.size for c in pr_comps],
    }
    return out

def print_component_report(name, metrics):
    print(f"\n==== {name} Component Report ====")
    print(f"Vertex-set Jaccard (IoU): {metrics['set_jaccard']:.3f}")
    print(f"#Components: GT={metrics['ncomp_gt']}  Pred={metrics['ncomp_pred']}")
    if metrics["matched_ious"]:
        print(f"Matched comp IoU: mean={np.mean(metrics['matched_ious']):.3f}  "
              f"median={np.median(metrics['matched_ious']):.3f}")
        print(f"Size error abs:   mean={np.mean(metrics['abs_size_err']):.2f}  "
              f"median={np.median(metrics['abs_size_err']):.2f}")
        print(f"Size error rel:   mean={np.mean(metrics['rel_size_err']):.3f}  "
              f"median={np.median(metrics['rel_size_err']):.3f}")
    else:
        print("No matched components.")
    if metrics["unmatched_pred_sizes"]:
        print("Unmatched pred comp sizes:", metrics["unmatched_pred_sizes"][:10])
def enforce_connected_topk(W, prob, k, max_comps=None):
    # start from top-k candidates
    idx = np.argpartition(prob, -k)[-k:]
    cand = np.zeros_like(prob, dtype=bool)
    cand[idx] = True

    comps = mask_connected_components(W, cand)
    if not comps:
        return cand

    # keep biggest components first
    kept = np.zeros_like(prob, dtype=bool)
    count = 0
    used_comps = 0
    for c in comps:
        if max_comps is not None and used_comps >= max_comps:
            break
        if count >= k:
            break
        kept[c] = True
        count += c.size
        used_comps += 1

    # if we overshot k, trim within the last component by prob
    if kept.sum() > k:
        kept_idx = np.where(kept)[0]
        keep_prob = prob[kept_idx]
        trim = kept_idx[np.argsort(keep_prob)[-(k):]]   # keep top prob among kept
        kept2 = np.zeros_like(prob, dtype=bool)
        kept2[trim] = True
        kept = kept2

    return kept
def grow_k_regions(W, prob, K, k_total):
    W = W.tocsr()
    p = len(prob)
    chosen = np.zeros(p, dtype=bool)

    # pick K seeds as top-K peaks
    seeds = np.argsort(prob)[-K:][::-1].tolist()

    # region grow using BFS-like expansion prioritized by prob
    frontier = [set([s]) for s in seeds]
    regions = [set([s]) for s in seeds]
    chosen[seeds] = True

    # precompute neighbors list
    nbrs = [W[i].indices for i in range(p)]

    while chosen.sum() < k_total:
        best_gain = -1.0
        best_r = None
        best_v = None

        for r in range(K):
            # candidate neighbors not already chosen
            cand = set()
            for u in list(frontier[r])[:50]:   # limit work
                for v in nbrs[u]:
                    if not chosen[v]:
                        cand.add(v)
            if not cand:
                continue

            # pick highest-prob candidate
            v = max(cand, key=lambda x: prob[x])
            gain = prob[v]
            if gain > best_gain:
                best_gain = gain
                best_r = r
                best_v = v

        if best_r is None:  # no expansion possible
            break

        regions[best_r].add(best_v)
        chosen[best_v] = True
        frontier[best_r].add(best_v)

    mask = np.zeros(p, dtype=bool)
    mask[np.where(chosen)[0]] = True
    return mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from scipy.signal import welch

def build_reference_matrix(n, i_ref0):
    M = np.eye(n-1, dtype=np.float64)
    return np.concatenate([M[:, :i_ref0], -np.ones((n-1,1)), M[:, i_ref0:]], axis=1)

def compute_betta_silencemap_low(eeg, L, Fs=512, i_ref0=63):
    
    eeg = np.asarray(eeg, dtype=np.float64)   # (n,t)
    L   = np.asarray(L,   dtype=np.float64)   # (n,p)
    n, t = eeg.shape
    M = build_reference_matrix(n, int(i_ref0))

    Y = M @ eeg           # (n-1,t)
    Lr = M @ L            # (n-1,p)

    # Welch window like MATLAB
    w_length = min(int(np.floor(0.5*t)), 256)
    w_over   = int(np.floor(0.5*w_length))

    # ---- Noise PSD 90-100Hz ----
    f, pxx = welch(Y, fs=Fs, nperseg=w_length, noverlap=w_over, nfft=256, axis=1)
    band = (f >= 90.0) & (f <= 100.0)
    eta = pxx[:, band].mean(axis=1)                    # (n-1,)
    sigma_z_sqrd = eta * (100.0 - 0.1)                 # (n-1,)
    Cz = np.diag(sigma_z_sqrd)

    # ---- Var_norm_fact ----
    G = Lr.T @ Lr
    Var_norm_fact = np.sum((G**2), axis=1) + 1e-12     # (p,)

    # ---- Mu_tilda ----
    Mu = Lr.T @ Y                                      # (p,t)

    # ---- Var(mu) from PSD integral ----
    fmu, pxx_mu = welch(Mu, fs=Fs, nperseg=w_length, noverlap=w_over, nfft=256, axis=1)
    df = (fmu[1]-fmu[0]) if len(fmu) > 1 else 1.0
    sigma_mu_sqrd = pxx_mu.sum(axis=1)*df - (Mu.mean(axis=1)**2)

    # ---- subtract diag(L' Cz L) ----
    noise_term = np.diag(Lr.T @ Cz @ Lr)
    sigma_mu_wo_noise = sigma_mu_sqrd - noise_term

    Betta = sigma_mu_wo_noise / Var_norm_fact          # (p,)
    return Betta
def orient_silent_high(arr, X_act):
    arr = np.asarray(arr).reshape(-1).astype(np.float64)
    if arr[X_act].mean() < arr[~X_act].mean():
        return arr  # already silent-high
    else:
        return -arr # flip so silent-high
def topk_mask(arr, k):
    arr = np.asarray(arr).reshape(-1)
    idx = np.argpartition(arr, -k)[-k:]   # k largest
    m = np.zeros(arr.shape[0], dtype=bool)
    m[idx] = True
    return m

import numpy as np
from scipy.signal import welch

def build_reference_matrix(n, i_ref0):
    M = np.eye(n-1, dtype=np.float64)
    return np.concatenate([M[:, :i_ref0], -np.ones((n-1,1)), M[:, i_ref0:]], axis=1)

def estimate_noise_Cz_from_psd(Y, Fs, f0=90.0, f1=100.0):
    wlen  = min(int(np.floor(0.5 * Y.shape[1])), 256)
    wover = int(np.floor(0.5 * wlen))
    f, Pxx = welch(Y, fs=Fs, nperseg=wlen, noverlap=wover, nfft=256, axis=1)

    df = f[1] - f[0] if len(f) > 1 else 1.0
    band = (f >= f0) & (f <= f1)
    if not np.any(band):
        band = f >= f.max() * 0.8

    # variance in the band = ∫ PSD df
    sigma_z_sqrd = (Pxx[:, band].sum(axis=1) * df).astype(np.float64)  # (n-1,)
    Cz = np.diag(sigma_z_sqrd)
    return Cz, sigma_z_sqrd


def var_from_psd(X, Fs):
    # MATLAB: sigma^2 = ∫PSD df - mean^2
    wlen = min(int(np.floor(0.5 * X.shape[1])), 256)
    wover = int(np.floor(0.5 * wlen))
    f, Pxx = welch(X, fs=Fs, nperseg=wlen, noverlap=wover, nfft=256, axis=1)
    df = f[1] - f[0] if len(f) > 1 else 1.0
    sig2 = (Pxx.sum(axis=1) * df) - (X.mean(axis=1) ** 2)
    return sig2.astype(np.float64)


def silencemap_beta_highres(eeg, L, Cs, Fs=512, i_ref0=64):
        
    eeg = np.asarray(eeg, dtype=np.float64)
    L   = np.asarray(L,   dtype=np.float64)
    Cs  = np.asarray(Cs,  dtype=np.float64)

    n = L.shape[0]
    M = build_reference_matrix(n, i_ref0)

    Y    = M @ eeg
    Lref = M @ L

    Cz, _ = estimate_noise_Cz_from_psd(Y, Fs)

    Mu_tilda = Lref.T @ Y                             # (p,t)
    sigma_mu_sqrd = var_from_psd(Mu_tilda, Fs)         # (p,)
    sigma_mu_wo_noise = sigma_mu_sqrd - np.diag(Lref.T @ Cz @ Lref)

    G = Lref.T @ Lref                                 # (p,p)
    denom = np.diag(G @ Cs @ G) + 1e-12

    beta = sigma_mu_wo_noise / denom
    return beta


# ========================= Utilities =========================
def knn_graph_gauss(coords, k=12, sigma=12.0):
    p = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k+1,p)).fit(coords)
    dists, idxs = nn.kneighbors(coords)
    rows, cols, vals = [], [], []
    for i in range(p):
        for j, d in zip(idxs[i,1:], dists[i,1:]):  # skip self
            w = np.exp(-(d**2)/(2.0*sigma**2))
            rows.append(i); cols.append(j); vals.append(w)
    # symmetrize by max
    d = {}
    for r,c,v in zip(rows, cols, vals):
        if (r,c) not in d or d[(r,c)] < v: d[(r,c)] = v
        if (c,r) not in d or d[(c,r)] < v: d[(c,r)] = v
    if not d:
        return coo_matrix(([],([],[])), shape=(p,p)), np.zeros(p, dtype=np.float32)
    rr, cc, vv = zip(*[(r,c,v) for (r,c),v in d.items()])
    W = coo_matrix((vv, (rr,cc)), shape=(p,p))
    deg = np.array(W.sum(axis=1)).ravel().astype(np.float32)
    return W.tocoo(), deg
def neighbor_stats_from_W(W, x):
    """
    W: sparse adjacency (p,p) (weights ok)
    x: (p,) float
    Returns: nbr_mean (p,), nbr_std (p,)
    """
    W = W.tocsr()
    x = np.asarray(x).reshape(-1).astype(np.float32)

    deg = np.array(W.sum(axis=1)).ravel().astype(np.float32)
    deg_safe = np.maximum(deg, 1e-8)

    nbr_mean = (W @ x) / deg_safe
    nbr_mean2 = (W @ (x**2)) / deg_safe
    nbr_var = np.maximum(nbr_mean2 - nbr_mean**2, 0.0)
    nbr_std = np.sqrt(nbr_var + 1e-8)

    return nbr_mean.astype(np.float32), nbr_std.astype(np.float32), deg.astype(np.float32)

def laplacian_from_W(W):
    deg = np.array(W.sum(axis=1)).ravel()
    return (diags(deg) - W).tocoo(), deg

def butter_lowpass_filter(data, fs, cutoff=90.0, order=4):
    b, a = butter(order, cutoff/(fs/2.0), btype='low')
    return filtfilt(b, a, data, axis=-1)

def pr_re_f1(pred_mask, gt_mask):
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2*prec*rec / (prec+rec + 1e-12)
    return prec, rec, f1

def show_cortex_mask(coords, mask, title="", save=False, outdir="/content/sample_data/figs/", fname="mask.png"):
    """
    3D scatter of cortex with highlighted silent nodes.
    """
    c = np.where(mask, 1.0, 0.1)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=c, s=8, cmap='cool')
    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()
    show_or_save(fig, fname, save=save, outdir=outdir)


def build_torch_graph(W, deg):
    W = W.tocoo()
    idx = torch.tensor(np.vstack([W.row, W.col]).astype(np.int64), device=device)
    di = 1.0 / torch.sqrt(torch.clamp(torch.tensor(deg, dtype=torch.float32, device=device), min=1e-8))
    v  = torch.tensor(W.data.astype(np.float32), device=device)
    v  = v * di[idx[0]] * di[idx[1]]  # normalized adjacency Ahat
    Ahat = torch.sparse_coo_tensor(idx, v, W.shape, device=device).coalesce()
    return Ahat

def lap_energy(g, Lcoo):
    # g: (p,1)
    i = torch.tensor(np.vstack([Lcoo.row, Lcoo.col]).astype(np.int64), device=device)
    v = torch.tensor(Lcoo.data.astype(np.float32), device=device)
    L_t = torch.sparse_coo_tensor(i, v, Lcoo.shape).coalesce()
    Lg = torch.sparse.mm(L_t, g)
    return (g * Lg).sum()

def normalize_beta_for_gnn(beta, X_act=None):
    beta = np.asarray(beta, dtype=np.float64)

    # min-max normalize
    beta = (beta - beta.min()) / (beta.max() - beta.min() + 1e-12)

    # ensure silent should be LOW beta (optional if you have GT in sim)
    if X_act is not None:
        if beta[X_act].mean() > beta[~X_act].mean():
            beta = 1.0 - beta

    return beta.astype(np.float32)

# ========================= GNN =========================
"""class BetaGNN(nn.Module):
    def __init__(self, p, hidden, Ahat):
        super().__init__()
        self.lin_in  = nn.Linear(3, hidden, bias=True)     # features: [β, β^2, degree]
        self.lin_mp1 = nn.Linear(hidden, hidden, bias=False)
        self.lin_mp2 = nn.Linear(hidden, hidden, bias=False)
        self.lin_out = nn.Linear(hidden, 1, bias=True)
        self.Ahat = Ahat

    def mp(self, H):
        AH  = torch.sparse.mm(self.Ahat, H)
        A2H = torch.sparse.mm(self.Ahat, AH)
        return AH, A2H

    def forward(self, beta, degree):
        x = torch.cat([beta, beta**2, degree], dim=1)           # (p,3)
        H = torch.relu(self.lin_in(x))
        AH, A2H = self.mp(H)
        H = torch.relu(self.lin_mp1(AH) + self.lin_mp2(A2H))
        logits = self.lin_out(H)   # logits for BCEWithLogits
        return logits
"""
class BetaGNN(nn.Module):
    def __init__(self, in_dim, hidden, Ahat):
        super().__init__()
        self.lin_in  = nn.Linear(in_dim, hidden, bias=True)
        self.lin_mp1 = nn.Linear(hidden, hidden, bias=False)
        self.lin_mp2 = nn.Linear(hidden, hidden, bias=False)
        self.lin_out = nn.Linear(hidden, 1, bias=True)
        self.Ahat = Ahat

    def mp(self, H):
        AH  = torch.sparse.mm(self.Ahat, H)
        A2H = torch.sparse.mm(self.Ahat, AH)
        return AH, A2H

    def forward(self, X):
        H = torch.relu(self.lin_in(X))
        AH, A2H = self.mp(H)
        H = torch.relu(self.lin_mp1(AH) + self.lin_mp2(A2H))
        logits = self.lin_out(H)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# ========================= Main pipeline =========================
def main():
    parser = argparse.ArgumentParser(description="SilenceMap + GNN (single file)")
    parser.add_argument("--use_mat", action="store_true", help="Load real leadfield/headmodel from .mat files")
    parser.add_argument("--leadfield_path", type=str, default="/content/sample_data/OT_leadfield_symmetric_1662-128.mat")
    parser.add_argument("--headmodel_path", type=str, default="/content/sample_data/OT_headmodel_symmetric_1662-128.mat")
    parser.add_argument("--p", type=int, default=1662, help="Number of sources (synthetic)")
    parser.add_argument("--n", type=int, default=128, help="Number of sensors (synthetic)")
    parser.add_argument("--K", type=int, default=5, help="Number of silent regions")
    parser.add_argument("--per_region_k", type=int, default=10, help="Nodes per region")
    parser.add_argument("--t", type=int, default=10000, help="Time points")
    parser.add_argument("--Fs", type=int, default=512, help="Sampling rate")
    parser.add_argument("--kNN", type=int, default=12, help="Graph neighbors")
    parser.add_argument("--sigmaW", type=float, default=12.0, help="Graph RBF width (mm)")
    parser.add_argument("--lambda_lap", type=float, default=1.0, help="Laplacian smoother lambda")
    parser.add_argument("--q_silent", type=float, default=12.0, help="Silent percentile for masks")
    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_steps", type=int, default=2000)
    parser.add_argument("--gnn_lr", type=float, default=1e-2)
    parser.add_argument("--gnn_lambda", type=float, default=5.0, help="Smooth term weight")
    parser.add_argument("--gnn_gamma", type=float, default=0.5, help="Seed term weight")
    parser.add_argument("--save_figs", action="store_true", help="Save figures instead of showing")
    parser.add_argument("--fig_dir", type=str, default="/content/sample_data/figs/", help="Where to save figures")
    parser.add_argument("--leadfield_var", type=str, default=None,
                    help="Variable path for leadfield (e.g., 'L' or 'leadfield.L')")
    parser.add_argument("--vertices_var", type=str, default=None,
                    help="Variable path for vertices (e.g., 'Cortex.Pial.vertices')")

    args = parser.parse_args()
    print("Device:", device)

    if args.use_mat:
      print("Loading .mat leadfield/headmodel...")
      try:
        L, src_xyz = load_leadfield_any(
            args.leadfield_path,
            args.headmodel_path,
            leadfield_var=args.leadfield_var,
            vertices_var=args.vertices_var
        )
        n, p = L.shape
        print("Loaded leadfield:", L.shape, "vertices:", src_xyz.shape)
      except Exception as e:
        print("Failed to parse .mat structures:", e)
        print("Falling back to synthetic.")
        args.use_mat = False
        
    # -------- Simulate multi-region silence + EEG --------
    rng = np.random.default_rng(42)
    K, per_region_k = args.K, args.per_region_k
    t, Fs = args.t, args.Fs

    # disallow midline band (optional)
    mid_gap_mm = 5.0
    allowed = np.where((src_xyz[:,0] <= -mid_gap_mm) | (src_xyz[:,0] >= mid_gap_mm))[0]

    silence_idx = []
    available = set(allowed.tolist())
    for _ in range(K):
        if not available: break
        center = rng.choice(list(available))
        d2 = np.sum((src_xyz - src_xyz[center])**2, axis=1)
        order = np.argsort(d2)
        pick = [i for i in order if i in available][:per_region_k]
        silence_idx.extend(pick)
        for i in pick:
            if i in available: available.remove(i)
    silence_idx = np.unique(silence_idx)
    X_act = np.zeros(src_xyz.shape[0], dtype=bool); X_act[silence_idx] = True

    # source covariance (exp decay)
    gamma = 0.12
    d = cdist(src_xyz, src_xyz)
    Cs_full = np.exp(-gamma * d).astype(np.float32)

    S = rng.multivariate_normal(mean=np.zeros(src_xyz.shape[0]), cov=Cs_full, size=t).T
    S[X_act,:] = 0.0
    Noise_pow = 5e-8
    E = rng.normal(0, np.sqrt(Noise_pow), size=(L.shape[0], t)).astype(np.float32)

    eeg_clean = L @ S
    eeg_lp = butter_lowpass_filter(eeg_clean, fs=Fs, cutoff=90.0, order=4)
    eeg = eeg_lp + E

    # SNR sanity
    eeg_nosil = L @ rng.multivariate_normal(mean=np.zeros(src_xyz.shape[0]), cov=Cs_full, size=t).T
    eeg_nosil = butter_lowpass_filter(eeg_nosil, fs=Fs, cutoff=90.0, order=4)
    snr = np.mean(10*np.log10(np.var(eeg_nosil,axis=1)/(Noise_pow+1e-12)))
    print(f"Avg SNR ≈ {snr:.2f} dB")
    """i_ref0 = 95 - 1
    beta = compute_betta_silencemap_low(eeg, L, Fs=512, i_ref0=i_ref0)
    """
    #Cs_I = np.eye(src_xyz.shape[0], dtype=np.float64)
    #eta_raw = silencemap_beta_highres(eeg=eeg, L=L, Cs=Cs_I, Fs=Fs, i_ref0=63)

    beta_raw = silencemap_beta_highres(eeg=eeg, L=L, Cs=Cs_full, Fs=Fs, i_ref0=64)
    beta = normalize_beta_for_gnn(beta_raw, X_act=X_act)

    #beta = torch.tensor(beta, dtype=torch.float32, device=device).view(-1,1)
    #beta = beta.detach().to(device).float().view(-1, 1) if torch.is_tensor(beta) \
     #    else torch.tensor(beta, dtype=torch.float32, device=device).view(-1, 1)
    
    #beta = compute_betta_silencemap_low(eeg, L, Fs=512, i_ref0=i_ref_tot[ref_ind]-1)

    print(beta)
    print(beta.shape)
  
    #beta = np.ones(p, dtype=np.float32)
    #beta[X_act] = 0.0
        # -------- Graph + Laplacian smoother baseline --------
    kNN, sigmaW = args.kNN, args.sigmaW
    W, deg = knn_graph_gauss(src_xyz, k=kNN, sigma=sigmaW)
    L_g, _ = laplacian_from_W(W)
    lam = args.lambda_lap
    I = identity(src_xyz.shape[0], format='coo')
    A = (I + lam * L_g).tocsc()
    g_lap = spsolve(A, beta).astype(np.float32)
    g_lap -= g_lap.min()
    g_lap /= (g_lap.max() + 1e-12)
    ##################Feature building###########################
    # ----- Feature construction (do NOT change beta computation) -----
    #beta = beta.astype(np.float32).reshape(-1)

    # z-score is more stable than min-max for real data transfer
    beta_z = (beta - beta.mean()) / (beta.std() + 1e-8)

    nbr_mean, nbr_std, deg2 = neighbor_stats_from_W(W, beta_z)
    delta = beta_z - nbr_mean

    # Optional but recommended: geometry priors
    xyz = src_xyz.astype(np.float32)
    xyz = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-8)

    # log-degree helps numerics
    logdeg = np.log(deg2 + 1.0).astype(np.float32)

    # Final node feature matrix
    X = np.column_stack([beta_z, delta, nbr_std, logdeg]).astype(np.float32)
    # dims = 1 + 1 + 1 + 1 + 3 = 7 features, adding xyz above is optional

    #############################################################
    # number of truly silent nodes (ground truth)
    k_silent = int(X_act.sum())          # or int(round(K * per_region_k))
    q_silent = 100.0 * k_silent / p
    print(f"[auto] q_silent set to {q_silent:.2f}% for |S|={k_silent}/{p}")

    # ---- Laplacian mask: 
    idx_lap = np.argpartition(g_lap, k_silent - 1)[:k_silent]
    mask_lap = np.zeros_like(g_lap, dtype=bool)
    mask_lap[idx_lap] = True
    #P, R, F1 = pr_re_f1(mask_lap, X_act)
    #print(f"Laplacian: P={P:.3f} R={R:.3f} F1={F1:.3f}")

    # -------- GNN (supervised on binary GT X_act; input is non-binary beta_in) --------
    Ahat = build_torch_graph(W, deg)
    Lcoo = L_g.tocoo()

    k_silent = int(X_act.sum())
    y_bin = X_act.astype(np.float32)   # 1 = silent, 0 = active

    hidden   = args.gnn_hidden
    steps    = args.gnn_steps
    lr       = args.gnn_lr
    lam_gnn  = args.gnn_lambda

    #model = BetaGNN(p=src_xyz.shape[0], hidden=hidden, Ahat=Ahat).to(device)
    model = BetaGNN(in_dim=X.shape[1], hidden=hidden, Ahat=Ahat).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)

    """beta_t   = torch.tensor(beta, dtype=torch.float32, device=device).view(-1, 1)
    deg_feat = torch.tensor(deg,     dtype=torch.float32, device=device).view(-1, 1)
    y_t      = torch.tensor(y_bin,   dtype=torch.float32, device=device).view(-1, 1)
    """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)         # (p, in_dim)
    y_t = torch.tensor(y_bin, dtype=torch.float32, device=device).view(-1, 1)

    # class imbalance handling
    pos_weight = torch.tensor([(len(y_bin) - k_silent) / (k_silent + 1e-12)], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for it in range(steps):
        opt.zero_grad()
        #logits = model(beta_t, deg_feat)                 # (p,1)
        logits = model(X_t)
        prob   = torch.sigmoid(logits)                   # (p,1) in [0,1]

        cls_term = bce(logits, y_t)
        smooth_term = lap_energy(prob, Lcoo) / src_xyz.shape[0]

        loss = cls_term + lam_gnn * smooth_term
        loss.backward()
        opt.step()

        if (it + 1) % 200 == 0:
            print(f"[{it+1:04d}] loss={loss.item():.5f} cls={cls_term.item():.5f} smooth={smooth_term.item():.5f}")

    # ---- Evaluate F1@k (top-k predicted silent) ----
    with torch.no_grad():
        #logits = model(beta_t, deg_feat).squeeze(1).detach().cpu().numpy()
        logits = model(X_t).squeeze(1).detach().cpu().numpy()
        #prob = 1.0 / (1.0 + np.exp(-logits))
        prob = torch.sigmoid(torch.tensor(logits)).numpy()
    idx_pred = np.argpartition(prob, -k_silent)[-k_silent:]
    mask_gnn = np.zeros_like(prob, dtype=bool)
    mask_gnn[idx_pred] = True

    P, R, F1 = pr_re_f1(mask_gnn, X_act)
    print(f"GNN(BCE):  P={P:.3f} R={R:.3f} F1={F1:.3f}")
    mask_gnn = enforce_connected_topk(W, prob, k_silent, max_comps=K)

    #mask_gnn = grow_k_regions(W, prob, K=K, k_total=k_silent)

    m = component_metrics(W, X_act, mask_gnn)
    print_component_report("GNN", m)


    def stats(name, arr):
        arr = np.asarray(arr).reshape(-1)
        gt  = X_act.reshape(-1).astype(bool)
        print(f"\n[{name}]")
        print("  mean(silent):", float(arr[gt].mean()))
        print("  mean(active):", float(arr[~gt].mean()))
        if np.std(arr) > 1e-12:
            print("  corr with X_act:", float(np.corrcoef(arr, gt.astype(np.float32))[0, 1]))
    ##################Saving#########################
    # ================= SAVE EVERYTHING (RUN THIS ONCE) =================
    import json, time
    from pathlib import Path

    SAVE_DIR = "/content/sample_data/safe_save_run1"
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # 1) Save trained model weights
    torch.save(model.state_dict(), f"{SAVE_DIR}/model_state.pt")

    # 2) Save config needed to reload model safely
    config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "in_dim": int(X.shape[1]),
        "hidden": int(args.gnn_hidden),
        "kNN": int(args.kNN),
        "sigmaW": float(args.sigmaW),
        "gnn_lambda": float(args.gnn_lambda),
        "features": "beta_z, delta, nbr_std, logdeg (NO xyz)",
        "note": "Ready for NHP monkey data"}
        
    with open(f"{SAVE_DIR}/config.json", "w") as f:
         json.dump(config, f, indent=2)

    # 3) Save one full snapshot for sanity/debug
    np.save(f"{SAVE_DIR}/src_xyz.npy", src_xyz.astype(np.float32))
    np.save(f"{SAVE_DIR}/beta.npy", beta.astype(np.float32))
    np.save(f"{SAVE_DIR}/prob.npy", prob.astype(np.float32))
    np.save(f"{SAVE_DIR}/mask.npy", mask_gnn.astype(np.uint8))

    # 4) Save graph (W) safely
    Wcoo = W.tocoo()
    np.save(f"{SAVE_DIR}/W_row.npy", Wcoo.row.astype(np.int32))
    np.save(f"{SAVE_DIR}/W_col.npy", Wcoo.col.astype(np.int32))
    np.save(f"{SAVE_DIR}/W_data.npy", Wcoo.data.astype(np.float32))
    np.save(f"{SAVE_DIR}/W_shape.npy", np.array(Wcoo.shape, dtype=np.int32))

    print("\n✅ EVERYTHING SAVED SAFELY")
    print("📁 Folder:", SAVE_DIR)
    print("➡️ You can now move to MONKEY (NHP) data using these weights.")
    # ================================================================

    # -------- Plots --------
    save = args.save_figs
    out  = args.fig_dir

    # 3D masks
    show_cortex_mask(src_xyz, X_act,    title=f"GT silent (K={K})",             save=save, outdir=out, fname="1_gt.png")
    show_cortex_mask(src_xyz, mask_gnn, title=f"GNN mask (q=%s%%)" % q_silent,       save=save, outdir=out, fname="3_gnn.png")

if __name__ == "__main__":
    main()
