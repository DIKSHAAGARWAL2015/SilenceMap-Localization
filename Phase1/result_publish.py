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
from pathlib import Path
import os
import pathlib
import matplotlib.pyplot as plt


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
    Robust loader:
      1) Try mat73 (true MATLAB v7.3 files)
      2) If that fails, try h5py (plain HDF5 datasets: L, src_xyz, sensor_locs)
      3) If that fails, try scipy.io.loadmat (MATLAB <= v7)
    Returns:
      L: (n_eeg, p_vertices)
      src_xyz: (p_vertices, 3)
    """
    import numpy as np

    def _resolve_dotted(d, dotted):
        cur = d
        for part in dotted.split("."):
            cur = cur[part]
        return cur

    def _normalize_vertices(V):
        V = np.array(V).astype(np.float32)
        if V.ndim != 2:
            V = V.reshape(V.shape[0], -1)
        if V.shape[1] == 3:
            return V
        if V.shape[0] == 3:
            return V.T
        raise RuntimeError(f"Vertices shape {V.shape} not (p,3) or (3,p).")

    def _normalize_L(L, pV):
        L = np.array(L).astype(np.float32)
        if L.ndim != 2:
            L = L.reshape(L.shape[0], -1)
        nL, pL = L.shape
        if pL != pV:
            if nL == pV:
                L = L.T
            else:
                raise RuntimeError(f"Dimension mismatch: L {L.shape}, vertices {pV}x3")
        return L

    # -------------------------
    # 1) Try mat73
    # -------------------------
    try:
        import mat73
        # Leadfield
        lead = mat73.loadmat(leadfield_path)
        if leadfield_var:
            L = _resolve_dotted(lead, leadfield_var)
        else:
            items = _walk_py(lead)
            L = _pick_leadfield_from_py(items)
            if L is None:
                raise RuntimeError("Could not find leadfield matrix L in file (mat73).")

        # Vertices from headmodel
        head = mat73.loadmat(headmodel_path)
        if vertices_var:
            V = _resolve_dotted(head, vertices_var)
        else:
            items = _walk_py(head)
            V = _pick_vertices_from_py(items)
            if V is None:
                raise RuntimeError("Could not find cortex vertices in file (mat73).")

        src_xyz = _normalize_vertices(V)
        L = _normalize_L(L, src_xyz.shape[0])
        return L.astype(np.float32), src_xyz.astype(np.float32)

    except Exception as e_mat73:
        # fall through to other loaders
        pass

    # -------------------------
    # 2) Try h5py (plain HDF5 datasets)
    # -------------------------
    try:
        import h5py
        with h5py.File(leadfield_path, "r") as f:
            # Expect these keys in your saved HDF5 file
            if "L" in f:
                L = np.array(f["L"])
            else:
                raise KeyError("HDF5 leadfield file missing dataset 'L'")

            # Prefer src_xyz from the same file if present
            if "src_xyz" in f:
                V = np.array(f["src_xyz"])
                src_xyz = _normalize_vertices(V)
                L = _normalize_L(L, src_xyz.shape[0])
                return L.astype(np.float32), src_xyz.astype(np.float32)

        # If leadfield file doesn't contain src_xyz, read vertices from headmodel_path
        with h5py.File(headmodel_path, "r") as f:
            # Try common layouts
            if "src_xyz" in f:
                V = np.array(f["src_xyz"])
            elif "vertices" in f:
                V = np.array(f["vertices"])
            elif "Cortex" in f and "vertices" in f["Cortex"]:
                V = np.array(f["Cortex"]["vertices"])
            elif "DownsampledCortex" in f and "V" in f["DownsampledCortex"]:
                V = np.array(f["DownsampledCortex"]["V"])
            else:
                raise KeyError("Could not find vertices in HDF5 headmodel file")

            src_xyz = _normalize_vertices(V)
            L = _normalize_L(L, src_xyz.shape[0])
            return L.astype(np.float32), src_xyz.astype(np.float32)

    except Exception as e_h5:
        pass

    # -------------------------
    # 3) Try scipy.io.loadmat (MATLAB <= v7)
    # -------------------------
    try:
        import scipy.io as sio
        lead = sio.loadmat(leadfield_path)
        if leadfield_var and leadfield_var in lead:
            L = lead[leadfield_var]
        else:
            L = lead.get("L", None)
        if L is None:
            raise KeyError("Could not find L in scipy loadmat")

        head = sio.loadmat(headmodel_path)
        # try explicit vertices_var if provided
        if vertices_var and vertices_var in head:
            V = head[vertices_var]
        else:
            # common keys
            V = head.get("src_xyz", None) or head.get("vertices", None)

            # try Cortex.vertices style if present
            if V is None and "Cortex" in head:
                C = head["Cortex"]
                try:
                    V = C["vertices"][0, 0]
                except Exception:
                    pass

            # try DownsampledCortex.V
            if V is None and "DownsampledCortex" in head:
                D = head["DownsampledCortex"]
                try:
                    V = D["V"][0, 0]
                except Exception:
                    pass

        if V is None:
            raise KeyError("Could not find vertices in scipy loadmat")

        src_xyz = _normalize_vertices(V)
        L = _normalize_L(L, src_xyz.shape[0])
        return L.astype(np.float32), src_xyz.astype(np.float32)

    except Exception as e_sio:
        raise RuntimeError(
            f"Could not load leadfield/headmodel via mat73, h5py, or scipy.io.loadmat.\n"
            f"Last error: {e_sio}"
        )

def show_overlay_mask(coords, mask, title="", save=False,
                      outdir="/content/sample_data/figs_monkey/",
                      fname="overlay.png"):
    """
    Plot entire cortex in gray, silent/active nodes highlighted in color.
    mask=True → silent region
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c="lightgray", s=6, alpha=0.3
    )

    silent_coords = coords[mask]
    ax.scatter(
        silent_coords[:, 0], silent_coords[:, 1], silent_coords[:, 2],
        c="red", s=12, alpha=1.0
    )

    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()
    show_or_save(fig, fname, save=save, outdir=outdir)

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
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    W = W.tocsr()
    sub = W[idx][:, idx]
    sub = ((sub + sub.T) > 0).astype(np.int8)

    n_comp, labels = connected_components(sub, directed=False)
    comps = [idx[labels == c] for c in range(n_comp)]
    return comps

def centroid(nodes, coords):
    return coords[nodes].mean(axis=0)

def jaccard_nodes(a, b):
    a = set(a); b = set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(a & b) / (len(a | b) + 1e-12)
# ---------- main evaluation ----------
def evaluate_silence_from_masks(W, gt_mask, pred_mask, coords):
    """
    Returns:
      mean_jaccard       : mean component-wise IoU
      mean_com_error     : mean centroid distance
      mean_size_error    : mean |pred_size - gt_size| / gt_size
    """

    gt_comps = mask_connected_components(W, gt_mask)
    pr_comps = mask_connected_components(W, pred_mask)

    if len(gt_comps) == 0 or len(pr_comps) == 0:
        return {
            "mean_jaccard": np.nan,
            "mean_com_error": np.nan,
            "mean_size_error": np.nan,
        }

    gt_cent = np.stack([centroid(c, coords) for c in gt_comps])
    pr_cent = np.stack([centroid(c, coords) for c in pr_comps])

    # ---------- greedy centroid matching ----------
    D = np.linalg.norm(gt_cent[:, None, :] - pr_cent[None, :, :], axis=2)

    used_gt = set()
    used_pr = set()
    pairs = []

    for gi, pj in sorted(
        [(i, j) for i in range(len(gt_comps)) for j in range(len(pr_comps))],
        key=lambda x: D[x[0], x[1]]
    ):
        if gi in used_gt or pj in used_pr:
            continue
        used_gt.add(gi)
        used_pr.add(pj)
        pairs.append((gi, pj))

    # ---------- metrics ----------
    jaccards = []
    com_errors = []
    size_errors = []

    for gi, pj in pairs:
        g = gt_comps[gi]
        p = pr_comps[pj]

        jaccards.append(jaccard_nodes(g, p))
        com_errors.append(np.linalg.norm(gt_cent[gi] - pr_cent[pj]))
        size_errors.append(abs(len(p) - len(g)) / (len(g) + 1e-12))

    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "mean_com_error": float(np.mean(com_errors)),
        "mean_size_error": float(np.mean(size_errors)),
    }
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

"""def normalize_beta_for_gnn(beta, X_act=None):
    beta = np.asarray(beta, dtype=np.float64)

    # min-max normalize
    beta = (beta - beta.min()) / (beta.max() - beta.min() + 1e-12)

    # ensure silent should be LOW beta (optional if you have GT in sim)
    if X_act is not None:
        if beta[X_act].mean() > beta[~X_act].mean():
            beta = 1.0 - beta

    return beta.astype(np.float32)"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

def save_paginated_gt_gnn_plots(plot_cases, save_dir, cases_per_page=5):
    save_dir.mkdir(parents=True, exist_ok=True)

    n_cases = len(plot_cases)
    n_pages = math.ceil(n_cases / cases_per_page)

    for page in range(n_pages):
        start = page * cases_per_page
        end   = min(start + cases_per_page, n_cases)
        page_cases = plot_cases[start:end]

        n_rows = len(page_cases)
        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(8, 2.5 * n_rows),
            subplot_kw={"projection": "3d"}
        )

        if n_rows == 1:
            axes = axes.reshape(1, 2)

        for i, case in enumerate(page_cases):
            xyz = case["src_xyz"]
            gt  = case["X_act"]
            pr  = case["pred_mask"]

            # -------- GT --------
            ax = axes[i, 0]
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
                       c="lightgray", s=1, alpha=0.15)
            ax.scatter(xyz[gt,0], xyz[gt,1], xyz[gt,2],
                       c="red", s=8)
            ax.set_title(
                f"GT | SNR={case['snr']} | K={case['K']} | per={case['per_region_k']}",
                fontsize=8
            )
            ax.set_axis_off()
            ax.set_box_aspect([1,1,1])

            # -------- GNN --------
            ax = axes[i, 1]
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
                       c="lightgray", s=1, alpha=0.15)
            ax.scatter(xyz[pr,0], xyz[pr,1], xyz[pr,2],
                       c="blue", s=8)
            ax.set_title("GNN", fontsize=8)
            ax.set_axis_off()
            ax.set_box_aspect([1,1,1])

        plt.tight_layout()
        out_path = save_dir / f"gt_vs_gnn_page_{page+1:02d}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📄 Saved page {page+1}/{n_pages}: {out_path}")

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

import csv
from pathlib import Path
def main():
    parser = argparse.ArgumentParser(description="SilenceMap + GNN (grid cases)")
    parser.add_argument("--leadfield_path", type=str, required=True)
    parser.add_argument("--headmodel_path", type=str, required=True)
    parser.add_argument("--leadfield_var", type=str, default=None)
    parser.add_argument("--vertices_var", type=str, default=None)
    parser.add_argument("--t", type=int, default=10000)
    parser.add_argument("--Fs", type=int, default=512)
    parser.add_argument("--kNN", type=int, default=12)
    parser.add_argument("--sigmaW", type=float, default=12.0)
    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_steps", type=int, default=2000)
    parser.add_argument("--gnn_lr", type=float, default=1e-2)
    parser.add_argument("--gnn_lambda", type=float, default=5.0)
    parser.add_argument("--num_cases", type=int, default=50)
    parser.add_argument("--save_figs", action="store_true", help="Save GT vs GNN figures")
    args = parser.parse_args()

    print("Device:", device)

    # ---------------- Load leadfield + cortex ----------------
    L, src_xyz = load_leadfield_any(
        args.leadfield_path, args.headmodel_path,
        leadfield_var=args.leadfield_var, vertices_var=args.vertices_var
    )
    n, p = L.shape
    print("Loaded L:", L.shape, "Vertices:", src_xyz.shape)

    # ---------------- Build graph ----------------
    W, deg = knn_graph_gauss(src_xyz, k=args.kNN, sigma=args.sigmaW)
    L_g, _ = laplacian_from_W(W)

    # ---------------- Define grid ----------------
    snrs  = [5.0, 10.0, 15.0, 20.0, 22.05]
    Ks    = [2, 4, 6]
    sizes = [8, 12, 16]
    grid  = [(snr, K, per_k) for snr in snrs for K in Ks for per_k in sizes]

    results = []
    plot_cases = []

    SAVE_DIR = pathlib.Path("/content/sample_data")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SAVE_DIR / "case_results.csv"

    # ---------------- CSV Header ----------------
    headers = [
        "case", "snr_target", "snr_measured", "K", "per_region_k",
        "F1", "mean_comp_jaccard", "com_error", "mean_size_error"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    # ---------------- Run cases ----------------
    for case_id in range(args.num_cases):
        target_snr_db, K, per_region_k = grid[case_id % len(grid)]
        rng = np.random.default_rng(2000 + case_id)
        print(f"\n=== CASE {case_id} | SNR={target_snr_db}, K={K}, per_k={per_region_k} ===")

        # ---------------- Generate silent regions ----------------
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
            for i in pick: available.remove(i)
        silence_idx = np.unique(np.asarray(silence_idx, dtype=np.int64))
        X_act = np.zeros(p, dtype=bool)
        X_act[silence_idx] = True

        # ---------------- EEG simulation ----------------
        gamma = 0.12
        d = cdist(src_xyz, src_xyz)
        Cs_full = np.exp(-gamma * d).astype(np.float32)
        S = rng.multivariate_normal(mean=np.zeros(p), cov=Cs_full, size=args.t).T
        S[X_act, :] = 0.0
        eeg_clean = L @ S
        eeg_lp = butter_lowpass_filter(eeg_clean, fs=args.Fs, cutoff=90.0)
        sig_pow = np.mean(eeg_lp**2)
        noise_pow = sig_pow / (10**(target_snr_db / 10))
        E = rng.standard_normal(eeg_lp.shape) * np.sqrt(noise_pow)
        eeg = eeg_lp + E
        snr_meas = 10 * np.log10(np.mean(eeg_lp**2) / (np.mean(E**2) + 1e-12))

        # ---------------- Compute Beta + Features ----------------
        beta_raw = silencemap_beta_highres(eeg, L, Cs_full, Fs=args.Fs, i_ref0=63)
        beta_z = (beta_raw - beta_raw.mean()) / (beta_raw.std() + 1e-8)
        nbr_mean, nbr_std, deg2 = neighbor_stats_from_W(W, beta_z)
        delta = beta_z - nbr_mean
        logdeg = np.log(deg2 + 1.0).astype(np.float32)
        X = np.column_stack([beta_z, delta, nbr_std, logdeg]).astype(np.float32)

        # ---------------- GNN Training ----------------
        k_silent = int(X_act.sum())
        Ahat = build_torch_graph(W, deg)
        Lcoo = L_g.tocoo()
        model = BetaGNN(X.shape[1], args.gnn_hidden, Ahat).to(device)
        opt = optim.Adam(model.parameters(), lr=args.gnn_lr)
        X_t = torch.tensor(X, device=device)
        y_t = torch.tensor(X_act.astype(np.float32), device=device).view(-1,1)
        pos_weight = torch.tensor([(p - k_silent)/(k_silent+1e-12)], device=device)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        for _ in range(args.gnn_steps):
            opt.zero_grad()
            logits = model(X_t)
            prob = torch.sigmoid(logits)
            loss = bce(logits, y_t) + args.gnn_lambda * lap_energy(prob, Lcoo)/p
            loss.backward()
            opt.step()

        # ---------------- Inference ----------------
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(X_t)).cpu().numpy().ravel()
        pred_mask = enforce_connected_topk(W, prob, k_silent, max_comps=K)

        # ---------------- Metrics ----------------
        metrics = evaluate_silence_from_masks(W, X_act, pred_mask, src_xyz)
        P, R, F1 = pr_re_f1(pred_mask, X_act)

        # ---------------- Save case row ----------------
        case_row = {
            "case": case_id,
            "snr_target": target_snr_db,
            "snr_measured": snr_meas,
            "K": K,
            "per_region_k": per_region_k,
            "F1": F1,
            "mean_comp_jaccard": metrics["mean_jaccard"],
            "com_error": metrics["mean_com_error"],
            "mean_size_error": metrics["mean_size_error"]
        }
        results.append(case_row)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(case_row)

        print(f"[CASE {case_id}] F1={F1:.3f} | "
              f"Mean IoU={case_row['mean_comp_jaccard']:.3f} | "
              f"COM err={case_row['com_error']:.3f} | "
              f"Size err={case_row['mean_size_error']:.3f}")

        # ---------------- Store masks for plotting ----------------
        plot_cases.append({
            "case": case_id,
            "X_act": X_act.copy(),
            "pred_mask": pred_mask.copy(),
            "src_xyz": src_xyz,  # same for all cases
            "snr": target_snr_db,
            "K": K,
            "per_region_k": per_region_k
        })

    # ---------------- Average metrics ----------------
    avg_row = {k: np.nanmean([r[k] for r in results if isinstance(r[k], (float,int))]) 
               for k in ["F1","mean_comp_jaccard","com_error","mean_size_error"]}
    avg_row.update({h:"" for h in ["case","snr_target","snr_measured","K","per_region_k"]})
    avg_row["case"] = "average"
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(avg_row)

    # ---------------- Save plots ----------------
    if args.save_figs:
        PLOT_DIR = SAVE_DIR / "gt_vs_gnn_pages"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        save_paginated_gt_gnn_plots(plot_cases=plot_cases, save_dir=PLOT_DIR, cases_per_page=5)


if __name__ == "__main__":
    main()
