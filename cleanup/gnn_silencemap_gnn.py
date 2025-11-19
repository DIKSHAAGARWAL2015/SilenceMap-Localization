#!/usr/bin/env python3
# gnn_silencemap.py
# Multi-region Silence simulation + Laplacian baseline + Self-supervised GNN on a single beta
# Colab-ready single-file script

import os, sys, math, time, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # ensures 3D plotting works
import numpy as np
import mat73
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
#from scipy.signal import butter, filtfilt
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
from itertools import combinations
from scipy.sparse.csgraph import connected_components


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def extract_clusters_from_mask_sparse(mask, W):
    """
    Given:
        mask : boolean array of shape (p,)
            True for nodes selected by GNN (e.g. mask_gnn).
        W : (p,p) sparse adjacency (same W you built from knn_graph_gauss).

    Returns:
        clusters : list of lists
            Each inner list is a list of original node indices in that cluster.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]        # original node indices that are True

    if idx.size == 0:
        return []

    # coo_matrix is not subscriptable → convert to CSR first
    W_csr = W.tocsr()

    # Induced subgraph on masked nodes
    W_sub = W_csr[mask][:, mask]

    # Connected components on this subgraph
    n_comp, labels = connected_components(csgraph=W_sub,
                                          directed=False,
                                          connection='weak')
    clusters = []
    for c in range(n_comp):
        members = idx[labels == c]
        if members.size > 0:
            clusters.append(members.tolist())
    return clusters


def rank_clusters(clusters, coords):
    """
    clusters : list of lists of node indices
    coords   : (p,3) array of xyz coordinates (here: src_xyz)

    Returns:
        ranked list of dicts with size, mean distance, score, etc.
    """
    cluster_info = []

    for c_id, cluster in enumerate(clusters):
        nodes = np.array(cluster, dtype=int)
        points = coords[nodes]           # (k,3)
        size = len(nodes)

        if size > 1:
            # pairwise distances inside cluster
            dists = []
            for i, j in combinations(range(size), 2):
                dists.append(np.linalg.norm(points[i] - points[j]))
            mean_dist = float(np.mean(dists))
            max_dist  = float(np.max(dists))
        else:
            mean_dist = 0.0
            max_dist  = 0.0

        # Bigger & tighter cluster → higher score (you can change formula)
        score = size - mean_dist

        cluster_info.append({
            "cluster_id": c_id,
            "nodes": nodes.tolist(),
            "size": size,
            "mean_internal_distance": mean_dist,
            "max_internal_distance": max_dist,
            "score": score,
        })

    ranked = sorted(cluster_info, key=lambda x: x["score"], reverse=True)
    return ranked
def plot_ranked_clusters_numbered_with_mask(coords, mask, ranked_node_groups,
                                            title="Ranked clusters (1 = best)",
                                            save=False, outdir="/content/",
                                            fname="gnn_ranked_clusters.png"):
    """
    Separate figure:
      - background nodes: soft blue
      - detected silent nodes (mask=True): red
      - cluster numbers = rank IDs (1,2,3,...) drawn at centroids
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    coords = np.asarray(coords)
    mask   = np.asarray(mask, dtype=bool)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # ---- Background: soft blue ----
    ax.scatter(coords[:,0], coords[:,1], coords[:,2],
               c="#b0c4ff", s=8, edgecolors='none', alpha=0.4)

    # ---- Silent nodes: red ----
    silent_idx = np.where(mask)[0]
    ax.scatter(coords[silent_idx,0], coords[silent_idx,1], coords[silent_idx,2],
               c="#ff4444", s=18, edgecolors='none', alpha=0.9)

    # ---- Rank numbers at cluster centroids ----
    for rank_id, nodes in ranked_node_groups:
        nodes_arr = np.array(nodes, dtype=int)
        centroid = coords[nodes_arr].mean(axis=0)
        ax.text(
            centroid[0], centroid[1], centroid[2],
            str(rank_id),                # RANK number
            color="black",
            fontsize=12,
            ha='center', va='center',
            weight='bold'
        )

    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()

    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved → {path}]")
    else:
        plt.show()

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

# ========================= GNN =========================
class BetaGNN(nn.Module):
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
        g = torch.nn.functional.softplus(self.lin_out(H))        # ≥0
        #g = (g - g.min()) / (g.max() - g.min() + 1e-8)           # normalize
        return g
    
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
    
    ## -------- Compute beta from EEG --------
    """Ceeg = (eeg @ eeg.T) / float(t)        # (n,n)
    AtA  = L.T @ Ceeg @ L                  # (p,p)
    beta_eeg = np.diag(AtA).astype(np.float32)
    # assuming beta is not completely constant
    beta_eeg = beta_eeg.astype(np.float64)
    beta_oracle = np.ones(p, dtype=np.float32)
    beta_oracle[X_act] = 0.0
    alpha = 0.0  # 1.0 = pure oracle, 0.0 = pure EEG
    beta = (1 - alpha) * beta_eeg + alpha * beta_oracle
    beta -= beta.min()
    beta /= (beta.max() + 1e-12)"""
    
    #binary beta
    """mat = loadmat('/content/sample_data/silencemap_beta.mat')   # adjust path if needed

    beta_mat = mat['Betta'].squeeze()      # (p,) or (p,1) → (p,)
    #X_act_mat = mat['X_act'].squeeze()     # optional, but nice to compare
    # silence_indices = mat['silence_indices'].squeeze()  # if you stored these
    beta = beta_mat.astype(np.float32)
    # Normalize like in Python pipeline
    beta -= beta.min()
    beta /= (beta.max() + 1e-12)
    silent = X_act.astype(bool)
    active = ~silent
    mean_silent = float(beta[silent].mean())
    mean_active = float(beta[active].mean())
    print("mean(beta silent) :", mean_silent)
    print("mean(beta active) :", mean_active)

    # If silent nodes have larger beta than active ones, flip the scale
    if mean_silent > mean_active:
        print(">> Flipping beta: making silent = low values")
        beta = 1.0 - beta

    # If you want to sanity check:
    print("beta shape:", beta.shape)
    print("beta min/max:", beta.min(), beta.max())
    #print("mean(beta silent):", float(beta[X_act_mat == 1].mean()))
    #print("mean(beta active):", float(beta[X_act_mat == 0].mean()))
    #print("corr with X_act:", float(np.corrcoef(beta, X_act_mat.astype(float))[0,1]))
    """
    beta = np.ones(p, dtype=np.float32)
    beta[X_act] = 0.0
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

    # number of truly silent nodes (ground truth)
    k_silent = int(X_act.sum())          # or int(round(K * per_region_k))
    q_silent = 100.0 * k_silent / p
    print(f"[auto] q_silent set to {q_silent:.2f}% for |S|={k_silent}/{p}")

    # ---- Laplacian mask: choose k_silent *smallest* g_lap values as silent ----
    g_lap_arr = np.asarray(g_lap)
    idx_lap = np.argpartition(g_lap_arr, k_silent-1)[:k_silent]  # indices of k smallest
    mask_lap = np.zeros_like(g_lap_arr, dtype=bool)
    mask_lap[idx_lap] = True

    P, R, F1 = pr_re_f1(mask_lap, X_act)
    print(f"Laplacian: P={P:.3f} R={R:.3f} F1={F1:.3f}")

    # -------- GNN (self-supervised on single beta) --------
    Ahat = build_torch_graph(W, deg)
    Lcoo = L_g.tocoo()

    # seeds: also pick the k_silent smallest beta values as "most silent-looking"
    beta_arr = np.asarray(beta)
    """idx_seed = np.argpartition(beta_arr, k_silent-1)[:k_silent]
    seed_mask = torch.zeros(beta_arr.shape[0], dtype=torch.bool, device=device)
    seed_mask[idx_seed] = True"""
    # ---- NEW SEED SELECTION (replace old argpartition version) ----

    q_seed = 0.03   # 3% seeds (you can adjust)
    k_seed = max(5, int(q_seed * len(beta_arr)))   # ensure minimum 5 seeds

    # Sort beta: silent ~ 0, active ~ 1
    idx_sorted = np.argsort(beta_arr)

    # Lowest β → silent seeds
    silent_seeds = idx_sorted[:k_seed]

    # Highest β → active seeds
    active_seeds = idx_sorted[-k_seed:]

    # Create masks
    seed_silent_mask = torch.zeros(beta_arr.shape[0], dtype=torch.bool, device=device)
    seed_active_mask = torch.zeros(beta_arr.shape[0], dtype=torch.bool, device=device)

    seed_silent_mask[silent_seeds] = True
    seed_active_mask[active_seeds] = True


    hidden   = args.gnn_hidden
    steps    = args.gnn_steps
    lr       = args.gnn_lr
    lam_gnn  = args.gnn_lambda
    gamma_gnn = args.gnn_gamma

    model = BetaGNN(p=src_xyz.shape[0], hidden=hidden, Ahat=Ahat).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    beta_t   = torch.tensor(beta, dtype=torch.float32, device=device).view(-1, 1)
    deg_feat = torch.tensor(deg,  dtype=torch.float32, device=device).view(-1, 1)

    for it in range(steps):
        opt.zero_grad()
        g = model(beta_t, deg_feat)          # (p,1)

        data_term   = ((g - beta_t)**2).mean()
        smooth_term = lap_energy(g, Lcoo) / src_xyz.shape[0]
        #seed_term   = g[seed_mask].mean()
        loss_silent = torch.mean(g[seed_silent_mask]**2)          # g -> 0
        loss_active = torch.mean((g[seed_active_mask] - 1)**2)    # g -> 1
        loss_seed = loss_silent + loss_active

        loss = data_term + lam_gnn * smooth_term + gamma_gnn * loss_seed

        loss.backward()
        opt.step()
        if (it+1) % 200 == 0:
            print(f"[{it+1:04d}] loss={loss.item():.5f} "
                  f"data={data_term.item():.5f} "
                  f"smooth={smooth_term.item():.5f} "
                  f"seed={loss_seed.item():.5f}")

    with torch.no_grad():
        #g_hat = model(beta_t, deg_feat).squeeze(1).detach().cpu().numpy()
        g_raw = model(beta_t, deg_feat).squeeze(1).cpu().numpy()
        g_hat = (g_raw - g_raw.min()) / (g_raw.max() - g_raw.min() + 1e-12)
    # ---- GNN mask: choose k_silent *smallest* g_hat values as silent ----
    g_hat_arr = np.asarray(g_hat)
    idx_gnn = np.argpartition(g_hat_arr, k_silent-1)[:k_silent]
    mask_gnn = np.zeros_like(g_hat_arr, dtype=bool)
    mask_gnn[idx_gnn] = True

    P, R, F1 = pr_re_f1(mask_gnn, X_act)
    print(f"GNN:       P={P:.3f} R={R:.3f} F1={F1:.3f}")
        # -------- Cluster extraction + ranking from GNN mask --------
    clusters = extract_clusters_from_mask_sparse(mask_gnn, W)
    print(f"Found {len(clusters)} GNN clusters among silent nodes.")

    ranked_clusters = rank_clusters(clusters, src_xyz)
    # Build a mapping: cluster_nodes → ranked cluster ID
    ranked_node_groups = []        # list of (rank_id, nodes)
    for rank_id, info in enumerate(ranked_clusters, start=1):
         nodes = info["nodes"]       # already a list of node indices
         ranked_node_groups.append((rank_id, nodes))

    # Print top few clusters
    print("\nTop clusters (by size & compactness):")
    for c in ranked_clusters[:5]:
        print(
            f"  Cluster {c['cluster_id']}: "
            f"size={c['size']}, "
            f"mean_dist={c['mean_internal_distance']:.3f}, "
            f"score={c['score']:.3f}"
        )

        # ---- DEBUG: how aligned are beta, g_lap, g_gnn with true silence? ----
    silent = X_act.astype(bool)
    active = ~silent

    def stats(name, arr):
        arr = np.asarray(arr)
        print(f"\n[{name}]")
        print("  mean(silent) :", float(arr[silent].mean()))
        print("  mean(active) :", float(arr[active].mean()))
        print("  corr with X_act:",
              float(np.corrcoef(arr, X_act.astype(np.float32))[0,1]))

    stats("beta",  beta)
    stats("g_lap", g_lap)
    stats("g_gnn", g_hat)

    # -------- Plots --------
    save = args.save_figs
    out  = args.fig_dir

    # 3D masks
    show_cortex_mask(src_xyz, X_act,    title=f"GT silent (K={K})",             save=save, outdir=out, fname="1_gt.png")
    show_cortex_mask(src_xyz, mask_lap, title=f"Laplacian mask (q=%s%%)" % q_silent, save=save, outdir=out, fname="2_laplacian.png")
    show_cortex_mask(src_xyz, mask_gnn, title=f"GNN mask (q=%s%%)" % q_silent,       save=save, outdir=out, fname="3_gnn.png")
    plot_ranked_clusters_numbered_with_mask(
    src_xyz,
    mask_gnn,
    ranked_node_groups,
    title="GNN silent regions (ranked clusters)",
    save=args.save_figs,
    outdir=args.fig_dir,
    fname="7_gnn_ranked_clusters.png")
    # β / g curves
    fig = plt.figure(figsize=(6,3))
    plt.plot(beta[:300], label='beta')
    plt.plot(g_lap[:300], label='g_lap')
    plt.plot(g_hat[:300], label='g_gnn')
    plt.legend(); plt.title('First 300 nodes')
    show_or_save(fig, "4_curves.png", save=save, outdir=out)


if __name__ == "__main__":
    main()
