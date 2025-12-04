#!/usr/bin/env python3
# gnn_silencemap_gnn.py
# Use FINAL SilenceMap beta + exact MATLAB silence indices to evaluate
# Laplacian smoothing and a self-supervised GNN on the SAME nodes.

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn
import torch.optim as optim

# For MATLAB v7.3 files
import mat73

# ==================== Device ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== Plot utilities ====================
def show_or_save(fig, name, save=False, outdir="./figs/"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if save:
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved figure → {path}]")
    else:
        plt.show()


def show_cortex_mask(coords, mask, title="", save=False, outdir="./figs/", fname="mask.png"):
    """
    3D scatter of cortex with highlighted silent nodes.
    coords: (p,3)
    mask:   (p,) bool – True = highlighted
    """
    c = np.where(mask, 1.0, 0.1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=c, s=8, cmap='cool')
    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()
    show_or_save(fig, fname, save=save, outdir=outdir)


# ==================== Graph utilities ====================
def knn_graph_gauss(coords, k=12, sigma=12.0):
    """
    Build a kNN graph with Gaussian edge weights on Euclidean distances.
    coords: (p,3)
    """
    p = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, p)).fit(coords)
    dists, idxs = nn.kneighbors(coords)
    rows, cols, vals = [], [], []
    for i in range(p):
        # skip self neighbor at idxs[i,0]
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            w = math.exp(-(d ** 2) / (2.0 * sigma ** 2))
            rows.append(i)
            cols.append(j)
            vals.append(w)

    # Symmetrize by max weight
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
    deg = np.array(W.sum(axis=1)).ravel().astype(np.float32)
    return W.tocoo(), deg


def laplacian_from_W(W):
    deg = np.array(W.sum(axis=1)).ravel()
    return (diags(deg) - W).tocoo(), deg


def build_torch_graph(W, deg):
    W = W.tocoo()
    idx = torch.tensor(
        np.vstack([W.row, W.col]).astype(np.int64),
        device=device
    )
    di = 1.0 / torch.sqrt(
        torch.clamp(torch.tensor(deg, dtype=torch.float32, device=device), min=1e-8)
    )
    v = torch.tensor(W.data.astype(np.float32), device=device)
    v = v * di[idx[0]] * di[idx[1]]  # normalized adjacency Â
    Ahat = torch.sparse_coo_tensor(idx, v, W.shape, device=device).coalesce()
    return Ahat


def lap_energy(g, Lcoo):
    """
    g: (p,1) torch
    Lcoo: scipy.sparse.coo_matrix Laplacian
    """
    i = torch.tensor(
        np.vstack([Lcoo.row, Lcoo.col]).astype(np.int64),
        device=device
    )
    v = torch.tensor(Lcoo.data.astype(np.float32), device=device)
    L_t = torch.sparse_coo_tensor(i, v, Lcoo.shape, device=device).coalesce()
    Lg = torch.sparse.mm(L_t, g)
    return (g * Lg).sum()


# ==================== Metrics ====================
def pr_re_f1(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: bool arrays of shape (p,)
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


# ==================== GNN model ====================
class BetaGNN(nn.Module):
    """
    Simple 2-hop message-passing GNN that smooths a scalar field beta on the graph.
    Input features per node: [beta, beta^2, degree].
    """

    def __init__(self, p, hidden, Ahat):
        super().__init__()
        self.lin_in = nn.Linear(3, hidden, bias=True)
        self.lin_mp1 = nn.Linear(hidden, hidden, bias=False)
        self.lin_mp2 = nn.Linear(hidden, hidden, bias=False)
        self.lin_out = nn.Linear(hidden, 1, bias=True)
        self.Ahat = Ahat

    def mp(self, H):
        AH = torch.sparse.mm(self.Ahat, H)
        A2H = torch.sparse.mm(self.Ahat, AH)
        return AH, A2H

    def forward(self, beta, degree):
        # beta, degree: (p,1)
        x = torch.cat([beta, beta ** 2, degree], dim=1)  # (p,3)
        H = torch.relu(self.lin_in(x))
        AH, A2H = self.mp(H)
        H = torch.relu(self.lin_mp1(AH) + self.lin_mp2(A2H))
        g = torch.nn.functional.softplus(self.lin_out(H))  # ≥0
        return g


# ==================== Load SilenceMap export ====================
def load_silencemap_export(path):
    """
    Load silencemap_export struct from MATLAB .mat (v7.3 via mat73).
    Expected fields:
      beta            : [p,] final Betta
      all_idx         : [p,] 1-based indices (1..p)
      gt_silence_idx  : [k_gt,] 1-based indices for ground-truth silent nodes
      det_silence_idx : [k_det,] 1-based indices for detected silent nodes
      src_loc         : [p,3] coordinates
    """
    print(f"Loading SilenceMap export from: {path}")
    mat = mat73.loadmat(path)
    if "silencemap_export" not in mat:
        raise RuntimeError("silencemap_export not found in .mat file.")
    smap = mat["silencemap_export"]

    beta = np.asarray(smap["beta"]).astype(np.float32).ravel()
    all_idx = np.asarray(smap["all_idx"]).astype(int).ravel()
    gt_idx = np.asarray(smap["gt_silence_idx"]).astype(int).ravel()
    det_idx = np.asarray(smap["det_silence_idx"]).astype(int).ravel()
    src_loc = np.asarray(smap["src_loc"]).astype(np.float32)

    # Convert 1-based MATLAB indices -> 0-based Python indices
    gt_idx0 = gt_idx - 1
    det_idx0 = det_idx - 1
    all_idx0 = all_idx - 1

    p = beta.shape[0]
    if src_loc.shape[0] != p:
        raise RuntimeError(f"src_loc shape {src_loc.shape} does not match beta length {p}.")

    # Build boolean masks for ground-truth and detected silent nodes
    X_act = np.zeros(p, dtype=bool)
    X_act[gt_idx0] = True

    X_det = np.zeros(p, dtype=bool)
    X_det[det_idx0] = True

    print(f"Loaded beta of length {p}")
    print(f"Ground-truth silent nodes: {X_act.sum()} / {p}")
    print(f"Detected silent nodes:     {X_det.sum()} / {p}")

    return beta, src_loc, X_act, X_det


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="GNN smoothing on SilenceMap beta + exact MATLAB indices")
    parser.add_argument("--silencemap_mat", type=str, required=True,
                        help="Path to SilenceMap_final.mat with silencemap_export struct")
    parser.add_argument("--kNN", type=int, default=12, help="k for kNN graph")
    parser.add_argument("--sigmaW", type=float, default=12.0, help="RBF sigma for graph weights")
    parser.add_argument("--lambda_lap", type=float, default=1.0, help="Laplacian smoother lambda")
    parser.add_argument("--gnn_hidden", type=int, default=64, help="Hidden dim in GNN")
    parser.add_argument("--gnn_steps", type=int, default=2000, help="Training steps for GNN")
    parser.add_argument("--gnn_lr", type=float, default=1e-2, help="Learning rate for GNN")
    parser.add_argument("--gnn_lambda", type=float, default=5.0, help="Weight for Laplacian smooth term in GNN loss")
    parser.add_argument("--gnn_gamma", type=float, default=0.5, help="Weight for seed term in GNN loss")
    parser.add_argument("--save_figs", action="store_true", help="Save figures instead of showing them")
    parser.add_argument("--fig_dir", type=str, default="./figs", help="Directory to save figures")

    args = parser.parse_args()
    print("Device:", device)

    # ---------- Load SilenceMap beta + indices ----------
    beta_raw, src_xyz, X_act, X_det = load_silencemap_export(args.silencemap_mat)
    p = beta_raw.shape[0]

    # ---------- Normalize beta and orient it so that silent ~ low ----------
    beta = beta_raw.copy().astype(np.float32)
    beta -= beta.min()
    beta /= (beta.max() + 1e-12)

    silent = X_act
    active = ~silent
    mean_silent = float(beta[silent].mean())
    mean_active = float(beta[active].mean())
    print("mean(beta silent) :", mean_silent)
    print("mean(beta active) :", mean_active)

    # If silent nodes have higher beta than active ones, flip the scale
    if mean_silent > mean_active:
        print(">> Flipping beta: making silent = low values")
        beta = 1.0 - beta

    print("beta shape:", beta.shape)
    print("beta min/max:", beta.min(), beta.max())

    # ---------- Build graph on EXACT same nodes as MATLAB ----------
    kNN, sigmaW = args.kNN, args.sigmaW
    print(f"Building kNN graph: k={kNN}, sigma={sigmaW}")
    W, deg = knn_graph_gauss(src_xyz, k=kNN, sigma=sigmaW)
    L_g, _ = laplacian_from_W(W)

    # ---------- Laplacian smoother baseline ----------
    lam = args.lambda_lap
    I = identity(p, format="coo")
    A = (I + lam * L_g).tocsc()
    g_lap = spsolve(A, beta).astype(np.float32)
    g_lap -= g_lap.min()
    g_lap /= (g_lap.max() + 1e-12)

    # Number of true silent nodes from MATLAB
    k_silent = int(X_act.sum())
    q_silent = 100.0 * k_silent / p
    print(f"[auto] |S| (true silent) = {k_silent}/{p} → q_silent ≈ {q_silent:.2f}%")

    # Pick k_silent *smallest* g_lap values as silent
    g_lap_arr = np.asarray(g_lap)
    idx_lap = np.argpartition(g_lap_arr, k_silent - 1)[:k_silent]
    mask_lap = np.zeros_like(g_lap_arr, dtype=bool)
    mask_lap[idx_lap] = True

    P, R, F1 = pr_re_f1(mask_lap, X_act)
    print(f"Laplacian baseline: P={P:.3f} R={R:.3f} F1={F1:.3f}")

    # ---------- GNN (self-supervised on THIS beta) ----------
    Ahat = build_torch_graph(W, deg)
    Lcoo = L_g.tocoo()

    # Seed selection: extremes of beta
    beta_arr = np.asarray(beta)
    q_seed = 0.03  # 3% seeds each side
    k_seed = max(5, int(q_seed * len(beta_arr)))

    idx_sorted = np.argsort(beta_arr)
    silent_seeds = idx_sorted[:k_seed]       # lowest beta → silent
    active_seeds = idx_sorted[-k_seed:]      # highest beta → active

    seed_silent_mask = torch.zeros(beta_arr.shape[0], dtype=torch.bool, device=device)
    seed_active_mask = torch.zeros(beta_arr.shape[0], dtype=torch.bool, device=device)
    seed_silent_mask[silent_seeds] = True
    seed_active_mask[active_seeds] = True

    hidden = args.gnn_hidden
    steps = args.gnn_steps
    lr = args.gnn_lr
    lam_gnn = args.gnn_lambda
    gamma_gnn = args.gnn_gamma

    model = BetaGNN(p=p, hidden=hidden, Ahat=Ahat).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    beta_t = torch.tensor(beta, dtype=torch.float32, device=device).view(-1, 1)
    deg_feat = torch.tensor(deg, dtype=torch.float32, device=device).view(-1, 1)

    print("Starting GNN training...")
    for it in range(steps):
        opt.zero_grad()
        g = model(beta_t, deg_feat)  # (p,1)

        data_term = ((g - beta_t) ** 2).mean()
        smooth_term = lap_energy(g, Lcoo) / p

        loss_silent = torch.mean(g[seed_silent_mask] ** 2)       # g -> 0
        loss_active = torch.mean((g[seed_active_mask] - 1) ** 2) # g -> 1
        loss_seed = loss_silent + loss_active

        loss = data_term + lam_gnn * smooth_term + gamma_gnn * loss_seed
        loss.backward()
        opt.step()

        if (it + 1) % 200 == 0:
            print(f"[{it+1:04d}] loss={loss.item():.5f} "
                  f"data={data_term.item():.5f} "
                  f"smooth={smooth_term.item():.5f} "
                  f"seed={loss_seed.item():.5f}")

    with torch.no_grad():
        g_raw = model(beta_t, deg_feat).squeeze(1).cpu().numpy()
        g_hat = (g_raw - g_raw.min()) / (g_raw.max() - g_raw.min() + 1e-12)

    # Pick k_silent *smallest* g_hat values as silent
    g_hat_arr = np.asarray(g_hat)
    idx_gnn = np.argpartition(g_hat_arr, k_silent - 1)[:k_silent]
    mask_gnn = np.zeros_like(g_hat_arr, dtype=bool)
    mask_gnn[idx_gnn] = True

    P, R, F1 = pr_re_f1(mask_gnn, X_act)
    print(f"GNN:                P={P:.3f} R={R:.3f} F1={F1:.3f}")

    # ---------- Extra stats ----------
    def stats(name, arr):
        arr = np.asarray(arr)
        print(f"\n[{name}]")
        print("  mean(silent) :", float(arr[silent].mean()))
        print("  mean(active) :", float(arr[active].mean()))
        print("  corr with X_act:",
              float(np.corrcoef(arr, X_act.astype(np.float32))[0, 1]))

    stats("beta", beta)
    stats("g_lap", g_lap)
    stats("g_gnn", g_hat)

    # ---------- Plots ----------
    save = args.save_figs
    out = args.fig_dir

    show_cortex_mask(src_xyz, X_act,
                     title=f"GT silence (|S|={k_silent})",
                     save=save, outdir=out, fname="1_gt.png")
    show_cortex_mask(src_xyz, mask_lap,
                     title="Laplacian estimate",
                     save=save, outdir=out, fname="2_laplacian.png")
    show_cortex_mask(src_xyz, mask_gnn,
                     title="GNN estimate",
                     save=save, outdir=out, fname="3_gnn.png")

    fig = plt.figure(figsize=(6, 3))
    plt.plot(beta[:300], label='beta (SilenceMap)')
    plt.plot(g_lap[:300], label='g_lap (smooth)')
    plt.plot(g_hat[:300], label='g_gnn (GNN)')
    plt.legend()
    plt.title('First 300 nodes')
    show_or_save(fig, "4_curves.png", save=save, outdir=out)


if __name__ == "__main__":
    main()
