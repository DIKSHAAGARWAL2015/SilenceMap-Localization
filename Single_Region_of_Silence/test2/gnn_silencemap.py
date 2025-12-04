#!/usr/bin/env python3
# gnn_silencemap_gnn.py
# Use MATLAB SilenceMap beta + indices, run Laplacian + GNN,
# and compare to the same ground-truth silence region (single region).

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import mat73
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn
import torch.optim as optim


# ------------------- Basic checks -------------------
def _need(pkg):
    try:
        __import__(pkg)
        return False
    except Exception:
        return True


_missing = []
for pkg in ["numpy", "matplotlib", "scipy", "torch", "mat73"]:
    if _need(pkg):
        _missing.append(pkg)

if _missing:
    print(">> Missing packages:", _missing)
    print(">> In Colab, run: !pip -q install " + " ".join(_missing))
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- Helpers: graph & plotting -------------------
def knn_graph_gauss(coords, k=12, sigma=12.0):
    """
    Build a k-NN graph with Gaussian weights from 3D coordinates.
    Returns W (sparse adjacency) and degree vector.
    """
    from sklearn.neighbors import NearestNeighbors

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

    # symmetrize by max
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


def show_or_save(fig, name, save=False, outdir="./figs"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if save:
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved figure → {path}]")
    else:
        plt.show()


def show_cortex_mask(coords, mask, title="", save=False, outdir="./figs", fname="mask.png"):
    """
    3D scatter of cortex with highlighted nodes.
    """
    mask = np.asarray(mask, dtype=bool)
    c = np.where(mask, 1.0, 0.1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=c, s=8, cmap="cool")
    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()
    show_or_save(fig, fname, save=save, outdir=outdir)


def pr_re_f1(pred_mask, gt_mask):
    """
    Precision / Recall / F1 for binary masks.
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


# ------------------- Cluster metrics (single region) -------------------
def _cluster_metrics(gt_nodes, pred_nodes, coords):
    """
    Compute Jaccard, ΔCOM, and size stats for a single GT vs predicted region.
    gt_nodes, pred_nodes: 1D arrays/list of node indices
    coords: (p,3) array of coordinates
    """
    gt_nodes = set([int(i) for i in gt_nodes])
    pred_nodes = set([int(i) for i in pred_nodes])

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


def cluster_metrics_single_region(gt_mask, pred_mask, coords):
    """
    Convenience wrapper: extract indices from masks and call _cluster_metrics.
    """
    gt_idx = np.where(np.asarray(gt_mask, dtype=bool))[0]
    pred_idx = np.where(np.asarray(pred_mask, dtype=bool))[0]
    return _cluster_metrics(gt_idx, pred_idx, coords)


# ------------------- Torch graph helpers -------------------
def build_torch_graph(W, deg):
    W = W.tocoo()
    idx = torch.tensor(
        np.vstack([W.row, W.col]).astype(np.int64), device=device
    )
    di = 1.0 / torch.sqrt(
        torch.clamp(
            torch.tensor(deg, dtype=torch.float32, device=device),
            min=1e-8,
        )
    )
    v = torch.tensor(W.data.astype(np.float32), device=device)
    v = v * di[idx[0]] * di[idx[1]]  # normalized adjacency
    Ahat = torch.sparse_coo_tensor(idx, v, W.shape, device=device).coalesce()
    return Ahat


def lap_energy(g, Lcoo):
    # g: (p,1)
    i = torch.tensor(
        np.vstack([Lcoo.row, Lcoo.col]).astype(np.int64), device=device
    )
    v = torch.tensor(Lcoo.data.astype(np.float32), device=device)
    L_t = torch.sparse_coo_tensor(i, v, Lcoo.shape).coalesce()
    Lg = torch.sparse.mm(L_t, g)
    return (g * Lg).sum()


# ------------------- GNN model -------------------
class BetaGNN(nn.Module):
    def __init__(self, p, hidden, Ahat):
        super().__init__()
        self.lin_in = nn.Linear(3, hidden, bias=True)  # [β, β^2, degree]
        self.lin_mp1 = nn.Linear(hidden, hidden, bias=False)
        self.lin_mp2 = nn.Linear(hidden, hidden, bias=False)
        self.lin_out = nn.Linear(hidden, 1, bias=True)
        self.Ahat = Ahat

    def mp(self, H):
        AH = torch.sparse.mm(self.Ahat, H)
        A2H = torch.sparse.mm(self.Ahat, AH)
        return AH, A2H

    def forward(self, beta, degree):
        x = torch.cat([beta, beta ** 2, degree], dim=1)  # (p,3)
        H = torch.relu(self.lin_in(x))
        AH, A2H = self.mp(H)
        H = torch.relu(self.lin_mp1(AH) + self.lin_mp2(A2H))
        g = torch.nn.functional.softplus(self.lin_out(H))  # ≥0
        return g


# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser(
        description="GNN on SilenceMap beta (single-region, using MATLAB export)."
    )
    parser.add_argument(
        "--silencemap_path",
        type=str,
        required=True,
        help="Path to MATLAB file containing 'silencemap_export' struct.",
    )
    parser.add_argument("--kNN", type=int, default=8, help="k in k-NN graph")
    parser.add_argument(
        "--sigmaW", type=float, default=12.0, help="Gaussian width (mm) for graph"
    )
    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_steps", type=int, default=1500)
    parser.add_argument("--gnn_lr", type=float, default=1e-2)
    parser.add_argument("--gnn_lambda", type=float, default=5.0, help="Smooth term weight")
    parser.add_argument("--gnn_gamma", type=float, default=0.5, help="Seed term weight")
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--fig_dir", type=str, default="./figs")

    args = parser.parse_args()
    print("Device:", device)

    # -------- Load SilenceMap export from MATLAB --------
    print(f"Loading SilenceMap file: {args.silencemap_path}")
    mat = mat73.loadmat(args.silencemap_path)
    if "silencemap_export" not in mat:
        raise RuntimeError("Could not find 'silencemap_export' in MATLAB file.")

    smap = mat["silencemap_export"]

    # Extract beta, indices, and coordinates
    beta = np.array(smap["beta"]).astype(np.float32).reshape(-1)
    all_idx = np.array(smap["all_idx"]).astype(int).ravel()
    gt_idx = np.array(smap["gt_silence_idx"]).astype(int).ravel()
    det_idx = np.array(smap["det_silence_idx"]).astype(int).ravel()
    src_xyz = np.array(smap["src_loc"]).astype(np.float32)

    # Convert MATLAB 1-based indices -> 0-based
    all_idx0 = all_idx - 1
    gt_idx0 = gt_idx - 1
    det_idx0 = det_idx - 1

    p = beta.shape[0]
    print(f"beta length: {p}")
    print(f"coords shape: {src_xyz.shape}")
    print(f"|GT silence| (from MATLAB): {len(gt_idx0)}")
    print(f"|det silence| (MATLAB SilenceMap): {len(det_idx0)}")

    # Sanity: masks
    gt_mask = np.zeros(p, dtype=bool)
    gt_mask[gt_idx0] = True

    det_mask = np.zeros(p, dtype=bool)
    det_mask[det_idx0] = True

    # Normalize beta, and ensure "silent = low beta"
    beta = beta.astype(np.float32)
    beta_min, beta_max = float(beta.min()), float(beta.max())
    beta = (beta - beta_min) / (beta_max - beta_min + 1e-12)

    mean_silent = float(beta[gt_mask].mean())
    mean_active = float(beta[~gt_mask].mean())
    print("mean(beta GT silent) :", mean_silent)
    print("mean(beta GT active) :", mean_active)

    if mean_silent > mean_active:
        print(">> Flipping beta so silent nodes have smaller values.")
        beta = 1.0 - beta

    print("beta min/max after norm:", float(beta.min()), float(beta.max()))

    # -------- Build graph --------
    print("Building k-NN graph...")
    W, deg = knn_graph_gauss(src_xyz, k=args.kNN, sigma=args.sigmaW)
    L_g, _ = laplacian_from_W(W)
    Ahat = build_torch_graph(W, deg)
    Lcoo = L_g.tocoo()

    # -------- Laplacian baseline (on beta directly) --------
    lam_lap = 1.0
    I = identity(p, format="coo")
    A = (I + lam_lap * L_g).tocsc()
    g_lap = spsolve(A, beta).astype(np.float32)
    g_lap -= g_lap.min()
    g_lap /= (g_lap.max() + 1e-12)

    k_silent = int(gt_mask.sum())
    print(f"k_silent (from GT) = {k_silent}")

    # pick k_silent smallest g_lap as silent
    idx_lap = np.argpartition(g_lap, k_silent - 1)[:k_silent]
    mask_lap = np.zeros_like(g_lap, dtype=bool)
    mask_lap[idx_lap] = True

    P, R, F1 = pr_re_f1(mask_lap, gt_mask)
    print(f"\nLaplacian baseline: P={P:.3f} R={R:.3f} F1={F1:.3f}")

    cm_lap = cluster_metrics_single_region(gt_mask, mask_lap, src_xyz)
    print("Laplacian cluster metrics:")
    print(f"  Jaccard      : {cm_lap['jaccard']:.3f}")
    print(f"  ΔCOM (mm)    : {cm_lap['delta_com']:.3f}")
    print(f"  size_gt      : {cm_lap['size_gt']}")
    print(f"  size_pred    : {cm_lap['size_pred']}")
    print(f"  size_rel_err : {cm_lap['size_rel_error']:.3f}")

    # -------- GNN: self-supervised on single beta --------
    hidden = args.gnn_hidden
    steps = args.gnn_steps
    lr = args.gnn_lr
    lam_gnn = args.gnn_lambda
    gamma_gnn = args.gnn_gamma

    model = BetaGNN(p=p, hidden=hidden, Ahat=Ahat).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    beta_t = torch.tensor(beta, dtype=torch.float32, device=device).view(-1, 1)
    deg_feat = torch.tensor(deg, dtype=torch.float32, device=device).view(-1, 1)

    # Seed selection: use beta to pick silent vs active seeds
    beta_arr = np.asarray(beta)
    q_seed = 0.03  # 3% seeds
    k_seed = max(5, int(q_seed * len(beta_arr)))

    idx_sorted = np.argsort(beta_arr)  # small = silent
    silent_seeds = idx_sorted[:k_seed]
    active_seeds = idx_sorted[-k_seed:]

    seed_silent_mask = torch.zeros(p, dtype=torch.bool, device=device)
    seed_active_mask = torch.zeros(p, dtype=torch.bool, device=device)
    seed_silent_mask[silent_seeds] = True
    seed_active_mask[active_seeds] = True

    print("\nTraining GNN...")
    for it in range(steps):
        opt.zero_grad()
        g = model(beta_t, deg_feat)  # (p,1)

        data_term = ((g - beta_t) ** 2).mean()
        smooth_term = lap_energy(g, Lcoo) / p

        loss_silent = torch.mean(g[seed_silent_mask] ** 2)  # -> 0
        loss_active = torch.mean((g[seed_active_mask] - 1) ** 2)  # -> 1
        loss_seed = loss_silent + loss_active

        loss = data_term + lam_gnn * smooth_term + gamma_gnn * loss_seed
        loss.backward()
        opt.step()

        if (it + 1) % 200 == 0:
            print(
                f"[{it+1:04d}] loss={loss.item():.5f} "
                f"data={data_term.item():.5f} "
                f"smooth={smooth_term.item():.5f} "
                f"seed={loss_seed.item():.5f}"
            )

    with torch.no_grad():
        g_raw = model(beta_t, deg_feat).squeeze(1).cpu().numpy()
        g_hat = (g_raw - g_raw.min()) / (g_raw.max() - g_raw.min() + 1e-12)

    # choose k_silent smallest g_hat as silent
    idx_gnn = np.argpartition(g_hat, k_silent - 1)[:k_silent]
    mask_gnn = np.zeros_like(g_hat, dtype=bool)
    mask_gnn[idx_gnn] = True

    P, R, F1 = pr_re_f1(mask_gnn, gt_mask)
    print(f"\nGNN: P={P:.3f} R={R:.3f} F1={F1:.3f}")

    cm_gnn = cluster_metrics_single_region(gt_mask, mask_gnn, src_xyz)
    print("GNN cluster metrics:")
    print(f"  Jaccard      : {cm_gnn['jaccard']:.3f}")
    print(f"  ΔCOM (mm)    : {cm_gnn['delta_com']:.3f}")
    print(f"  size_gt      : {cm_gnn['size_gt']}")
    print(f"  size_pred    : {cm_gnn['size_pred']}")
    print(f"  size_rel_err : {cm_gnn['size_rel_error']:.3f}")

    # -------- Plots --------
    save = args.save_figs
    out = args.fig_dir

    show_cortex_mask(
        src_xyz,
        gt_mask,
        title="GT silent (from MATLAB)",
        save=save,
        outdir=out,
        fname="1_gt.png",
    )
    show_cortex_mask(
        src_xyz,
        mask_lap,
        title="Laplacian silent",
        save=save,
        outdir=out,
        fname="2_laplacian.png",
    )
    show_cortex_mask(
        src_xyz,
        mask_gnn,
        title="GNN silent",
        save=save,
        outdir=out,
        fname="3_gnn.png",
    )

    fig = plt.figure(figsize=(6, 3))
    plt.plot(beta[:300], label="beta (MATLAB)")
    plt.plot(g_lap[:300], label="g_lap")
    plt.plot(g_hat[:300], label="g_gnn")
    plt.legend()
    plt.title("First 300 nodes")
    show_or_save(fig, "4_curves.png", save=save, outdir=out)


if __name__ == "__main__":
    main()
