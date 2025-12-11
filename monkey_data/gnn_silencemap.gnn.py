#!/usr/bin/env python3
# gnn_silencemap_gnn.py
#
# SilenceMap-style experiment on monkey cortex:
# - Load DownsampledCortex.V (p x 3) from cortex_downsampled.mat
# - Pick MULTIPLE contiguous regions of nodes as "silent" (oracle X_act)
# - Define binary beta from X_act (silent=0, active=1)
# - Run Laplacian smoother baseline
# - Train a SUPERVISED GNN to predict silent vs active
# - Compare P/R/F1 and plot masks & curves

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import spsolve
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim

# ---- dependency check ----
def _need(pkg):
    try:
        __import__(pkg)
        return False
    except Exception:
        return True

_missing = []
for pkg in ["numpy", "matplotlib", "scipy", "sklearn", "torch"]:
    if _need(pkg):
        _missing.append(pkg)

if _missing:
    print(">> Missing packages:", _missing)
    print(">> In Colab, run: !pip -q install " + " ".join(_missing))
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= Robust .mat loader =========================
def smart_loadmat(path):
    """
    Load a .mat file that may be:
      - v7/v6 (classic MATLAB)  -> use scipy.io.loadmat
      - v7.3 (HDF5-based)      -> try mat73, then raw h5py as fallback

    Returns: dict-like structure.
    """
    print(f"[smart_loadmat] Loading {path} ...")

    # 1) Try SciPy (works for <= v7.2)
    try:
        return sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError as e:
        print(f"[smart_loadmat] SciPy cannot read this file (likely v7.3): {e}")
    except Exception as e:
        print(f"[smart_loadmat] SciPy loadmat failed: {e}")

    # 2) Try mat73 (best for v7.3)
    try:
        import mat73
        print("[smart_loadmat] Trying mat73.loadmat ...")
        return mat73.loadmat(path)
    except ImportError:
        print("[smart_loadmat] mat73 is not installed; skipping mat73.")
    except Exception as e:
        print(f"[smart_loadmat] mat73.loadmat failed: {e}")

    # 3) Last resort: raw HDF5 via h5py
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "This appears to be a MATLAB v7.3 file. "
            "Please install either 'mat73' or 'h5py' (or both) to load it."
        )

    print("[smart_loadmat] Falling back to raw h5py read ...")
    data = {}

    def _visitor(name, obj):
        # collect datasets; skip groups
        if isinstance(obj, h5py.Dataset):
            data[name] = obj[()]

    with h5py.File(path, "r") as f:
        f.visititems(_visitor)

    print(f"[smart_loadmat] Loaded keys (first 10): {list(data.keys())[:10]}")
    return data


def get_struct_field(struct_obj, field_name):
    """
    Handle both:
      - MATLAB structs loaded via SciPy (as simple objects with attributes)
      - dicts from mat73/h5py where fields are dict keys
    """
    if hasattr(struct_obj, field_name):
        return getattr(struct_obj, field_name)

    if isinstance(struct_obj, dict) and field_name in struct_obj:
        return struct_obj[field_name]

    raise KeyError(
        f"Could not find field '{field_name}' in structure. "
        f"Available keys/attrs: {getattr(struct_obj, '__dict__', None) or (struct_obj.keys() if isinstance(struct_obj, dict) else 'unknown')}"
    )


# ========================= Utilities =========================
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


def show_or_save(fig, name, save=False, outdir="/content/sample_data/figs_monkey/"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if save:
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved figure → {path}]")
    else:
        plt.show()


def knn_graph_gauss(coords, k=12, sigma=12.0):
    """
    Build symmetric kNN graph with RBF weights.
    coords: (p, 3)
    Returns:
        W: sparse (p, p) adjacency
        deg: degree vector (p,)
    """
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


def pr_re_f1(pred_mask, gt_mask):
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


def show_cortex_mask(coords, mask, title="", save=False, outdir="/content/sample_data/figs_monkey/", fname="mask.png"):
    c = np.where(mask, 1.0, 0.1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=c, s=8, cmap="cool")
    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()
    show_or_save(fig, fname, save=save, outdir=outdir)


def build_torch_graph(W, deg):
    W = W.tocoo()
    idx = torch.tensor(np.vstack([W.row, W.col]).astype(np.int64), device=device)
    di = 1.0 / torch.sqrt(torch.clamp(torch.tensor(deg, dtype=torch.float32, device=device), min=1e-8))
    v = torch.tensor(W.data.astype(np.float32), device=device)
    v = v * di[idx[0]] * di[idx[1]]  # normalized adjacency Ahat
    Ahat = torch.sparse_coo_tensor(idx, v, W.shape, device=device).coalesce()
    return Ahat


def lap_energy(g, Lcoo):
    i = torch.tensor(np.vstack([Lcoo.row, Lcoo.col]).astype(np.int64), device=device)
    v = torch.tensor(Lcoo.data.astype(np.float32), device=device)
    L_t = torch.sparse_coo_tensor(i, v, Lcoo.shape, device=device).coalesce()
    Lg = torch.sparse.mm(L_t, g)
    return (g * Lg).sum()


# ========================= GNN =========================
class BetaGNN(nn.Module):
    """
    Simple 2-hop message passing network:
    Input features: [beta, beta^2, degree]
    Output: logit for "silent" (before sigmoid)
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
        x = torch.cat([beta, beta ** 2, degree], dim=1)  # (p,3)
        H = torch.relu(self.lin_in(x))
        AH, A2H = self.mp(H)
        H = torch.relu(self.lin_mp1(AH) + self.lin_mp2(A2H))
        logits = self.lin_out(H)  # (p,1), raw logits for "silent"
        return logits


# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="SilenceMap + supervised GNN on monkey cortex (binary beta)")
    parser.add_argument(
        "--use_mat",
        action="store_true",
        help="Load monkey cortex vertices from cortex_downsampled.mat (DownsampledCortex.V)",
    )
    parser.add_argument(
        "--headmodel_path",
        type=str,
        default="/content/sample_data/cortex_downsampled_v73.mat",
        help="Path to .mat with DownsampledCortex struct or datasets",
    )
    parser.add_argument(
        "--silent_percent",
        type=float,
        default=1.0,
        help="TOTAL percent of nodes to mark as silent (oracle, across all regions)",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=3,
        help="Number of separate silent regions to simulate",
    )
    parser.add_argument("--kNN", type=int, default=12, help="Graph neighbors")
    parser.add_argument("--sigmaW", type=float, default=12.0, help="Graph RBF width (mm)")
    parser.add_argument("--lambda_lap", type=float, default=1.0, help="Laplacian smoother lambda")
    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_steps", type=int, default=2000)
    parser.add_argument("--gnn_lr", type=float, default=1e-2)
    parser.add_argument("--gnn_smooth", type=float, default=1.0, help="GNN Laplacian smoothness weight")
    parser.add_argument("--save_figs", action="store_true", help="Save figures instead of showing")
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="/content/sample_data/figs_monkey/",
        help="Where to save figures",
    )

    args = parser.parse_args()
    print("Device:", device)

    # -------- LOAD MONKEY CORTEX VERTICES --------
    if args.use_mat:
        print("Loading cortex vertices from .mat using smart_loadmat ...")
        head = smart_loadmat(args.headmodel_path)

        # Case 1: classic struct at top-level
        if "DownsampledCortex" in head:
            cortex_struct = head["DownsampledCortex"]
            V = get_struct_field(cortex_struct, "V")

        # Case 2: maybe lowercase
        elif "downsampledCortex" in head:
            cortex_struct = head["downsampledCortex"]
            V = get_struct_field(cortex_struct, "V")

        # Case 3: h5py-style flat datasets: "DownsampledCortex/V"
        elif "DownsampledCortex/V" in head:
            V = head["DownsampledCortex/V"]

        else:
            raise KeyError(
                "Could not find 'DownsampledCortex' or 'DownsampledCortex/V' in loaded .mat file. "
                f"Available top-level keys: {list(head.keys())}"
            )

        V = np.array(V, dtype=np.float32)
        if V.ndim != 2:
            V = V.reshape(V.shape[0], -1)

        if V.shape[1] == 3:
            src_xyz = V
        elif V.shape[0] == 3:
            src_xyz = V.T
        else:
            raise RuntimeError(f"Vertices shape {V.shape} must be (p,3) or (3,p).")

        p = src_xyz.shape[0]
        print("Loaded vertices:", src_xyz.shape)
    else:
        raise RuntimeError("Please run with --use_mat and a valid cortex_downsampled.mat")

    # -------- ORACLE SILENT MASK: MULTIPLE CONTIGUOUS REGIONS --------
    rng = np.random.default_rng(42)
    k_total = max(1, int(args.silent_percent / 100.0 * p))
    K = max(1, args.num_regions)
    k_per = max(1, k_total // K)

    print(f"Simulating {K} silent regions on cortex")
    print(f"Total silent nodes ≈ {k_total} ({args.silent_percent:.2f}% of {p})")
    print(f"Target per-region size ≈ {k_per}")

    silent_idx_list = []
    available = set(range(p))

    for r in range(K):
        if not available:
            print(f"Region {r}: no available vertices left, stopping.")
            break

        center = rng.choice(list(available))
        print(f"Region {r}: center index = {center}")

        d2 = np.sum((src_xyz - src_xyz[center]) ** 2, axis=1)
        order = np.argsort(d2)

        region_picks = []
        for idx in order:
            if idx in available:
                region_picks.append(idx)
                if len(region_picks) >= k_per:
                    break

        if not region_picks:
            print(f"Region {r}: could not pick any vertices (weird).")
            continue

        print(f"Region {r}: picked {len(region_picks)} vertices")
        silent_idx_list.extend(region_picks)
        for idx in region_picks:
            if idx in available:
                available.remove(idx)

    if len(silent_idx_list) < k_total and len(available) > 0:
        need = k_total - len(silent_idx_list)
        extra = rng.choice(list(available), size=min(need, len(available)), replace=False)
        print(f"Filling remaining {len(extra)} silent vertices to reach total ≈ {k_total}")
        silent_idx_list.extend(extra.tolist())
        for idx in extra:
            if idx in available:
                available.remove(idx)

    silent_idx = np.unique(np.array(silent_idx_list, dtype=np.int64))
    k_silent = len(silent_idx)
    q_silent = 100.0 * k_silent / p
    print(f"Final |S| = {k_silent} silent nodes ({q_silent:.2f}% of cortex)")

    X_act = np.zeros(p, dtype=bool)
    X_act[silent_idx] = True  # True = silent

    # -------- BINARY BETA FROM ORACLE SILENCE --------
    beta = np.ones(p, dtype=np.float32)
    beta[X_act] = 0.0
    beta -= beta.min()
    beta /= (beta.max() + 1e-12)

    silent = X_act
    active = ~silent
    print("mean(beta silent):", float(beta[silent].mean()))
    print("mean(beta active):", float(beta[active].mean()))
    print("beta shape:", beta.shape)
    print("beta min/max:", beta.min(), beta.max())

    # -------- GRAPH + LAPLACIAN SMOOTHER (BASELINE) --------
    print("Building kNN graph ...")
    W, deg = knn_graph_gauss(src_xyz, k=args.kNN, sigma=args.sigmaW)
    L_g, _ = laplacian_from_W(W)

    lam = args.lambda_lap
    I = identity(p, format="coo")
    A = (I + lam * L_g).tocsc()

    print("Solving Laplacian smoother ...")
    g_lap = spsolve(A, beta).astype(np.float32)
    g_lap -= g_lap.min()
    g_lap /= (g_lap.max() + 1e-12)

    print(f"[auto] q_silent set to {q_silent:.2f}% for |S|={k_silent}/{p}")

    g_lap_arr = np.asarray(g_lap)
    idx_lap = np.argpartition(g_lap_arr, k_silent - 1)[:k_silent]
    mask_lap = np.zeros_like(g_lap_arr, dtype=bool)
    mask_lap[idx_lap] = True

    P, R, F1 = pr_re_f1(mask_lap, X_act)
    print(f"Laplacian: P={P:.3f} R={R:.3f} F1={F1:.3f}")

    # -------- SUPERVISED GNN --------
    print("Building GNN ...")
    Ahat = build_torch_graph(W, deg)
    Lcoo = L_g.tocoo()

    beta_t = torch.tensor(beta, dtype=torch.float32, device=device).view(-1, 1)
    deg_feat = torch.tensor(deg, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.tensor(X_act.astype(np.float32), device=device).view(-1, 1)

    hidden = args.gnn_hidden
    steps = args.gnn_steps
    lr = args.gnn_lr
    lam_gnn_smooth = args.gnn_smooth

    model = BetaGNN(p=p, hidden=hidden, Ahat=Ahat).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("Training supervised GNN ...")
    for it in range(steps):
        opt.zero_grad()
        logits = model(beta_t, deg_feat)
        prob = torch.sigmoid(logits)

        cls_loss = criterion(logits, y)
        smooth_term = lap_energy(prob, Lcoo) / p
        loss = cls_loss + lam_gnn_smooth * smooth_term

        loss.backward()
        opt.step()

        if (it + 1) % 200 == 0:
            print(
                f"[{it+1:04d}] loss={loss.item():.5f} "
                f"cls={cls_loss.item():.5f} "
                f"smooth={smooth_term.item():.5f}"
            )

    with torch.no_grad():
        logits = model(beta_t, deg_feat)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    prob_arr = np.asarray(prob)
    idx_gnn = np.argpartition(prob_arr, -k_silent)[-k_silent:]
    mask_gnn = np.zeros_like(prob_arr, dtype=bool)
    mask_gnn[idx_gnn] = True

    P, R, F1 = pr_re_f1(mask_gnn, X_act)
    print(f"GNN:       P={P:.3f} R={R:.3f} F1={F1:.3f}")

    def stats(name, arr):
        arr = np.asarray(arr)
        print(f"\n[{name}]")
        print("  mean(silent):", float(arr[silent].mean()))
        print("  mean(active):", float(arr[active].mean()))
        print(
            "  corr with X_act:",
            float(np.corrcoef(arr, X_act.astype(np.float32))[0, 1]),
        )

    stats("beta", beta)
    stats("g_lap", g_lap)
    stats("g_gnn_prob", prob_arr)

    save = args.save_figs
    out = args.fig_dir

    show_cortex_mask(src_xyz, X_act,
                     title=f"GT silent (K={K}, q={q_silent:.2f}%)",
                     save=save, outdir=out, fname="1_gt.png")
    show_cortex_mask(src_xyz, mask_lap,
                     title=f"Laplacian mask (q={q_silent:.2f}%)",
                     save=save, outdir=out, fname="2_laplacian.png")
    show_cortex_mask(src_xyz, mask_gnn,
                     title=f"GNN mask (q={q_silent:.2f}%)",
                     save=save, outdir=out, fname="3_gnn.png")

    show_overlay_mask(src_xyz, X_act,
                      title="GT Silent Regions (Overlay)",
                      save=args.save_figs,
                      outdir=args.fig_dir,
                      fname="1_gt_overlay.png")

    show_overlay_mask(src_xyz, mask_lap,
                      title="Laplacian Detected Regions (Overlay)",
                      save=args.save_figs,
                      outdir=args.fig_dir,
                      fname="2_laplacian_overlay.png")

    show_overlay_mask(src_xyz, mask_gnn,
                      title="GNN Detected Regions (Overlay)",
                      save=args.fig_dir,
                      outdir=args.fig_dir,
                      fname="3_gnn_overlay.png")

    fig = plt.figure(figsize=(6, 3))
    plt.plot(beta[:300], label="beta")
    plt.plot(g_lap[:300], label="g_lap")
    plt.plot(prob_arr[:300], label="g_gnn_prob")
    plt.legend()
    plt.title("First 300 nodes")
    show_or_save(fig, "4_curves.png", save=save, outdir=out)


if __name__ == "__main__":
    main()
