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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
def compute_beta_silencemap(eeg, L, Fs):
    """
    Approximate SilenceMap-style beta:
      - Welch PSD to estimate noise in 90–100 Hz band
      - propagate noise to source space via L
      - compute Var(mu_tilda) from projected signals
      - normalize by Var_norm_fact = sum((L' L).^2, 2)

    eeg: (n, t) array, observed EEG
    L:   (n, p) leadfield
    Fs:  sampling rate
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    L1  = np.asarray(L,   dtype=np.float64)
    n, t = eeg.shape

    # --- Welch parameters (match MATLAB logic) ---
    w_length = int(min(math.floor(0.5 * t), 256))
    w_over   = int(math.floor(0.5 * w_length))
    if w_length < 8:  # safety
        w_length = min(t, 64)
        w_over   = w_length // 2

    # =============== Noise PSD estimate (sensor space) ===============
    # MATLAB: [pxx,f] = pwelch(Y',w_length,w_over,256,Fs);
    # Here Y = eeg (no rereferencing)
    f, pxx = welch(
        eeg,
        fs=Fs,
        nperseg=w_length,
        noverlap=w_over,
        nfft=256,
        axis=1
    )  # pxx: (n, F)

    # Average PSD between 90–100 Hz
    band = (f >= 90.0) & (f <= 100.0)
    if not np.any(band):
        # fallback: use the highest frequencies if 90–100 not in grid
        band = f >= f.max() * 0.8
    eta = pxx[:, band].mean(axis=1)  # (n,)

    # sigma_z_sqrd ~ eta * (100 - 0.1)  (matching MATLAB's 100-0.1)
    sigma_z_sqrd = eta * (100.0 - 0.1)  # (n,)
    # Cz is diag(sigma_z_sqrd) conceptually

    # =============== Var_norm_fact from L' L ===============
    LL = L1.T @ L1            # (p, p)
    LL_sq = LL ** 2
    Var_norm_fact = LL_sq.sum(axis=1) + 1e-12   # (p,)

    # =============== Mu_tilda = L1' * eeg (projected sources) ===============
    Mu_tilda = L1.T @ eeg     # (p, t)

    # PSD of Mu_tilda (per source)
    f_mu, pxx_mu = welch(
        Mu_tilda,
        fs=Fs,
        nperseg=w_length,
        noverlap=w_over,
        nfft=256,
        axis=1
    )  # (p, F_mu)

    if len(f_mu) > 1:
        df_mu = f_mu[1] - f_mu[0]
    else:
        df_mu = 1.0

    # Var(mu) = ∫ PSD df - mean^2
    sigma_mu_sqrd = pxx_mu.sum(axis=1) * df_mu - (Mu_tilda.mean(axis=1) ** 2)
    sigma_mu_sqrd = sigma_mu_sqrd.astype(np.float64)  # (p,)

    # =============== Var(eeg) (not strictly needed for beta) ===============
    if len(f) > 1:
        df = f[1] - f[0]
    else:
        df = 1.0
    sigma_eeg_sqrd = pxx.sum(axis=1) * df - (eeg.mean(axis=1) ** 2)
    # P_M = sigma_eeg_sqrd - sigma_z_sqrd  # scalp power without noise (unused here)

    # =============== Noise contribution in source space: diag(L1' Cz L1) ===============
    # Cz = diag(sigma_z_sqrd). So Cz * L1 = sigma_z_sqrd[:,None] * L1
    CzL1 = sigma_z_sqrd[:, None] * L1           # (n, p)
    L1TCzL1 = L1.T @ CzL1                       # (p, p)
    noise_src_var = np.diag(L1TCzL1)           # (p,)

    sigma_mu_sqrd_wo_noise = sigma_mu_sqrd - noise_src_var
    sigma_mu_sqrd_wo_noise = np.maximum(sigma_mu_sqrd_wo_noise, 0.0)

    # =============== Beta as in SilenceMap: sigma_mu_wo_noise ./ Var_norm_fact ===============
    beta = sigma_mu_sqrd_wo_noise / Var_norm_fact
    beta = beta.astype(np.float32)

    # Normalize to [0,1]
    beta -= beta.min()
    beta /= (beta.max() + 1e-12)

    return beta

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
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)           # normalize
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
    parser.add_argument("--gnn_lambda", type=float, default=1.0, help="Smooth term weight")
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
    #Ceeg = (eeg @ eeg.T) / float(t)        # (n,n)
    #AtA  = L.T @ Ceeg @ L                  # (p,p)
    #beta = np.diag(AtA).astype(np.float32)
    #beta -= beta.min()
    #beta /= (beta.max() + 1e-12)
    # -------- Compute beta from EEG (SilenceMap-style) --------
    
    #beta = compute_beta_silencemap(eeg, L, Fs)
    ref_indices = [40,50,56,63,64,65,68,73,84,95]
    ref_indices = [i - 1 for i in ref_indices]  # convert to 0-based
    # -------- Compute beta from EEG (simulation) --------
    '''beta = compute_beta_silencemap(eeg, L, Fs)
    k_silent_gt = int(X_act.sum())
    thr_beta = np.partition(beta, k_silent_gt-1)[k_silent_gt-1]  # exact k smallest
    mask_beta = beta <= thr_beta

    overlap = (mask_beta & X_act).sum()
    print("\n[beta raw mask]")
    print("  GT silent:", int(X_act.sum()),
          "beta-silent:", int(mask_beta.sum()),
          "overlap:", int(overlap))'''
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
    idx_seed = np.argpartition(beta_arr, k_silent-1)[:k_silent]
    seed_mask = torch.zeros(beta_arr.shape[0], dtype=torch.bool, device=device)
    seed_mask[idx_seed] = True

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
        seed_term   = g[seed_mask].mean()
        loss = data_term + lam_gnn * smooth_term + gamma_gnn * seed_term

        loss.backward()
        opt.step()
        if (it+1) % 200 == 0:
            print(f"[{it+1:04d}] loss={loss.item():.5f} "
                  f"data={data_term.item():.5f} "
                  f"smooth={smooth_term.item():.5f} "
                  f"seed={seed_term.item():.5f}")

    with torch.no_grad():
        g_hat = model(beta_t, deg_feat).squeeze(1).detach().cpu().numpy()

    # ---- GNN mask: choose k_silent *smallest* g_hat values as silent ----
    g_hat_arr = np.asarray(g_hat)
    idx_gnn = np.argpartition(g_hat_arr, k_silent-1)[:k_silent]
    mask_gnn = np.zeros_like(g_hat_arr, dtype=bool)
    mask_gnn[idx_gnn] = True

    P, R, F1 = pr_re_f1(mask_gnn, X_act)
    print(f"GNN:       P={P:.3f} R={R:.3f} F1={F1:.3f}")
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

    # β / g curves
    fig = plt.figure(figsize=(6,3))
    plt.plot(beta[:300], label='beta')
    plt.plot(g_lap[:300], label='g_lap')
    plt.plot(g_hat[:300], label='g_gnn')
    plt.legend(); plt.title('First 300 nodes')
    show_or_save(fig, "4_curves.png", save=save, outdir=out)


if __name__ == "__main__":
    main()
