#!/usr/bin/env python3
import os, sys, time, argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------- Utils: .mat loader (your robust loader) -----------------
def _walk_py(obj, prefix=""):
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
    best = None
    for path, arr in items:
        if arr.ndim == 2 and arr.size >= 32*256:
            name = path.lower()
            score = arr.size
            if path.split(".")[-1].lower() in ("l",): score += 10_000_000
            if "leadfield" in name: score += 5_000_000
            if best is None or score > best[0]:
                best = (score, path, arr)
    return None if best is None else best[2]

def _pick_vertices_from_py(items):
    best = None
    for path, arr in items:
        if arr.ndim == 2 and 3 in arr.shape:
            name = path.lower()
            score = max(arr.shape)
            if "vertices" in name: score += 1_000_000
            if "pial" in name: score += 500_000
            if "cortex" in name: score += 100_000
            if best is None or score > best[0]:
                best = (score, path, arr)
    return None if best is None else best[2]

def load_leadfield_any(leadfield_path, headmodel_path, leadfield_var=None, vertices_var=None):
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

    # mat73
    try:
        import mat73
        lead = mat73.loadmat(leadfield_path)
        if leadfield_var:
            L = _resolve_dotted(lead, leadfield_var)
        else:
            items = _walk_py(lead)
            L = _pick_leadfield_from_py(items)
            if L is None:
                raise RuntimeError("Could not find leadfield matrix L in file (mat73).")

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
    except Exception:
        pass

    # HDF5
    try:
        import h5py
        with h5py.File(leadfield_path, "r") as f:
            if "L" in f:
                L = np.array(f["L"])
            else:
                raise KeyError("HDF5 leadfield file missing dataset 'L'")
            if "src_xyz" in f:
                V = np.array(f["src_xyz"])
                src_xyz = _normalize_vertices(V)
                L = _normalize_L(L, src_xyz.shape[0])
                return L.astype(np.float32), src_xyz.astype(np.float32)

        with h5py.File(headmodel_path, "r") as f:
            if "src_xyz" in f:
                V = np.array(f["src_xyz"])
            elif "vertices" in f:
                V = np.array(f["vertices"])
            else:
                raise KeyError("Could not find vertices in HDF5 headmodel file")
            src_xyz = _normalize_vertices(V)
            L = _normalize_L(L, src_xyz.shape[0])
            return L.astype(np.float32), src_xyz.astype(np.float32)
    except Exception:
        pass

    # scipy loadmat
    import scipy.io as sio
    lead = sio.loadmat(leadfield_path)
    L = lead.get("L", None) if not leadfield_var else lead.get(leadfield_var, None)
    if L is None:
        raise RuntimeError("Could not find L in leadfield file")

    head = sio.loadmat(headmodel_path)
    V = head.get("src_xyz", None) or head.get("vertices", None)
    if V is None:
        raise RuntimeError("Could not find vertices in headmodel file")

    src_xyz = _normalize_vertices(V)
    L = _normalize_L(L, src_xyz.shape[0])
    return L.astype(np.float32), src_xyz.astype(np.float32)

# ----------------- Graph + features -----------------
def knn_graph_gauss(coords, k=12, sigma=12.0):
    p = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k+1, p)).fit(coords)
    dists, idxs = nn.kneighbors(coords)
    rows, cols, vals = [], [], []
    for i in range(p):
        for j, d in zip(idxs[i,1:], dists[i,1:]):
            w = np.exp(-(d*d)/(2.0*sigma*sigma))
            rows.append(i); cols.append(j); vals.append(w)
    dct = {}
    for r,c,v in zip(rows, cols, vals):
        if (r,c) not in dct or dct[(r,c)] < v: dct[(r,c)] = v
        if (c,r) not in dct or dct[(c,r)] < v: dct[(c,r)] = v
    rr, cc, vv = zip(*[(r,c,v) for (r,c),v in dct.items()])
    W = coo_matrix((vv,(rr,cc)), shape=(p,p))
    deg = np.array(W.sum(axis=1)).ravel().astype(np.float32)
    return W.tocoo(), deg

def neighbor_stats_from_W(W, x):
    W = W.tocsr()
    x = np.asarray(x).reshape(-1).astype(np.float32)
    deg = np.array(W.sum(axis=1)).ravel().astype(np.float32)
    deg_safe = np.maximum(deg, 1e-8)
    nbr_mean = (W @ x) / deg_safe
    nbr_mean2 = (W @ (x*x)) / deg_safe
    nbr_var = np.maximum(nbr_mean2 - nbr_mean*nbr_mean, 0.0)
    nbr_std = np.sqrt(nbr_var + 1e-8)
    return nbr_mean.astype(np.float32), nbr_std.astype(np.float32), deg.astype(np.float32)

def build_torch_graph(W, deg, device):
    W = W.tocoo()
    idx = torch.tensor(np.vstack([W.row, W.col]).astype(np.int64), device=device)
    di = 1.0 / torch.sqrt(torch.clamp(torch.tensor(deg, dtype=torch.float32, device=device), min=1e-8))
    v  = torch.tensor(W.data.astype(np.float32), device=device)
    v  = v * di[idx[0]] * di[idx[1]]
    Ahat = torch.sparse_coo_tensor(idx, v, W.shape, device=device).coalesce()
    return Ahat

def laplacian_from_W(W):
    deg = np.array(W.sum(axis=1)).ravel()
    return (diags(deg) - W).tocoo(), deg

def lap_energy(prob, Lcoo, device):
    i = torch.tensor(np.vstack([Lcoo.row, Lcoo.col]).astype(np.int64), device=device)
    v = torch.tensor(Lcoo.data.astype(np.float32), device=device)
    L_t = torch.sparse_coo_tensor(i, v, Lcoo.shape, device=device).coalesce()
    Lp = torch.sparse.mm(L_t, prob)
    return (prob * Lp).sum()

# ----------------- SilenceMap FAST beta -----------------
def build_reference_matrix(n, i_ref0):
    i_ref0 = int(i_ref0)
    M = np.eye(n-1, dtype=np.float64)
    return np.concatenate([M[:, :i_ref0], -np.ones((n-1,1)), M[:, i_ref0:]], axis=1)

def estimate_noise_Cz_fast(Y, Fs=None):
    Y = np.asarray(Y, dtype=np.float64)
    sigma_z_sqrd = np.var(Y, axis=1, ddof=0) + 1e-12
    return np.diag(sigma_z_sqrd)

def var_fast(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1)
    return (X*X).mean(axis=1) - mu*mu

def silencemap_beta_highres_fast(eeg, L, Cs, Fs=512, i_ref0=63):
    eeg = np.asarray(eeg, dtype=np.float64)
    L   = np.asarray(L, dtype=np.float64)
    Cs  = np.asarray(Cs, dtype=np.float64)

    n = L.shape[0]
    M = build_reference_matrix(n, i_ref0)
    Y = M @ eeg
    Lref = M @ L

    Cz = estimate_noise_Cz_fast(Y, Fs)
    Mu = Lref.T @ Y
    sig_mu = var_fast(Mu)
    sig_mu_wo = sig_mu - np.diag(Lref.T @ Cz @ Lref)
    sig_mu_wo = np.maximum(sig_mu_wo, 0.0)
    # normalize by per-vertex leadfield energy (diag(G)), not diag(G^2)
    G = Lref.T @ Lref
    denom = np.diag(G) + 1e-12
    beta = sigma_mu_wo / denom
    return beta

# ----------------- Fast source simulation (no huge MVN) -----------------
def simulate_sources_fast(W, t, rng, smooth_passes=3):
    Wcsr = W.tocsr()
    p = Wcsr.shape[0]
    S = rng.standard_normal((p, t)).astype(np.float32)
    deg = np.array(Wcsr.sum(axis=1)).ravel().astype(np.float32)
    deg_safe = np.maximum(deg, 1e-8)
    for _ in range(smooth_passes):
        S = (Wcsr @ S) / deg_safe[:, None]
    return S

def add_noise_to_target_snr(eeg, snr_db, rng):
    sig_pow = np.mean(eeg**2)
    noise_pow = sig_pow / (10**(snr_db/10))
    E = rng.standard_normal(eeg.shape).astype(np.float32) * np.sqrt(noise_pow)
    return eeg + E

def make_one_simulation(L, src_xyz, W, Fs, t, K, per_region_k, snr_db, seed,
                        Cs_for_beta="I", i_ref0=63):
    rng = np.random.default_rng(seed)
    p = src_xyz.shape[0]

    # pick K regions (simple: nearest around random centers)
    mid_gap_mm = 5.0
    allowed = np.where((src_xyz[:,0] <= -mid_gap_mm) | (src_xyz[:,0] >= mid_gap_mm))[0]

    silence_idx = []
    available = set(allowed.tolist())
    for _ in range(K):
        if not available:
            break
        center = rng.choice(list(available))
        d2 = np.sum((src_xyz - src_xyz[center])**2, axis=1)
        order = np.argsort(d2)
        pick = [i for i in order if i in available][:per_region_k]
        silence_idx.extend(pick)
        for i in pick:
            available.discard(i)

    silence_idx = np.unique(np.asarray(silence_idx, dtype=np.int64))
    gt_mask = np.zeros(p, dtype=bool)
    gt_mask[silence_idx] = True
    y = gt_mask.astype(np.float32)

    # simulate smooth sources + silence them
    S = simulate_sources_fast(W, t=t, rng=rng, smooth_passes=3)
    S[gt_mask, :] = 0.0

    eeg_clean = (L @ S).astype(np.float32)
    eeg = add_noise_to_target_snr(eeg_clean, snr_db, rng)

    # choose Cs inside beta (DEPLOYABLE: identity)
    Cs_beta = np.eye(p, dtype=np.float64)

    beta_raw = silencemap_beta_highres_fast(eeg=eeg, L=L, Cs=Cs_beta, Fs=Fs, i_ref0=i_ref0)
    beta = (beta_raw - beta_raw.mean()) / (beta_raw.std() + 1e-8)
    beta = beta.astype(np.float32)

    beta_z = (beta - beta.mean()) / (beta.std() + 1e-8)
    nbr_mean, nbr_std, deg2 = neighbor_stats_from_W(W, beta_z)
    delta = beta_z - nbr_mean
    logdeg = np.log(deg2 + 1.0).astype(np.float32)

    X = np.column_stack([beta_z, delta, nbr_std, logdeg]).astype(np.float32)
    return X, y, gt_mask

# ----------------- GNN -----------------
class BetaGNN(nn.Module):
    def __init__(self, in_dim, hidden, Ahat):
        super().__init__()
        self.lin_in  = nn.Linear(in_dim, hidden, bias=True)
        self.lin_mp1 = nn.Linear(hidden, hidden, bias=False)
        self.lin_mp2 = nn.Linear(hidden, hidden, bias=False)
        self.lin_out = nn.Linear(hidden, 1, bias=True)
        self.Ahat = Ahat

    def forward(self, X):
        H = torch.relu(self.lin_in(X))
        AH  = torch.sparse.mm(self.Ahat, H)
        A2H = torch.sparse.mm(self.Ahat, AH)
        H = torch.relu(self.lin_mp1(AH) + self.lin_mp2(A2H))
        return self.lin_out(H)

# ----------------- Metrics -----------------
def pr_re_f1(pred_mask, gt_mask):
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2*prec*rec / (prec+rec + 1e-12)
    return float(prec), float(rec), float(f1)

def mask_connected_components(W, mask):
    mask = np.asarray(mask).astype(bool).reshape(-1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    W = W.tocsr()
    sub = W[idx][:, idx]
    sub = ((sub + sub.T) > 0).astype(int)
    n_comp, labels = connected_components(sub, directed=False, connection="weak")
    comps = []
    for c in range(n_comp):
        comps.append(idx[labels == c])
    comps.sort(key=lambda a: -a.size)
    return comps

def jaccard_set(a, b):
    a = set(a); b = set(b)
    if len(a) == 0 and len(b) == 0: return 1.0
    if len(a) == 0 or len(b) == 0:  return 0.0
    return len(a & b) / (len(a | b) + 1e-12)

def component_centroid(nodes, coords):
    pts = coords[nodes]
    return pts.mean(axis=0)

def evaluate_mean_metrics(W, gt_mask, pred_mask, coords):
    gt_comps = mask_connected_components(W, gt_mask)
    pr_comps = mask_connected_components(W, pred_mask)
    if len(gt_comps) == 0 or len(pr_comps) == 0:
        return {"mean_jaccard": np.nan, "mean_com_error": np.nan, "mean_size_error": np.nan}

    gt_cent = np.stack([component_centroid(c, coords) for c in gt_comps], axis=0)
    pr_cent = np.stack([component_centroid(c, coords) for c in pr_comps], axis=0)

    # Greedy match by centroid distance
    used = np.zeros(len(pr_comps), dtype=bool)
    j_list, com_list, size_list = [], [], []

    for i, g in enumerate(gt_comps):
        d = np.linalg.norm(pr_cent - gt_cent[i:i+1], axis=1)
        d[used] = 1e9
        j = int(np.argmin(d))
        if d[j] > 1e8:
            continue
        used[j] = True

        gset = g.tolist()
        pset = pr_comps[j].tolist()

        jac = jaccard_set(gset, pset)
        com = float(np.linalg.norm(gt_cent[i] - pr_cent[j]))
        size_rel = abs(len(gset) - len(pset)) / (len(gset) + 1e-12)

        j_list.append(jac)
        com_list.append(com)
        size_list.append(size_rel)

    return {
        "mean_jaccard": float(np.mean(j_list)) if j_list else np.nan,
        "mean_com_error": float(np.mean(com_list)) if com_list else np.nan,
        "mean_size_error": float(np.mean(size_list)) if size_list else np.nan,
    }

# ----------------- Dataset cache -----------------
def make_or_load_dataset(cache_path, L, src_xyz, W, args):
    cache_path = Path(cache_path)
    if cache_path.exists() and not args.force_regen:
        data = np.load(cache_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    rng = np.random.default_rng(args.seed)
    train_seeds = rng.integers(0, 10_000_000, size=args.n_train_sims, dtype=np.int64)
    test_seeds  = rng.integers(0, 10_000_000, size=args.n_test_sims, dtype=np.int64)

    # ensure disjoint (just in case)
    test_seeds = np.array([s for s in test_seeds if s not in set(train_seeds.tolist())], dtype=np.int64)
    if test_seeds.size < args.n_test_sims:
        extra = rng.integers(0, 10_000_000, size=(args.n_test_sims - test_seeds.size), dtype=np.int64)
        test_seeds = np.concatenate([test_seeds, extra], axis=0)

    Xtr = np.zeros((args.n_train_sims, src_xyz.shape[0], args.in_dim), dtype=np.float32)
    ytr = np.zeros((args.n_train_sims, src_xyz.shape[0]), dtype=np.float32)
    mtr = np.zeros((args.n_train_sims, src_xyz.shape[0]), dtype=np.uint8)

    Xte = np.zeros((args.n_test_sims,  src_xyz.shape[0], args.in_dim), dtype=np.float32)
    yte = np.zeros((args.n_test_sims,  src_xyz.shape[0]), dtype=np.float32)
    mte = np.zeros((args.n_test_sims,  src_xyz.shape[0]), dtype=np.uint8)

    def gen_one(seed, tag, i, Xarr, yarr, marr):
        t0 = time.time()
        X, y, m = make_one_simulation(
            L, src_xyz, W,
            Fs=args.Fs, t=args.t, K=args.K, per_region_k=args.per_region_k,
            snr_db=args.snr_db, seed=int(seed),
            Cs_for_beta="I",
            i_ref0=args.i_ref0
        )
        Xarr[i] = X
        yarr[i] = y
        marr[i] = m.astype(np.uint8)
        print(f"[cache] {tag} {i+1}/{len(Xarr)} seed={int(seed)}  ({time.time()-t0:.2f}s)", flush=True)

    for i, s in enumerate(train_seeds):
        gen_one(s, "TRAIN", i, Xtr, ytr, mtr)
    for i, s in enumerate(test_seeds[:args.n_test_sims]):
        gen_one(s, "TEST", i, Xte, yte, mte)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=Xtr, y_train=ytr, m_train=mtr, seeds_train=train_seeds,
        X_test=Xte,  y_test=yte,  m_test=mte,  seeds_test=test_seeds[:args.n_test_sims],
    )
    print(f"[cache] saved → {cache_path}", flush=True)
    return {
        "X_train": Xtr, "y_train": ytr, "m_train": mtr, "seeds_train": train_seeds,
        "X_test": Xte, "y_test": yte, "m_test": mte, "seeds_test": test_seeds[:args.n_test_sims],
    }

# ----------------- Train / Eval -----------------
def eval_split(model, X, m, W, coords, K, per_region_k, device):
    model.eval()
    k = int(K * per_region_k)
    f1s, jacs, coms, sizes = [], [], [], []

    with torch.no_grad():
        for i in range(X.shape[0]):
            Xt = torch.tensor(X[i], dtype=torch.float32, device=device)
            logits = model(Xt).squeeze(1).detach().cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logits))

            idx = np.argpartition(prob, -k)[-k:]
            pred = np.zeros_like(prob, dtype=bool)
            pred[idx] = True
            gt = m[i].astype(bool)

            _, _, f1 = pr_re_f1(pred, gt)
            comp = evaluate_mean_metrics(W, gt, pred, coords)

            f1s.append(f1)
            jacs.append(comp["mean_jaccard"])
            coms.append(comp["mean_com_error"])
            sizes.append(comp["mean_size_error"])

    return {
        "F1@k": float(np.nanmean(f1s)),
        "mean_jaccard": float(np.nanmean(jacs)),
        "mean_com_error": float(np.nanmean(coms)),
        "mean_size_error": float(np.nanmean(sizes)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_mat", action="store_true")
    ap.add_argument("--leadfield_path", type=str, required=True)
    ap.add_argument("--headmodel_path", type=str, required=True)
    ap.add_argument("--kNN", type=int, default=12)
    ap.add_argument("--sigmaW", type=float, default=12.0)

    ap.add_argument("--Fs", type=int, default=512)
    ap.add_argument("--t", type=int, default=800)

    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--per_region_k", type=int, default=12)
    ap.add_argument("--snr_db", type=float, default=10.0)
    ap.add_argument("--i_ref0", type=int, default=63)

    ap.add_argument("--n_train_sims", type=int, default=10)
    ap.add_argument("--n_test_sims", type=int, default=5)
    ap.add_argument("--cache", type=str, default="/content/sample_data/cache_dataset.npz")
    ap.add_argument("--force_regen", action="store_true")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda_smooth", type=float, default=5.0)

    args = ap.parse_args()
    args.in_dim = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    print("Loading leadfield/headmodel...", flush=True)
    L, src_xyz = load_leadfield_any(args.leadfield_path, args.headmodel_path)
    n, p = L.shape
    print("Loaded L:", L.shape, "src_xyz:", src_xyz.shape, flush=True)

    W, deg = knn_graph_gauss(src_xyz, k=args.kNN, sigma=args.sigmaW)
    L_g, _ = laplacian_from_W(W)
    Ahat = build_torch_graph(W, deg, device=device)
    Lcoo = L_g.tocoo()

    data = make_or_load_dataset(args.cache, L, src_xyz, W, args)

    model = BetaGNN(in_dim=args.in_dim, hidden=args.hidden, Ahat=Ahat).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # imbalance weight computed from average k
    k_pos = args.K * args.per_region_k
    pos_weight = torch.tensor([(p - k_pos) / (k_pos + 1e-12)], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("\nTraining on cached TRAIN set (fast)...", flush=True)
    model.train()
    for it in range(args.steps):
        j = np.random.randint(data["X_train"].shape[0])
        X_np = data["X_train"][j]
        y_np = data["y_train"][j]

        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_np, dtype=torch.float32, device=device).view(-1, 1)

        opt.zero_grad()
        logits = model(X_t)
        prob = torch.sigmoid(logits)

        cls = bce(logits, y_t)
        sm  = lap_energy(prob, Lcoo, device=device) / p
        loss = cls + args.lambda_smooth * sm

        loss.backward()
        opt.step()

        if (it+1) % 50 == 0:
            print(f"[{it+1:04d}] loss={loss.item():.5f} cls={cls.item():.5f} smooth={sm.item():.5f}", flush=True)

    print("\nEvaluating (no train=test leakage):", flush=True)
    tr = eval_split(model, data["X_train"], data["m_train"], W, src_xyz, args.K, args.per_region_k, device)
    te = eval_split(model, data["X_test"],  data["m_test"],  W, src_xyz, args.K, args.per_region_k, device)

    print("\n=== TRAIN (in-sample sanity) ===")
    for k,v in tr.items():
        print(f"{k}: {v}")

    print("\n=== TEST (held-out simulations) ===")
    for k,v in te.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
