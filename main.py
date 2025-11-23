#!/usr/bin/env python3
# main.py
# Run multi-region SilenceMap simulation + Laplacian baseline + self-supervised GNN.

import argparse
import numpy as np
import torch
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from dataloader import load_leadfield_any
from compute_eeg import simulate_multiregion_silence_and_eeg
from beta import beta_from_oracle_silence  # you can later swap to EEG-based beta
from graph import knn_graph_gauss, laplacian_from_W, build_torch_graph
from gnn import BetaGNN, lap_energy, device
from plotting import (
    show_cortex_mask,
    plot_ranked_clusters_numbered_with_mask,
    plot_beta_g_curves,
)
from clusters_ranking import extract_clusters_from_mask_sparse, rank_clusters
from eval_silence_localization import evaluate_silence_localization_multi_region

# ---------- Metrics ----------

def pr_re_f1(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


def stats_against_ground_truth(name, arr, X_act):
    arr = np.asarray(arr)
    X_act = np.asarray(X_act, dtype=bool)
    silent = X_act
    active = ~X_act

    print(f"\n[{name}]")
    print("  mean(silent) :", float(arr[silent].mean()))
    print("  mean(active) :", float(arr[active].mean()))
    print(
        "  corr with X_act:",
        float(np.corrcoef(arr, X_act.astype(np.float32))[0, 1]),
    )


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-region SilenceMap + Laplacian baseline + self-supervised GNN"
    )

    parser.add_argument(
        "--use_mat",
        action="store_true",
        help="Load real leadfield/headmodel from .mat files",
    )
    parser.add_argument(
        "--leadfield_path",
        type=str,
        default="./OT_leadfield_symmetric_1662-128.mat",
    )
    parser.add_argument(
        "--headmodel_path",
        type=str,
        default="./OT_headmodel_symmetric_1662-128.mat",
    )
    parser.add_argument(
        "--leadfield_var",
        type=str,
        default=None,
        help="Variable path for leadfield (e.g., 'L' or 'leadfield.L')",
    )
    parser.add_argument(
        "--vertices_var",
        type=str,
        default=None,
        help="Variable path for vertices (e.g., 'Cortex.Pial.vertices')",
    )

    parser.add_argument("--p", type=int, default=1662, help="Sources (synthetic)")
    parser.add_argument("--n", type=int, default=128, help="Sensors (synthetic)")
    parser.add_argument("--K", type=int, default=5, help="Silent regions")
    parser.add_argument("--per_region_k", type=int, default=10, help="Nodes/region")
    parser.add_argument("--t", type=int, default=10_000, help="Time points")
    parser.add_argument("--Fs", type=int, default=512, help="Sampling rate")

    parser.add_argument("--kNN", type=int, default=12, help="Graph kNN")
    parser.add_argument("--sigmaW", type=float, default=12.0, help="Graph RBF width")
    parser.add_argument(
        "--lambda_lap",
        type=float,
        default=1.0,
        help="Laplacian smoothness weight (baseline)",
    )

    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_steps", type=int, default=2000)
    parser.add_argument("--gnn_lr", type=float, default=1e-2)
    parser.add_argument(
        "--gnn_lambda",
        type=float,
        default=5.0,
        help="GNN smoothness term weight",
    )
    parser.add_argument(
        "--gnn_gamma",
        type=float,
        default=0.5,
        help="GNN seed term weight",
    )

    parser.add_argument(
        "--save_figs",
        action="store_true",
        help="Save figures instead of showing",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="./figs/",
        help="Directory for saved figures",
    )

    return parser.parse_args()


# ---------- Seeds from beta ----------

def select_seeds_from_beta(beta_arr, q_seed=0.03):
    """
    Select silent + active seeds from a single beta vector:

      - silent seeds → lowest beta values
      - active seeds → highest beta values
    """
    beta_arr = np.asarray(beta_arr, dtype=np.float32)
    p = len(beta_arr)
    k_seed = max(5, int(q_seed * p))

    idx_sorted = np.argsort(beta_arr)
    silent_seeds = idx_sorted[:k_seed]
    active_seeds = idx_sorted[-k_seed:]

    seed_silent_mask = np.zeros(p, dtype=bool)
    seed_active_mask = np.zeros(p, dtype=bool)
    seed_silent_mask[silent_seeds] = True
    seed_active_mask[active_seeds] = True

    return seed_silent_mask, seed_active_mask


# ---------- Main pipeline ----------

def run():
    args = parse_args()
    print("Device:", device)

    # --- Load leadfield + cortex ---
    if args.use_mat:
        print("Loading .mat leadfield/headmodel...")
        try:
            L, src_xyz = load_leadfield_any(
                args.leadfield_path,
                args.headmodel_path,
                leadfield_var=args.leadfield_var,
                vertices_var=args.vertices_var,
            )
            print("Loaded leadfield:", L.shape, "vertices:", src_xyz.shape)
        except Exception as e:
            print("Failed to parse .mat structures:", e)
            print("Falling back to synthetic leadfield.")
            args.use_mat = False

    p = src_xyz.shape[0]

    # --- Simulate silence + EEG ---
    X_act, eeg, snr, Cs_full = simulate_multiregion_silence_and_eeg(
        L=L,
        src_xyz=src_xyz,
        K=args.K,
        per_region_k=args.per_region_k,
        t=args.t,
        Fs=args.Fs,
    )
    print(f"Avg SNR ≈ {snr:.2f} dB")

    # --- Beta computation (currently: oracle binary) ---
    beta = beta_from_oracle_silence(X_act)

    # If later you want EEG-based beta, you can do:
    # from beta import beta_from_eeg, mix_beta
    # beta_eeg = beta_from_eeg(L, eeg)
    # beta = mix_beta(beta_eeg, beta, alpha=0.0)

    # --- Graph + Laplacian baseline ---
    W, deg = knn_graph_gauss(src_xyz, k=args.kNN, sigma=args.sigmaW)
    L_g, _ = laplacian_from_W(W)

    lam = args.lambda_lap
    I = identity(p, format="coo")
    A = (I + lam * L_g).tocsc()
    g_lap = spsolve(A, beta).astype(np.float32)
    g_lap -= g_lap.min()
    g_lap /= (g_lap.max() + 1e-12)

    # true #silent nodes
    k_silent = int(X_act.sum())
    q_silent = 100.0 * k_silent / p
    print(f"[auto] q_silent set to {q_silent:.2f}% for |S|={k_silent}/{p}")

    g_lap_arr = np.asarray(g_lap)
    idx_lap = np.argpartition(g_lap_arr, k_silent - 1)[:k_silent]
    mask_lap = np.zeros_like(g_lap_arr, dtype=bool)
    mask_lap[idx_lap] = True

    P, R, F1 = pr_re_f1(mask_lap, X_act)
    print(f"Laplacian: P={P:.3f} R={R:.3f} F1={F1:.3f}")

    # --- GNN (self-supervised on single beta) ---
    Ahat = build_torch_graph(W, deg)
    Lcoo = L_g.tocoo()

    beta_arr = np.asarray(beta)
    seed_silent_mask_np, seed_active_mask_np = select_seeds_from_beta(beta_arr, q_seed=0.03)
    seed_silent_mask = torch.tensor(seed_silent_mask_np, dtype=torch.bool, device=device)
    seed_active_mask = torch.tensor(seed_active_mask_np, dtype=torch.bool, device=device)

    model = BetaGNN(p=p, hidden=args.gnn_hidden, Ahat=Ahat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.gnn_lr)

    beta_t = torch.tensor(beta, dtype=torch.float32, device=device).view(-1, 1)
    deg_feat = torch.tensor(deg, dtype=torch.float32, device=device).view(-1, 1)

    for it in range(args.gnn_steps):
        opt.zero_grad()
        g = model(beta_t, deg_feat)

        data_term = ((g - beta_t) ** 2).mean()
        smooth_term = lap_energy(g, Lcoo) / p

        loss_silent = torch.mean(g[seed_silent_mask] ** 2)          # -> 0
        loss_active = torch.mean((g[seed_active_mask] - 1.0) ** 2)  # -> 1
        loss_seed = loss_silent + loss_active

        loss = data_term + args.gnn_lambda * smooth_term + args.gnn_gamma * loss_seed
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

    g_hat_arr = np.asarray(g_hat)
    idx_gnn = np.argpartition(g_hat_arr, k_silent - 1)[:k_silent]
    mask_gnn = np.zeros_like(g_hat_arr, dtype=bool)
    mask_gnn[idx_gnn] = True

    P, R, F1 = pr_re_f1(mask_gnn, X_act)
    print(f"GNN:       P={P:.3f} R={R:.3f} F1={F1:.3f}")

    # --- Clusters from GNN mask ---
    clusters = extract_clusters_from_mask_sparse(mask_gnn, W)
    print(f"Found {len(clusters)} GNN clusters among silent nodes.")
    
    ranked_clusters = rank_clusters(clusters, src_xyz)
    ranked_node_groups = [
        (rank_id, info["nodes"]) for rank_id, info in enumerate(ranked_clusters, start=1)
    ]
    for r, info in enumerate(ranked_clusters):
        print(f"Rank {r}: C{info['cluster_id']} "
          f"(size={info['size']}, radius={info['radius']:.2f}, "
          f"score={info['score']:.3f})")
    ranked_node_groups = [
    (rank_id, info["nodes"])
    for rank_id, info in enumerate(ranked_clusters)
    ]
    print(ranked_node_groups)
    # 1. Get clusters from masks (you already have this)
    gt_clusters_list = extract_clusters_from_mask_sparse(X_act, W)      # list[list[int]]
    pred_clusters_list = extract_clusters_from_mask_sparse(mask_gnn, W)  # list[list[int]]
    coords = src_xyz
    # 2. Rank them with YOUR function
    gt_ranked = rank_clusters(gt_clusters_list, coords)       # your rank_clusters
    pred_ranked = rank_clusters(pred_clusters_list, coords)

    # 3. Evaluate localization performance
    metrics = evaluate_silence_localization_multi_region(
    gt_ranked,
    pred_ranked,
    coords,
    top_k_gt=None,    # or e.g. 3 if you want top 3 GT regions
    top_k_pred=None,  # or e.g. 5
    )

    print("Mean Jaccard:", metrics["mean_jaccard"])
    print("Mean ΔCOM:", metrics["mean_delta_com"])
    print("Mean size rel error:", metrics["mean_size_rel_error"])

    for m in metrics["per_pair"]:
       print(
        f"GT id {m['gt_cluster_id']} ↔ Pred id {m['pred_cluster_id']}: "
        f"J={m['jaccard']:.3f}, ΔCOM={m['delta_com']:.2f}, "
        f"size_gt={m['size_gt']}, size_pred={m['size_pred']}"
       )

    # --- Stats + plots ---
    stats_against_ground_truth("beta", beta, X_act)
    stats_against_ground_truth("g_lap", g_lap, X_act)
    stats_against_ground_truth("g_gnn", g_hat, X_act)

    save = args.save_figs
    out = args.fig_dir

    show_cortex_mask(
        src_xyz,
        X_act,
        title=f"GT silent (K={args.K})",
        save=save,
        outdir=out,
        fname="1_gt.png",
    )
    show_cortex_mask(
        src_xyz,
        mask_lap,
        title=f"Laplacian mask (q={q_silent:.2f}%)",
        save=save,
        outdir=out,
        fname="2_laplacian.png",
    )
    show_cortex_mask(
        src_xyz,
        mask_gnn,
        title=f"GNN mask (q={q_silent:.2f}%)",
        save=save,
        outdir=out,
        fname="3_gnn.png",
    )

    plot_ranked_clusters_numbered_with_mask(
    src_xyz,
    mask_gnn,
    ranked_node_groups,ranked_clusters,
    title="GNN silent regions (ranked clusters)",
    save=args.save_figs,
    outdir=args.fig_dir,
    fname="gnn_ranked_clusters.png")

    plot_beta_g_curves(
        beta=beta,
        g_lap=g_lap,
        g_gnn=g_hat,
        save=save,
        outdir=out,
        fname="4_curves.png",
    )


def main():
    run()


if __name__ == "__main__":
    main()
