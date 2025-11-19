# plotting.py
# 3D cortex plotting + beta/g curves.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
    coords = np.asarray(coords)
    mask = np.asarray(mask, dtype=bool)

    c = np.where(mask, 1.0, 0.1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=c, s=8, cmap="cool")
    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()
    show_or_save(fig, fname, save=save, outdir=outdir)


def plot_ranked_clusters_numbered_with_mask(
    coords,
    mask,
    ranked_node_groups,
    title="Ranked clusters (1 = best)",
    save=False,
    outdir="./figs/",
    fname="gnn_ranked_clusters.png",
):
    coords = np.asarray(coords)
    mask = np.asarray(mask, dtype=bool)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Background
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c="#b0c4ff",
        s=8,
        edgecolors="none",
        alpha=0.4,
    )

    # Masked
    idx = np.where(mask)[0]
    ax.scatter(
        coords[idx, 0],
        coords[idx, 1],
        coords[idx, 2],
        c="#ff4444",
        s=18,
        edgecolors="none",
        alpha=0.9,
    )

    # Rank labels
    for rank_id, nodes in ranked_node_groups:
        nodes_arr = np.array(nodes, dtype=int)
        centroid = coords[nodes_arr].mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            centroid[2],
            str(rank_id),
            color="black",
            fontsize=12,
            ha="center",
            va="center",
            weight="bold",
        )

    ax.view_init(elev=20, azim=40)
    ax.set_title(title)
    ax.set_axis_off()

    show_or_save(fig, fname, save=save, outdir=outdir)


def plot_beta_g_curves(beta, g_lap, g_gnn, save=False, outdir="./figs/", fname="curves.png"):
    beta = np.asarray(beta)
    g_lap = np.asarray(g_lap)
    g_gnn = np.asarray(g_gnn)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(beta[:300], label="beta")
    plt.plot(g_lap[:300], label="g_lap")
    plt.plot(g_gnn[:300], label="g_gnn")
    plt.legend()
    plt.title("First 300 nodes")
    show_or_save(fig, fname, save=save, outdir=outdir)
