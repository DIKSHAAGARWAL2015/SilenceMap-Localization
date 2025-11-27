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


def plot_ranked_clusters_numbered_with_mask(coords, mask, ranked_node_groups,ranked_clusters,
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
        print(rank_id)
        nodes_arr = np.array(nodes, dtype=int)
        centroid = coords[nodes_arr].mean(axis=0)
         # 1. Rank number in GREEN
        ax.text(
            centroid[0]+2, centroid[1]+2, centroid[2]+2,
            f"R{rank_id}",
            color="green",
            fontsize=5,
            ha='center', va='center',
            weight='bold')

    for item in ranked_clusters:
        cluster_id = item["cluster_id"]
        nodes_1 = item["nodes"]          # this is your list of node indices

        nodes_arr_1 = np.array(nodes_1, dtype=int)
        centroid = coords[nodes_arr_1].mean(axis=0)
    # 2. Cluster ID in ORANGE (slightly shifted upward)
        ax.text(
            centroid[0]+6, centroid[1]+6, centroid[2]+6,   # small upward shift
            f"C{cluster_id}",
            color="black",
            fontsize=5,
             ha='center', va='bottom',
            weight='bold')
    ax.scatter([], [], [], c='green', label='Rank Number')
    ax.scatter([], [], [], c='black', label='Cluster ID')

    ax.legend(loc='upper left')


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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def rotate_cortex_360(coords, mask, title="", outdir="./figs/",
                      fname="rotate_brain.gif",
                      steps=60, elev=20, point_size=8):

    coords = np.asarray(coords)
    mask   = np.asarray(mask, dtype=bool)

    # Bright for masked nodes, dim for non-mask
    c = np.where(mask, 1.0, 0.1)

    # Create output path
    os.makedirs(outdir, exist_ok=True)
    temp_frames = []

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=c, s=point_size, cmap="cool"
    )
    ax.set_title(title)
    ax.set_axis_off()

    # Rotate for 360 degrees
    for azim in np.linspace(0, 360, steps):
        ax.view_init(elev=elev, azim=azim)

        # Save each frame to buffer
        frame_path = f"{outdir}/frame_{int(azim)}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        temp_frames.append(frame_path)

    plt.close(fig)

    # Create GIF
    images = []
    for frame_path in temp_frames:
        images.append(imageio.imread(frame_path))
    imageio.mimsave(f"{outdir}/{fname}", images, duration=0.05)

    # Clean temp frames
    for frame_path in temp_frames:
        os.remove(frame_path)

    print(f"Saved rotating brain GIF: {outdir}/{fname}")
