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

def plot_gt_gnn_clusters_3x5(
    coords,
    gt_mask,
    gnn_mask,
    ranked_clusters,
    title_prefix="GT vs GNN vs Clusters (5 views)",
    save=False,
    outdir="./figs/",
    fname="gt_gnn_clusters_3x5.png",
):
    """
    Make a 3 x 5 panel:

        Row 1: GT silent mask (X_act) in 5 views
        Row 2: GNN silent mask in 5 views
        Row 3: Ranked clusters (each cluster = color) in 5 views

    Columns are standard anatomical views:
        front, back, left, right, top
    """

    coords = np.asarray(coords)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    gnn_mask = np.asarray(gnn_mask, dtype=bool)

    # ----- cluster colors -----
    # use tab10 or tab20 depending on how many clusters you have
    n_clusters = len(ranked_clusters)
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))

    cluster_nodes_and_colors = []
    for i, info in enumerate(ranked_clusters):
        nodes = np.array(info["nodes"], dtype=int)
        color = cmap(i % cmap.N)
        cluster_nodes_and_colors.append((nodes, color, i))

    # ----- standard views: (name, elev, azim) -----
    views = [
        ("front", 20, 180),
        ("back",  20,   0),
        ("left",  20,  90),
        ("right", 20, -90),
        ("top",   90, 180),
    ]

    fig = plt.figure(figsize=(22, 12))

    # colors for masks (silent nodes bright, others dim)
    gt_c = np.where(gt_mask, 1.0, 0.1)
    gnn_c = np.where(gnn_mask, 1.0, 0.1)

    for col, (view_name, elev, azim) in enumerate(views, start=1):

        # -------- Row 1: GT --------
        ax1 = fig.add_subplot(3, 5, col, projection="3d")
        ax1.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=gt_c, s=6, cmap="cool"
        )
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_title(f"GT – {view_name}")
        ax1.set_axis_off()

        # -------- Row 2: GNN --------
        ax2 = fig.add_subplot(3, 5, 5 + col, projection="3d")
        ax2.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=gnn_c, s=6, cmap="cool"
        )
        ax2.view_init(elev=elev, azim=azim)
        ax2.set_title(f"GNN – {view_name}")
        ax2.set_axis_off()

        # -------- Row 3: Ranked clusters --------
        ax3 = fig.add_subplot(3, 5, 10 + col, projection="3d")

        # plot each cluster in its color
        for nodes, color, rank_id in cluster_nodes_and_colors:
            pts = coords[nodes]
            ax3.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=15,
                color=color,
                label=f"R{rank_id}"
            )

        ax3.view_init(elev=elev, azim=azim)
        ax3.set_title(f"Clusters – {view_name}")
        ax3.set_axis_off()

    fig.suptitle(title_prefix, fontsize=18)

    # Optional: one legend for clusters (outside the grid)
    if len(cluster_nodes_and_colors) > 0:
        # build dummy handles
        handles = []
        labels = []
        for nodes, color, rank_id in cluster_nodes_and_colors:
            h = plt.Line2D([0], [0], marker="o", linestyle="", color=color)
            handles.append(h)
            labels.append(f"Rank {rank_id}")
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            title="Cluster ranks",
        )

    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, fname)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_ventral_clusters_2x2(src_xyz, mask_gt, mask_gnn,
                               injection_side="left",
                               outdir="./figs_ventral/",
                               fname="ventral_clusters_2x2.png"):
    """
    Plot GT vs GNN clusters in a single 2x2 grid figure:
    Row 0: GT, Row 1: GNN
    Column 0: Ipsilateral, Column 1: Contralateral
    Lighter hemisphere background so clusters are visible.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(outdir, exist_ok=True)

    # Hemisphere labels: 0 = left, 1 = right
    hemisphere_labels = np.zeros(src_xyz.shape[0], dtype=int)
    hemisphere_labels[src_xyz[:, 0] > 0] = 1

    # Split nodes
    left_gt = np.where(mask_gt & (hemisphere_labels == 0))[0]
    right_gt = np.where(mask_gt & (hemisphere_labels == 1))[0]
    left_gnn = np.where(mask_gnn & (hemisphere_labels == 0))[0]
    right_gnn = np.where(mask_gnn & (hemisphere_labels == 1))[0]

    # Determine ipsi / contra
    if injection_side.lower() == "left":
        ipsi_gt, contra_gt = left_gt, right_gt
        ipsi_gnn, contra_gnn = left_gnn, right_gnn
        ipsi_label, contra_label = "Left Hemisphere", "Right Hemisphere"
    else:
        ipsi_gt, contra_gt = right_gt, left_gt
        ipsi_gnn, contra_gnn = right_gnn, left_gnn
        ipsi_label, contra_label = "Right Hemisphere", "Left Hemisphere"

    # 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection':'3d'})

    plot_info = [
        (ipsi_gt, axes[0,0], f"GT Ipsilateral ({ipsi_label})"),
        (contra_gt, axes[0,1], f"GT Contralateral ({contra_label})"),
        (ipsi_gnn, axes[1,0], f"GNN Ipsilateral ({ipsi_label})"),
        (contra_gnn, axes[1,1], f"GNN Contralateral ({contra_label})"),
    ]

    for nodes, ax, title in plot_info:
        ax.set_title(title, fontsize=12)

        # Lighter hemisphere background
        left_idx = np.where(hemisphere_labels == 0)[0]
        right_idx = np.where(hemisphere_labels == 1)[0]
        ax.scatter(src_xyz[left_idx,0], src_xyz[left_idx,1], src_xyz[left_idx,2],
                   s=10, color='lightsteelblue', alpha=0.2)
        ax.scatter(src_xyz[right_idx,0], src_xyz[right_idx,1], src_xyz[right_idx,2],
                   s=10, color='lightpink', alpha=0.2)

        # Highlight cluster nodes in bright orange with black edge
        if len(nodes) > 0:
            ax.scatter(src_xyz[nodes,0], src_xyz[nodes,1], src_xyz[nodes,2],
                       s=80, color='orange', edgecolor='k', linewidth=0.5)

        ax.set_axis_off()
        ax.view_init(elev=15, azim=-60)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close()
    print(f"Saved 2x2 figure to {os.path.join(outdir, fname)}")


