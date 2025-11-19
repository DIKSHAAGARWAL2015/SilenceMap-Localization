# SilenceMap-Localization
SilenceMap Localization
Created multiple regions of silence. The result is stored in the rep folder. The number of region of silence can vary from 1-5.



<img width="562" height="372" alt="image" src="https://github.com/user-attachments/assets/4677c6b3-9c48-40b5-8511-1d4b4524f201" />
# SilenceMap + Self-Supervised GNN for Multi-Region Silence Detection

This repository implements a **multi-region SilenceMap simulation** on a cortical mesh and compares:

- a **graph Laplacian baseline**, and  
- a **self-supervised GNN (`BetaGNN`) trained on a single β vector**

for detecting **silent source regions** in the brain.

The code is modular, so each part of the pipeline (data loading, EEG simulation, β computation, graph construction, GNN, clustering, plotting) lives in its own file.

---

## 🎯 High-Level Pipeline

1. **Load leadfield & cortex geometry** from `.mat` (or generate a synthetic one).
2. **Simulate multi-region silence** on the cortex and generate EEG.
3. **Compute β** (currently: **binary** from ground-truth silence; optionally EEG-based).
4. **Build a k-NN graph** over cortical sources and compute the **Laplacian**.
5. Run the **Laplacian baseline** to get a smooth estimate `g_lap`.
6. Train the **self-supervised GNN** on a single β vector to get `g_gnn`.
7. Threshold `g_lap` and `g_gnn` to get silent masks, compute **Precision/Recall/F1**.
8. Extract **connected components**, rank clusters, and visualize the results.

---

## 📁 Repository Structure

Each file has a clear role in the pipeline:

### Core entrypoint

- **`main.py`**  
  Main script that ties everything together. It:
  - Parses command-line arguments (e.g., `--use_mat`, `--save_figs`, paths to `.mat` files).
  - Loads leadfield and cortex vertices (from `.mat` or synthetic).
  - Calls `simulate_multiregion_silence_and_eeg` to generate EEG and the true silent mask.
  - Computes β using `beta.py`.
  - Builds the graph and Laplacian (`graph.py`).
  - Runs the **Laplacian baseline** and **GNN**.
  - Computes metrics (Precision/Recall/F1).
  - Extracts & ranks silent clusters (`clusters_ranking.py`).
  - Generates all plots (`plotting.py`).

---

### Data loading

- **`dataloader.py`**  
  Responsible for loading or creating the forward model:
  - `load_leadfield_any(leadfield_path, headmodel_path, leadfield_var=None, vertices_var=None)`  
    Robust loader for MATLAB `.mat` leadfield and cortex vertices using `mat73`, even when the variable names are nested or unknown.
  - `_walk_py`, `_pick_leadfield_from_py`, `_pick_vertices_from_py`  
    Internal helpers to walk the loaded `.mat` structure and automatically find a suitable leadfield matrix and vertex array.
  - `make_synthetic_leadfield(p=1662, n=128, seed=0)`  
    Creates a synthetic leadfield and random 3D coordinates if `.mat` loading is disabled or fails.

---

### EEG simulation

- **`compute_eeg.py`**  
  Handles **multi-region silence simulation** and EEG generation:
  - `butter_lowpass_filter(data, fs, cutoff=90.0, order=4)`  
    Simple low-pass filter applied over time to EEG signals.
  - `simulate_multiregion_silence_and_eeg(L, src_xyz, K, per_region_k, t, Fs, noise_pow, mid_gap_mm, rng)`  
    - Selects `K` silent regions of size `per_region_k` on the cortex (avoiding the midline by `mid_gap_mm` if desired).
    - Builds a **distance-based exponential covariance matrix** over sources.
    - Samples Gaussian source activity and zeros it out in the silent regions.
    - Generates EEG: `eeg = L @ S + noise` with low-pass filtering.
    - Returns:
      - `X_act` – boolean mask of true silent nodes (`True = silent`).
      - `eeg` – simulated EEG `(n, t)`.
      - `snr_db` – average SNR estimate.
      - `Cs_full` – full source covariance matrix.

---

### β (beta) computation

- **`beta.py`**  
  Contains functions for defining β, which is the **per-node “activity” measure** used by both the Laplacian baseline and the GNN.
  - `beta_from_oracle_silence(X_act)`  
    - Builds a **binary β**:
      - `β = 0` for silent nodes.
      - `β = 1` for active nodes.
  - `beta_from_eeg(L, eeg)`  
    - Computes `C_eeg = (eeg @ eegᵀ) / T`,  
      then `Aᵀ C_eeg A`, and uses the **diagonal** as an EEG-driven β.
    - Normalizes β into `[0, 1]`.
  - `mix_beta(beta_eeg, beta_oracle, alpha=0.0)`  
    - Linearly mixes EEG-based and oracle β:
      - `β = (1 - α) * β_eeg + α * β_oracle`
    - Re-normalizes to `[0, 1]`.

> Currently, `main.py` uses **oracle β** by default, but can be easily switched to EEG-based or mixed β.

---

### GNN model

- **`gnn.py`**  
  Defines the self-supervised GNN used to refine β:
  - `device` – chooses `"cuda"` if available, otherwise `"cpu"`.
  - `class BetaGNN(nn.Module)`  
    - Node features: `[β, β², degree]`.
    - Two-hop message passing via `Ahat` and `Ahat²`.
    - Uses ReLU non-linearities and a final `softplus` output to ensure `g ≥ 0`.
    - Outputs `g` – a **continuous “silence score”** for each node.
  - `lap_energy(g, Lcoo)`  
    - Computes the graph smoothness term `gᵀ L g` using a sparse Laplacian `Lcoo`.

The GNN is trained **self-supervised**, using only:

- the single β vector,  
- graph smoothness, and  
- seed constraints (silent vs active seeds chosen from β).

---

### Graph and Laplacian

- **`graph.py`**  
  Builds the graph over cortical sources and converts it for PyTorch:
  - `knn_graph_gauss(coords, k=12, sigma=12.0)`  
    - Builds a **k-NN graph** over 3D coordinates (`coords`).
    - Edge weights are Gaussian / heat-kernel:  
      `w_ij = exp(-‖x_i - x_j‖² / (2 σ²))`.
    - Returns:
      - `W` – sparse adjacency matrix (COO).
      - `deg` – degree vector.
  - `laplacian_from_W(W)`  
    - Computes the **combinatorial Laplacian** `L = D - W`.
  - `build_torch_graph(W, deg)`  
    - Builds a normalized adjacency `Â = D^{-1/2} W D^{-1/2}` as a PyTorch sparse tensor.
    - This `Â` is passed into `BetaGNN` for message passing.

---

### Plotting

- **`plotting.py`**  
  All visualization utilities for cortex and curves:
  - `show_or_save(fig, name, save=False, outdir="./figs/")`  
    - Unified helper to **either show** or **save** figures depending on `--save_figs`.
  - `show_cortex_mask(coords, mask, title, save, outdir, fname)`  
    - 3D scatter of cortical vertices, with masked nodes highlighted (e.g., ground truth silent, Laplacian mask, or GNN mask).
  - `plot_ranked_clusters_numbered_with_mask(coords, mask, ranked_node_groups, ...)`  
    - Plots:
      - All nodes in soft blue,
      - Masked nodes in red,
      - Cluster **rank numbers** (1, 2, 3, …) at cluster centroids.
  - `plot_beta_g_curves(beta, g_lap, g_gnn, ...)`  
    - 2D line plot of the first 300 nodes showing:
      - β,
      - Laplacian output `g_lap`,
      - GNN output `g_gnn`.

---

### Clustering & ranking

- **`clusters_ranking.py`**  
  Post-processes the GNN’s silent mask:
  - `extract_clusters_from_mask_sparse(mask, W)`  
    - Restricts the graph `W` to the nodes where `mask=True`.
    - Runs connected-components to get **clusters of silent nodes**.
  - `rank_clusters(clusters, coords)`  
    - For each cluster, computes:
      - size,
      - mean intra-cluster distance,
      - max intra-cluster distance,
      - a **score = size − mean_distance** (larger & tighter clusters rank higher).
    - Returns a list of cluster dicts sorted in descending score.

---

### Package marker

- **`__init__.py`**  
  Empty (or minimal) file so Python can treat this directory as a package if imported.  
  You can later expose convenient imports here if you want.

---

## 🧪 How to Run

Install dependencies (in a fresh environment):

```bash
pip install -r requirements.txt
