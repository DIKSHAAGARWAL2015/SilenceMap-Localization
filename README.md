# SilenceMap-Localization
Created multiple regions of silence.The number of regions of silence can vary from 1-5.



<img width="562" height="372" alt="image" src="https://github.com/user-attachments/assets/4677c6b3-9c48-40b5-8511-1d4b4524f201" />

# Multi-Region Silence Detection

## Repository Structure
main.py               → Runs the entire pipeline (loading → EEG → beta → graph → GNN → clustering → plots)

dataloader.py         → Loads leadfield & cortex from .mat files

compute_eeg.py        → Simulates multi-region silence, generates EEG, computes SNR

beta.py               → Computes beta values

graph.py              → Builds k-NN graph

gnn.py                → Defines the BetaGNN model used in self-supervised GNN optimization

clusters_ranking.py   → Extracts silent clusters from GNN/Laplacian masks and ranks them by size & compactness

plotting.py           → Produces all visualizations: cortex masks, GNN clusters, beta/g curves

requirements.txt      → Lists Python dependencies

__init__.py           → Marks the repository as a Python package
''
figs/                 → Stores generated figures

---
Results for binary beta:
### Ground Truth Silent Regions
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/7a38d847-85b6-44e4-8c55-e5f54decc5ac" />

### GNN Output
<img width="400" height="402" alt="image" src="https://github.com/user-attachments/assets/e08378e8-7cd3-4903-b6b3-2e929ddaaeab" />

### Ranked Silent Clusters (GNN) [ 0 - best and number in this fig corresponds to rank of clusters]
<img width="843" height="876" alt="image" src="https://github.com/user-attachments/assets/15281544-5fdf-4ce7-9258-2922de56e177" />

## Result inference:

RANK 0 → cluster 2

RANK 1 → cluster 4

RANK 2 → cluster 3

RANK 3 → cluster 1

RANK 4 → cluster 0

## Top clusters (by size & compactness):

Cluster 2: size=13, mean_dist=12.117, score=0.883
  
Cluster 4: size=12, mean_dist=13.365, score=-1.365
  
Cluster 3: size=10, mean_dist=13.397, score=-3.397
  
Cluster 1: size=9, mean_dist=13.542, score=-4.542
  
Cluster 0: size=6, mean_dist=12.535, score=-6.535
  
size = number of nodes in each cluster, score = size - mean_dist, mean_dist = within a cluster calculated all distances and averaged out.
## 🧪 How to Run

Install dependencies (in a fresh environment):

```bash
pip install -r requirements.txt
```cli
!python /content/SilenceMap-Localization/SilenceMap-Localization/main.py --save_figs --use_mat \
  --leadfield_path /content/sample_data/OT_leadfield_symmetric_1662-128.mat \
  --headmodel_path /content/sample_data/OT_headmodel_symmetric_1662-128.mat
Make sure leadfield and cortex are present



