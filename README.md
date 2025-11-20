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
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/58bfb83c-a7aa-4dc9-8bfb-e9538cf5368c" />


### GNN Output
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/47911863-1df0-4603-9f9c-52f42a274335" />


### Ranked Silent Clusters (GNN) [ 0 - best and number in this fig corresponds to rank of clusters]
<img width="863" height="895" alt="image" src="https://github.com/user-attachments/assets/8e19fb15-5483-4913-ac0d-6d8fcb213051" />


## Result inference:

RANK 0 → cluster 2

RANK 1 → cluster 4

RANK 2 → cluster 3

RANK 3 → cluster 1

RANK 4 → cluster 0

## Top clusters (by size & compactness):

Cluster 2: size=13, radius=8.33
Cluster 4: size=12, radius=9.02
Cluster 3: size=10, radius=8.95
Cluster 1: size=9, radius=9.07
Cluster 0: size=6, radius=8.19
  
size = number of nodes in each cluster, radius = mean Euclidean distance of cluster nodes to their centroid. So 
## 🧪 How to Run

Install dependencies (in a fresh environment):

```bash
pip install -r requirements.txt
```cli
!python /content/SilenceMap-Localization/SilenceMap-Localization/main.py --save_figs --use_mat \
  --leadfield_path /content/sample_data/OT_leadfield_symmetric_1662-128.mat \
  --headmodel_path /content/sample_data/OT_headmodel_symmetric_1662-128.mat
Make sure leadfield and cortex are present



