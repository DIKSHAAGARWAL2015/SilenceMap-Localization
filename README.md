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
<img width="797" height="841" alt="image" src="https://github.com/user-attachments/assets/7a38d847-85b6-44e4-8c55-e5f54decc5ac" />


### GNN Output
<img width="581" height="602" alt="image" src="https://github.com/user-attachments/assets/e08378e8-7cd3-4903-b6b3-2e929ddaaeab" />

### Ranked Silent Clusters (GNN)
<img width="582" height="602" alt="image" src="https://github.com/user-attachments/assets/4eb5090c-46dd-4f57-bd16-ef5fcfc9f981" />

## 🧪 How to Run

Install dependencies (in a fresh environment):

```bash
pip install -r requirements.txt
```cli
!python /content/SilenceMap-Localization/SilenceMap-Localization/main.py --save_figs --use_mat \
  --leadfield_path /content/sample_data/OT_leadfield_symmetric_1662-128.mat \
  --headmodel_path /content/sample_data/OT_headmodel_symmetric_1662-128.mat
Make sure leadfield and cortex are present



