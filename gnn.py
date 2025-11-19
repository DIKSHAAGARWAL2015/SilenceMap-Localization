# gnn.py
# Self-supervised GNN model + Laplacian energy.

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BetaGNN(nn.Module):
    """
    Self-supervised GNN that operates on a single beta vector.

    Node features: [β, β^2, degree]
    Two-hop message passing + MLP readout, softplus output (>= 0).
    """

    def __init__(self, p, hidden, Ahat):
        super().__init__()
        self.lin_in = nn.Linear(3, hidden, bias=True)
        self.lin_mp1 = nn.Linear(hidden, hidden, bias=False)
        self.lin_mp2 = nn.Linear(hidden, hidden, bias=False)
        self.lin_out = nn.Linear(hidden, 1, bias=True)
        self.Ahat = Ahat

    def mp(self, H):
        AH = torch.sparse.mm(self.Ahat, H)
        A2H = torch.sparse.mm(self.Ahat, AH)
        return AH, A2H

    def forward(self, beta, degree):
        # beta, degree: (p,1)
        x = torch.cat([beta, beta ** 2, degree], dim=1)  # (p,3)
        H = torch.relu(self.lin_in(x))
        AH, A2H = self.mp(H)
        H = torch.relu(self.lin_mp1(AH) + self.lin_mp2(A2H))
        g = torch.nn.functional.softplus(self.lin_out(H))
        return g


def lap_energy(g, Lcoo):
    """
    Graph smoothness term: g^T L g.

    g : (p,1) torch.Tensor
    Lcoo : scipy.sparse.coo_matrix
    """
    i = torch.tensor(
        np.vstack([Lcoo.row, Lcoo.col]).astype(np.int64),
        device=device,
    )
    v = torch.tensor(Lcoo.data.astype(np.float32), device=device)
    L_t = torch.sparse_coo_tensor(i, v, Lcoo.shape, device=device).coalesce()
    Lg = torch.sparse.mm(L_t, g)
    return (g * Lg).sum()
