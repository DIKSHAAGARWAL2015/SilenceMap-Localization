# beta.py
# Beta computation utilities (oracle + EEG-based options).

import numpy as np


def beta_from_oracle_silence(X_act):
    """
    Binary beta from ground-truth silence mask.

    X_act : (p,) bool, True = silent
    Returns beta : (p,) float32, 0 = silent, 1 = active
    """
    X_act = np.asarray(X_act, dtype=bool)
    beta = np.ones(X_act.shape[0], dtype=np.float32)
    beta[X_act] = 0.0
    return beta


def beta_from_eeg(L, eeg):
    """
    Beta from EEG covariance:

        C_eeg = (eeg @ eeg^T) / T
        A^T C_eeg A
        beta_eeg = diag(A^T C_eeg A)

    Returns normalized beta in [0,1].
    """
    L = np.asarray(L, dtype=np.float32)
    eeg = np.asarray(eeg, dtype=np.float32)

    t = eeg.shape[1]
    Ceeg = (eeg @ eeg.T) / float(t)         # (n,n)
    AtA = L.T @ Ceeg @ L                    # (p,p)
    beta_eeg = np.diag(AtA).astype(np.float32)

    beta_eeg -= beta_eeg.min()
    beta_eeg /= (beta_eeg.max() + 1e-12)
    return beta_eeg


def mix_beta(beta_eeg, beta_oracle, alpha=0.0):
    """
    Linear mixture of EEG-derived beta and oracle beta.

    beta = (1 - alpha) * beta_eeg + alpha * beta_oracle
    """
    beta_eeg = np.asarray(beta_eeg, dtype=np.float32)
    beta_oracle = np.asarray(beta_oracle, dtype=np.float32)

    beta = (1.0 - alpha) * beta_eeg + alpha * beta_oracle
    beta -= beta.min()
    beta /= (beta.max() + 1e-12)
    return beta
