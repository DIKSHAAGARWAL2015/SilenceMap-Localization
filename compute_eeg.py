# compute_eeg.py
# Multi-region silence + EEG simulation.

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist


def butter_lowpass_filter(data, fs, cutoff=90.0, order=4):
    """
    Low-pass filter along last axis (time).
    """
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, data, axis=-1)


def simulate_multiregion_silence_and_eeg(
    L,
    src_xyz,
    K=5,
    per_region_k=10,
    t=10_000,
    Fs=512,
    noise_pow=5e-8,
    mid_gap_mm=5.0,
    rng=None,
):
    """
    Simulate:
      - K silent regions
      - Gaussian sources with spatial covariance
      - EEG recordings with additive noise

    Returns
    -------
    X_act : (p,) bool           # True = silent
    eeg : (n,t) float32
    snr_db : float
    Cs_full : (p,p) float32
    """
    if rng is None:
        rng = np.random.default_rng(42)

    p = src_xyz.shape[0]
    n = L.shape[0]

    # Disallow midline band
    allowed = np.where(
        (src_xyz[:, 0] <= -mid_gap_mm) | (src_xyz[:, 0] >= mid_gap_mm)
    )[0]

    silence_idx = []
    available = set(allowed.tolist())
    for _ in range(K):
        if not available:
            break
        center = rng.choice(list(available))
        d2 = np.sum((src_xyz - src_xyz[center]) ** 2, axis=1)
        order = np.argsort(d2)
        pick = [i for i in order if i in available][:per_region_k]
        silence_idx.extend(pick)
        for i in pick:
            if i in available:
                available.remove(i)

    silence_idx = np.unique(silence_idx)

    X_act = np.zeros(p, dtype=bool)
    X_act[silence_idx] = True  # True = silent

    # Source covariance (exp decay)
    gamma = 0.12
    d = cdist(src_xyz, src_xyz)
    Cs_full = np.exp(-gamma * d).astype(np.float32)

    # Sources
    S = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=Cs_full,
        size=t,
    ).T.astype(np.float32)
    S[X_act, :] = 0.0  # silence

    # Noise
    E = rng.normal(0, np.sqrt(noise_pow), size=(n, t)).astype(np.float32)

    # Clean EEG and filtered EEG
    eeg_clean = L @ S
    eeg_lp = butter_lowpass_filter(eeg_clean, fs=Fs, cutoff=90.0, order=4)
    eeg = eeg_lp + E

    # SNR sanity: simulate non-silent
    S_nosil = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=Cs_full,
        size=t,
    ).T.astype(np.float32)
    eeg_nosil = L @ S_nosil
    eeg_nosil = butter_lowpass_filter(eeg_nosil, fs=Fs, cutoff=90.0, order=4)
    snr = float(
        (10.0 * np.log10(np.var(eeg_nosil, axis=1) / (noise_pow + 1e-12))).mean()
    )

    return X_act, eeg, snr, Cs_full
