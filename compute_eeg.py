# compute_eeg.py
# Multi-region silence + EEG simulation.

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist

def make_multiregion_silence_mask_bilateral(src_xyz, K, per_region_k, enforce_bilateral=True):
    """
    Generate a multi-region silent mask X_act with K regions,
    each region ~per_region_k nodes, and (optionally) enforce
    that the overall silent set covers both hemispheres.
    """
    src_xyz = np.asarray(src_xyz)
    p = src_xyz.shape[0]
    X_act = np.zeros(p, dtype=bool)

    # --- split by hemisphere using x-coordinate ---
    x = src_xyz[:, 0]
    eps = 1e-6
    left_idx  = np.where(x < -eps)[0]
    right_idx = np.where(x >  eps)[0]
    mid_idx   = np.where(np.abs(x) <= eps)[0]  # midline (rare)

    # Fallback: if we can't split hemispheres, just do old behavior
    if len(left_idx) == 0 or len(right_idx) == 0:
        for _ in range(K):
            center = np.random.randint(p)
            d = np.linalg.norm(src_xyz - src_xyz[center], axis=1)
            region = np.argsort(d)[:per_region_k]
            X_act[region] = True
        return X_act

    # --- decide how many regions per hemisphere ---
    if enforce_bilateral and K >= 2:
        K_left = K // 2
        K_right = K - K_left
    else:
        # No bilateral enforcement: just sample anywhere
        K_left = 0
        K_right = 0

    # --- left hemisphere regions ---
    for _ in range(K_left):
        c = np.random.choice(left_idx)
        d = np.linalg.norm(src_xyz[left_idx] - src_xyz[c], axis=1)
        region = left_idx[np.argsort(d)[:per_region_k]]
        X_act[region] = True

    # --- right hemisphere regions ---
    for _ in range(K_right):
        c = np.random.choice(right_idx)
        d = np.linalg.norm(src_xyz[right_idx] - src_xyz[c], axis=1)
        region = right_idx[np.argsort(d)[:per_region_k]]
        X_act[region] = True

    # --- if K > (K_left+K_right), place remaining regions anywhere ---
    remaining = K - (K_left + K_right)
    for _ in range(remaining):
        c = np.random.randint(p)
        d = np.linalg.norm(src_xyz - src_xyz[c], axis=1)
        region = np.argsort(d)[:per_region_k]
        X_act[region] = True

    return X_act


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

   

    # ---- NEW: bilateral mask (left + right hemispheres) ----
    X_act = make_multiregion_silence_mask_bilateral(
    src_xyz,
    K=K,
    per_region_k=per_region_k,
    enforce_bilateral=True,)

    """ # Disallow midline band
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
    p = src_xyz.shape[0]"""

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
    #################################
    #Noiseless
    #E = 0
    ###############################
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
    ##############################
    #noiseless
    #snr = 100.0  # define 100 dB for noiseless baseline
    ####################################

    return X_act, eeg, snr, Cs_full
