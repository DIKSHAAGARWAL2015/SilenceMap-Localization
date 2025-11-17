#!/usr/bin/env python3
"""
SilenceMap without baseline (Python translation of your MATLAB script)

This follows the same stages:

1. Load high-res (1662) leadfield/headmodel
2. Simulate a region of silence (X_act) away from the midline gap
3. Simulate EEG with low-pass filter and additive noise
4. For each candidate reference electrode:
   - Low-res (818) SilenceMap: compute beta on low-res grid, find silence center
   - High-res (1662) iterative Cs + beta refinement, using Cs ~ sigma * gamma^{C_Exp}

NOTE: CSpeC and plot_source_space_signal_vF are left as TODO stubs.
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares


# ----------------- Helpers you already know conceptually ----------------- #

def butter_lowpass_filter(data, fs, cutoff=90.0, order=8):
    """MATLAB designfilt('lowpassiir', 'FilterOrder', 8, 'PassbandFrequency', 90, ...)"""
    b, a = butter(order, cutoff / (fs / 2.0), btype='low', analog=False)
    return filtfilt(b, a, data, axis=-1)


def build_reference_matrix(n, i_ref0):
    """
    Python version of:
        M = eye(n-1);
        M = [M(:,1:i_ref-1), -ones(n-1,1), M(:,i_ref:end)];
    i_ref0 is 0-based.
    """
    M = np.eye(n - 1, dtype=np.float64)
    left = M[:, :i_ref0]
    right = M[:, i_ref0:]
    mid = -np.ones((n - 1, 1), dtype=np.float64)
    return np.concatenate([left, mid, right], axis=1)  # shape (n-1, n)


def plot_source_space_signal_vF_py(X, sulc, cortex1, cortex_plot):
    """
    Placeholder for MATLAB plot_source_space_signal_vF.
    You can replace this with matplotlib 3D scatter, or keep it a no-op.
    """
    # TODO: implement actual 3D cortex plotting if you want visualization
    pass


import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist


def _get_vertices_array(cortex1):
    """
    cortex1 can be:
      - a dict with key 'vertices'
      - a simple object with attribute .vertices
      - already a numpy array of shape (p,3)
    """
    if isinstance(cortex1, np.ndarray):
        return cortex1
    if isinstance(cortex1, dict) and "vertices" in cortex1:
        return cortex1["vertices"]
    if hasattr(cortex1, "vertices"):
        return cortex1.vertices
    raise ValueError("cortex1 must provide vertices as (p,3) array.")


def CSpeC_py(L1, cortex1, Betta, Cs=None, P_M=None, k=10,
             Epsil=None, pow_flag=0, elec_index=None):
    """
    Python translation of MATLAB CSpeC.m (Convex Spectral Clustering).

    Parameters
    ----------
    L1 : (n, p) np.ndarray
        Leadfield matrix.
    cortex1 : struct-like or np.ndarray
        Cortex with vertices (or directly vertices array of shape (p,3)).
    Betta : (p,) np.ndarray
        Source contribution measure (beta).
    Cs : (p, p) np.ndarray or None
        Source covariance matrix without silence (only used if pow_flag==1).
    P_M : (n, n) or (n,) np.ndarray or None
        Estimated scalp power. If matrix, diag(P_M) is used.
    k : int
        Estimated number of silent sources.
    Epsil : (len(elec_index),) np.ndarray or None
        Upper bounds on power matching error per electrode (used if pow_flag==1).
    pow_flag : int (0 or 1)
        0 → no power constraints (low-res grid).
        1 → power constraints (high-res grid).
    elec_index : list/np.ndarray of ints or None
        Indices of electrodes for power constraints when pow_flag==1.

    Returns
    -------
    x : (p,) np.ndarray
        Solution vector of CSpeC optimization.
    """
    Betta = np.asarray(Betta, dtype=float).reshape(-1)
    L1 = np.asarray(L1, dtype=float)
    p = L1.shape[1]

    # ---- lambda_contig = logspace(0,2,5) * (ones(p,1)'*Betta) ----
    lambda_contig = np.logspace(0, 2, 5) * float(Betta.sum())

    # ---- Build kNN graph on cortex vertices ----
    Loc_graph = _get_vertices_array(cortex1)  # (p,3)
    if Loc_graph.shape[0] != p:
        raise ValueError(f"Vertices count {Loc_graph.shape[0]} != p={p}")

    # A(i,j) = ||loc_i - loc_j||_2
    A = cdist(Loc_graph, Loc_graph, metric="euclidean")  # (p,p)

    W = np.zeros_like(A)
    KK = int(k)  # same as MATLAB: KK = k

    for i in range(p):
        # horizontal neighbors
        inds_H = np.argsort(A[i, :])[:KK+1]  # +1 includes self
        # vertical neighbors
        inds_V = np.argsort(A[:, i])[:KK+1]
        inds = np.unique(np.concatenate([inds_H, inds_V]))
        W[i, inds] = A[i, inds]
        W[inds, i] = A[inds, i]

    # variance of non-zero weights
    W_temp = W[W != 0]
    if W_temp.size == 0:
        raise RuntimeError("Graph construction failed: no non-zero edges.")
    Sigm = np.var(W_temp)

    # Gaussian kernel
    W = np.exp(-(W**2) / (2.0 * Sigm))
    W[W == 1.0] = 0.0  # remove self-edges

    D = np.diag(W.sum(axis=1))
    L = D - W  # graph Laplacian

    one = np.ones(p, dtype=float)

    # ------------------------------------------------------------------
    # Helper to extract P_M(elec, elec)
    # ------------------------------------------------------------------
    def _get_P_M_diag(idx):
        if P_M is None:
            raise ValueError("P_M is required when pow_flag == 1.")
        P_arr = np.asarray(P_M, dtype=float)
        if P_arr.ndim == 1:
            # Already a diag vector
            return float(P_arr[idx])
        elif P_arr.ndim == 2:
            return float(P_arr[idx, idx])
        else:
            raise ValueError("P_M must be 1D (diag) or 2D (matrix).")

    # ------------------------------------------------------------------
    # pow_flag && k > 1 : high-res with power constraints
    # ------------------------------------------------------------------
    if pow_flag and (k > 1):
        if Cs is None or P_M is None or elec_index is None or Epsil is None:
            raise ValueError("Cs, P_M, Epsil, elec_index required when pow_flag==1 and k>1.")

        Cs = np.asarray(Cs, dtype=float)
        elec_index = np.asarray(elec_index, dtype=int).reshape(-1)
        Epsil = np.asarray(Epsil, dtype=float).reshape(-1)
        if len(Epsil) != len(elec_index):
            raise ValueError("Epsil and elec_index must have the same length.")

        Err = np.zeros(lambda_contig.shape[0], dtype=float)
        Err_Cont = np.zeros(lambda_contig.shape[0], dtype=float)
        X_store = []

        for ll, lam in enumerate(lambda_contig):
            x = cp.Variable(p)

            # data term: Betta' * (1 - x)
            data_term = cp.sum(cp.multiply(Betta, (one - x)))
            # contiguity term: (1 - x)' L (1 - x)
            smooth_term = cp.quad_form(one - x, L)

            obj = cp.Minimize(data_term + lam * smooth_term)

            constraints = [
                x <= 1,
                x >= 0,
                cp.norm1(x) <= (p - k)
            ]

            # power constraints for selected electrodes
            for r_idx, r in enumerate(elec_index):
                a_r = L1[r, :]  # (p,)
                A_tilda_sq_r = np.diag(a_r)
                # M_r = A_r * Cs * A_r
                M_r = A_tilda_sq_r @ Cs @ A_tilda_sq_r
                v_r = one @ M_r  # (p,) row vector
                expr = v_r @ x   # scalar

                constraints.append(cp.square(expr - _get_P_M_diag(r)) <= Epsil[r_idx])

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.SCS, verbose=False)  # you can try other solvers like ECOS

            if x.value is None:
                raise RuntimeError("CVXPY failed to find a solution for CSpeC (pow_flag=1,k>1).")

            x_val = np.array(x.value).reshape(-1)
            X_store.append(x_val)

            Err[ll] = float(Betta @ (one - x_val))
            Err_Cont[ll] = float((one - x_val) @ (L @ (one - x_val)))

        # choose best lambda based on normalized squared error sum
        max_E = Err.max()
        max_C = Err_Cont.max()
        if max_E == 0 or max_C == 0:
            dx = (Err**2) + (Err_Cont**2)
        else:
            dx = (Err / max_E)**2 + (Err_Cont / max_C)**2
        idx_best = int(np.argmin(dx))
        x_best = X_store[idx_best]
        return x_best

    # ------------------------------------------------------------------
    # pow_flag && k == 1 : high-res, power constraints, no contiguity term
    # ------------------------------------------------------------------
    if pow_flag and (k == 1):
        if Cs is None or P_M is None or elec_index is None or Epsil is None:
            raise ValueError("Cs, P_M, Epsil, elec_index required when pow_flag==1 and k==1.")

        Cs = np.asarray(Cs, dtype=float)
        elec_index = np.asarray(elec_index, dtype=int).reshape(-1)
        Epsil = np.asarray(Epsil, dtype=float).reshape(-1)
        if len(Epsil) != len(elec_index):
            raise ValueError("Epsil and elec_index must have the same length.")

        x = cp.Variable(p)
        data_term = cp.sum(cp.multiply(Betta, (one - x)))
        obj = cp.Minimize(data_term)

        constraints = [
            x <= 1,
            x >= 0,
            cp.norm1(x) <= (p - k)
        ]

        for r_idx, r in enumerate(elec_index):
            a_r = L1[r, :]
            A_tilda_sq_r = np.diag(a_r)
            M_r = A_tilda_sq_r @ Cs @ A_tilda_sq_r
            v_r = one @ M_r
            expr = v_r @ x
            constraints.append(cp.square(expr - _get_P_M_diag(r)) <= Epsil[r_idx])

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if x.value is None:
            raise RuntimeError("CVXPY failed to find a solution for CSpeC (pow_flag=1,k=1).")

        return np.array(x.value).reshape(-1)

    # ------------------------------------------------------------------
    # ~pow_flag && k > 1 : low-res, no power constraints
    # ------------------------------------------------------------------
    if (not pow_flag) and (k > 1):
        Err = np.zeros(lambda_contig.shape[0], dtype=float)
        Err_Cont = np.zeros(lambda_contig.shape[0], dtype=float)
        X_store = []

        for ll, lam in enumerate(lambda_contig):
            x = cp.Variable(p)
            data_term = cp.sum(cp.multiply(Betta, (one - x)))
            smooth_term = cp.quad_form(one - x, L)
            obj = cp.Minimize(data_term + lam * smooth_term)

            constraints = [
                x <= 1,
                x >= 0,
                cp.norm1(x) <= (p - k)
            ]

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            if x.value is None:
                raise RuntimeError("CVXPY failed to find a solution for CSpeC (pow_flag=0,k>1).")

            x_val = np.array(x.value).reshape(-1)
            X_store.append(x_val)

            Err[ll] = float(Betta @ (one - x_val))
            Err_Cont[ll] = float((one - x_val) @ (L @ (one - x_val)))

        max_E = Err.max()
        max_C = Err_Cont.max()
        if max_E == 0 or max_C == 0:
            dx = (Err**2) + (Err_Cont**2)
        else:
            dx = (Err / max_E)**2 + (Err_Cont / max_C)**2
        idx_best = int(np.argmin(dx))
        x_best = X_store[idx_best]
        return x_best

    # ------------------------------------------------------------------
    # ~pow_flag && k == 1 : low-res, no power constraints, no contiguity term
    # ------------------------------------------------------------------
    if (not pow_flag) and (k == 1):
        x = cp.Variable(p)
        data_term = cp.sum(cp.multiply(Betta, (one - x)))
        obj = cp.Minimize(data_term)

        constraints = [
            x <= 1,
            x >= 0,
            cp.norm1(x) <= (p - k)
        ]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if x.value is None:
            raise RuntimeError("CVXPY failed to find a solution for CSpeC (pow_flag=0,k=1).")

        return np.array(x.value).reshape(-1)

    # Should never reach here
    raise RuntimeError("Unexpected combination of pow_flag and k in CSpeC_py.")



# ----------------- Main function mirroring your MATLAB script ----------------- #

def run_silencemap_no_baseline(
    leadfield_1662_path="/content/sample_data/OT_leadfield_symmetric_1662-128.mat",
    headmodel_1662_path="/content/sample_data/OT_headmodel_symmetric_1662-128.mat",
    leadfield_818_path="/content/sample_data/OT_leadfield_symmetric_818-128.mat",
    headmodel_818_path="/content/sample_data/OT_headmodel_symmetric_818-128.mat",
    Fs=512,
    Noise_pow=0.5e-7,
    k_original=50,
    rng_seed=None,
):
    # ================== 1) Load high-res headmodel (1662) ================== #
    mat_L1662 = loadmat(leadfield_1662_path)
    mat_H1662 = loadmat(headmodel_1662_path)

    # You may need to adjust these keys depending on your .mat structure
    L1 = mat_L1662["L"]                     # (n, 1662)
    Cortex = mat_H1662["Cortex"].item()     # struct
    cortex1 = Cortex["Pial"].item()
    sulc = Cortex["Sulc"].item()
    source_loc = cortex1["vertices"]        # (1662, 3)
    sensor_locs = mat_H1662["sensor_locs"]  # must exist in your .mat

    n, p = L1.shape
    t = 10000

    # midline gap condition: source_loc(:,1) > -Gp & source_loc(:,1) < Gp
    Gp = 5.0
    ind_Long_fiss = np.where(
        (source_loc[:, 0] > -Gp) & (source_loc[:, 0] < Gp)
    )[0]

    # ================== 2) Random region of silence away from midline ================== #
    if rng_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rng_seed)

    GT_dist_to_sensor = 40.0
    silence_indices = ind_Long_fiss.copy()  # will be replaced inside loop

    while (np.mean(GT_dist_to_sensor) > 30.0) or (
        len(np.intersect1d(silence_indices, ind_Long_fiss)) > 0
    ):
        # random center index
        indx_center = rng.integers(0, source_loc.shape[0])
        print("indx_center =", indx_center)

        diff = source_loc - source_loc[indx_center]
        dist_vec = np.sum(diff**2, axis=1)
        I_dist = np.argsort(dist_vec)

        silence_indices = I_dist[:k_original]

        X_act = np.zeros((p,), dtype=np.float64)
        X_act[silence_indices] = 1.0

        # distance of silent region to sensor space
        GT_loc = cortex1["vertices"][X_act == 1.0, :]  # (k_original, 3)
        GT_dist_to_sensor = []
        for j in range(GT_loc.shape[0]):
            d = np.sqrt(np.sum((GT_loc[j] - sensor_locs) ** 2, axis=1))
            GT_dist_to_sensor.append(d.min())
        GT_dist_to_sensor = np.array(GT_dist_to_sensor)

    # Plot actual region of silence (placeholder)
    plot_source_space_signal_vF_py(X_act, sulc, cortex1, cortex1)

    # ================== 3) Source + noise covariance and EEG simulation ================== #
    gamma = 0.12  # exp decay coeff
    C_Exp = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i, p):
            C_Exp[i, j] = gamma * (-np.linalg.norm(source_loc[i] - source_loc[j]))
    C_Exp = C_Exp + C_Exp.T
    Sigma_S = np.exp(C_Exp)  # Cs^{full}

    # noise cov
    Sigma_n = np.diag(Noise_pow * rng.random(n))
    mu = np.zeros(p)
    mu_n = np.zeros(n)

    # S ~ N(mu, Sigma_S)
    S = rng.multivariate_normal(mean=mu, cov=Sigma_S, size=t).T  # (p, t)
    # E ~ N(mu_n, Sigma_n)
    E = rng.multivariate_normal(mean=mu_n, cov=Sigma_n, size=t).T  # (n, t)

    # apply silence
    S_silence = S.copy()
    S_silence[silence_indices, :] = 0.0
    eeg = L1 @ S_silence

    # low-pass filter to 90 Hz
    eeg = butter_lowpass_filter(eeg, fs=Fs, cutoff=90.0, order=8)
    eeg = eeg + E

    # SNR calculation
    eeg_wo_silence = L1 @ S
    eeg_wo_silence = butter_lowpass_filter(eeg_wo_silence, fs=Fs, cutoff=90.0, order=8)
    var_wo = np.var(eeg_wo_silence, axis=1)
    SNR_M = np.mean(10 * np.log10(var_wo / Noise_pow))
    print("Average SNR_M (dB) =", SNR_M)

    # ================== 4) SilenceMap algorithm (low-res + high-res) ================== #

    # ref lookup table (MATLAB 1-based → Python 0-based)
    i_ref_tot = np.array([40, 50, 56, 63, 64, 65, 68, 73, 84, 95]) - 1
    Ref_names = ['CPz', 'Pz', 'POz', 'Oz', 'Iz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']

    # loop over references
    for ref_ind in range(10):
        print("\n==== Reference {} ({}) ====".format(
            i_ref_tot[ref_ind] + 1, Ref_names[ref_ind])
        )

        # ---------- Low-res (818) SilenceMap ---------- #
        mat_L818 = loadmat(leadfield_818_path)
        mat_H818 = loadmat(headmodel_818_path)

        L1_low = mat_L818["L"]  # (n, p_low)
        Cortex_low = mat_H818["Cortex"].item()
        cortex1_low = Cortex_low["Pial"].item()
        sulc_low = Cortex_low["Sulc"].item()
        source_loc_low = cortex1_low["vertices"]

        n_low, p_low = L1_low.shape

        # referencing
        eeg_d = eeg.astype(np.float64)
        M = build_reference_matrix(n_low, int(i_ref_tot[ref_ind]))
        Y = M @ eeg_d
        L1_ref = M @ L1_low

        # PSD-based noise estimation
        w_length = min(int(np.floor(0.5 * Y.shape[1])), 256)
        w_over = int(np.floor(0.5 * w_length))

        f, pxx = welch(
            Y, fs=Fs, nperseg=w_length, noverlap=w_over, nfft=256, axis=1
        )  # shape (n_low-1, F)

        # select 90–100 Hz band
        band = (f >= 90.0) & (f <= 100.0)
        if not np.any(band):
            band = f >= f.max() * 0.8
        eta = pxx[:, band].mean(axis=1)  # (n_low-1,)
        sigma_z_sqrd = eta * (100 - 0.1)
        Cz = np.diag(sigma_z_sqrd)

        # variance reduction: Var_norm_fact
        LL = L1_ref.T @ L1_ref
        LL_sq = LL**2
        Var_norm_fact = np.sum(LL_sq, axis=1)  # (p_low,)

        M_Silence = Y

        # Var(mu)
        Mu_tilda = L1_ref.T @ M_Silence  # (p_low, t)
        f_mu, pxx_mu = welch(
            Mu_tilda, fs=Fs, nperseg=w_length, noverlap=w_over, nfft=256, axis=1
        )
        df = f_mu[1] - f_mu[0] if len(f_mu) > 1 else 1.0
        sigma_mu_sqrd = pxx_mu.sum(axis=1) * df - (Mu_tilda.mean(axis=1) ** 2)
        sigma_mu_sqrd = sigma_mu_sqrd.astype(np.float64)

        # Var(eeg)
        f_eeg, pxx_eeg = welch(
            M_Silence, fs=Fs, nperseg=w_length, noverlap=w_over, nfft=256, axis=1
        )
        df_eeg = f_eeg[1] - f_eeg[0] if len(f_eeg) > 1 else 1.0
        sigma_eeg_sqrd = pxx_eeg.sum(axis=1) * df_eeg - (M_Silence.mean(axis=1) ** 2)
        sigma_eeg_sqrd = sigma_eeg_sqrd.astype(np.float64)

        # P_M and beta (low-res)
        P_M_vec = sigma_eeg_sqrd - sigma_z_sqrd  # (n_low-1,)
        sigma_mu_sqrd_wo_noise_low = sigma_mu_sqrd - np.diag(L1_ref.T @ Cz @ L1_ref)
        Betta_low = sigma_mu_sqrd_wo_noise_low / Var_norm_fact

        # ---------- Find silence center on low-res grid ---------- #
        pow_flag = 0
        k_search_grid = np.floor(np.linspace(2, 200, 20)).astype(int)
        Err = []
        x_tot = []

        P_normalized = P_M_vec / np.max(P_M_vec)

        for k in k_search_grid:
            Sigma_s = np.ones((p_low,), dtype=np.float64)

            # CSpeC on low-res
            x = CSpeC_py(L1_ref, cortex1_low, Betta_low, None, None,
                         k, None, pow_flag, None)
            x_tot.append(x)

            idx_x = np.argsort(x)
            Sigma_s[idx_x[:k]] = 0.0
            P_hat = np.diag(L1_ref @ np.diag(Sigma_s) @ L1_ref.T)
            P_hat = P_hat / np.max(P_hat)
            Err.append(np.sum((P_normalized - P_hat) ** 2))

        Err = np.array(Err)
        best_idx = np.argmin(Err)
        k_best = k_search_grid[best_idx]

        # one more CSpeC call with best k
        x = CSpeC_py(L1_ref, cortex1_low, Betta_low, None, None,
                     int(k_best), None, pow_flag, None)

        X_det_low = np.zeros((p_low,), dtype=np.float64)
        idx_x = np.argsort(x)
        X_det_low[idx_x[:int(k_best)]] = 1.0

        silence_center = source_loc_low[X_det_low == 1.0, :].mean(axis=0)
        plot_source_space_signal_vF_py(X_det_low, sulc_low, cortex1_low, cortex1_low)

        # ---------- High-res (1662) SilenceMap iterations ---------- #
        P_M_diag = P_M_vec  # keep as a vector (instead of diag matrix)

        # reload high-res 1662
        L1_high = L1.copy()
        cortex1_high = cortex1
        sulc_high = sulc
        Src_loc = cortex1_high["vertices"]
        n_high, p_high = L1_high.shape

        # effect of reference: L1 <- M * L1 (same M as above)
        L1_high = M @ L1_high

        # LL, Mu_tilda, sigma_mu_sqrd_wo_noise
        LL_high = L1_high.T @ L1_high
        LL_high_sq = LL_high**2
        Mu_tilda_high = L1_high.T @ M_Silence

        f_mu_h, pxx_mu_h = welch(
            Mu_tilda_high, fs=Fs, nperseg=w_length,
            noverlap=w_over, nfft=256, axis=1
        )
        df_h = f_mu_h[1] - f_mu_h[0] if len(f_mu_h) > 1 else 1.0
        sigma_mu_sqrd_h = pxx_mu_h.sum(axis=1) * df_h - (Mu_tilda_high.mean(axis=1) ** 2)
        sigma_mu_sqrd_h = sigma_mu_sqrd_h.astype(np.float64)
        sigma_mu_sqrd_wo_noise_h = sigma_mu_sqrd_h - np.diag(L1_high.T @ Cz @ L1_high)

        # iterative Cs estimation
        repp = 0
        sigma_sq_hat = 1.0
        gamma_p = 1.0
        sigma_sq_hat_old = 0.0
        silence_center_old = np.array([0.0, 0.0, 0.0])
        silence_center_old_old = np.array([0.0, 0.0, 0.0])

        # construct C_Exp (distance matrix) on high-res
        C_Exp_high = np.zeros((p_high, p_high), dtype=np.float64)
        for i in range(p_high):
            for j in range(i, p_high):
                C_Exp_high[i, j] = -np.linalg.norm(Src_loc[i] - Src_loc[j])
        C_Exp_high = C_Exp_high + C_Exp_high.T

        # initial Cs from low-res center
        Cs = np.exp(C_Exp_high)
        dist_silence = np.sum((Src_loc - silence_center) ** 2, axis=1)
        i_silence = np.argmin(dist_silence)

        while (
            (np.sum((silence_center_old - silence_center) ** 2) > 100.0)
            or (np.sum((silence_center_old_old - silence_center_old) ** 2) > 100.0)
        ) and (repp < 100):

            repp += 1
            P_e = P_M_diag  # vector

            Cs_silent = Cs.copy()
            Cs_silent[i_silence, :] = 0.0
            Cs_silent[:, i_silence] = 0.0

            num = np.diag(L1_high @ Cs_silent @ L1_high.T)
            den = np.diag(L1_high @ Cs @ L1_high.T) + 1e-12
            elec_rank = num / den
            i_rank = np.argsort(elec_rank)[::-1]  # descending

            # LSQ for gamma and sigma_s (p in MATLAB is 1 here)
            kk = 90
            stp = 1

            # only first kk electrodes as in MATLAB
            elec_sub = i_rank[0:kk:stp]

            def residuals(x):
                sigma_sq = x[0]
                gamma_p_local = x[1]
                Cs_model = sigma_sq * (gamma_p_local ** C_Exp_high)

                L_sub = L1_high[elec_sub, :]
                P_hat_sub = np.diag(L_sub @ Cs_model @ L_sub.T)
                P_M_sub = P_e[elec_sub]
                return P_hat_sub - P_M_sub

            x0 = np.array([1.0, np.exp(1.0)])
            lb = [0.0, 0.1]
            ub = [np.inf, np.inf]

            res = least_squares(
                residuals,
                x0,
                bounds=(lb, ub),
                max_nfev=int(1e6),
                ftol=1e-30,
                xtol=1e-15,
                verbose=0,
            )

            xx = res.x
            sigma_sq_hat_old = sigma_sq_hat
            sigma_sq_hat = float(xx[0])
            gamma_p = float(xx[1])
            Cs = sigma_sq_hat * (gamma_p ** C_Exp_high)

            # Betta on high-res
            G = L1_high.T @ L1_high
            M_norm = G @ Cs @ G
            denom_beta = np.diag(M_norm) + 1e-12
            Betta_high = sigma_mu_sqrd_wo_noise_h / denom_beta

            # high-res CSpeC search over k
            k_search_grid_h = np.floor(np.linspace(2, 100, 20)).astype(int)
            Err_h = []
            P_normalized_h = P_e / np.max(P_e)
            Epsil = res.fun**2  # squared residuals (like res.^2)
            pow_flag_h = 1

            for k_h in k_search_grid_h:
                x_h = CSpeC_py(
                    L1_high, cortex1_high, Betta_high, Cs,
                    P_e, int(k_h), Epsil, pow_flag_h, elec_sub
                )
                idx_x_h = np.argsort(x_h)
                Cs_silence_h = Cs.copy()
                Cs_silence_h[idx_x_h[:int(k_h)], :] = 0.0
                Cs_silence_h[:, idx_x_h[:int(k_h)]] = 0.0
                P_hat_h = np.diag(L1_high @ Cs_silence_h @ L1_high.T)
                P_hat_h = P_hat_h / np.max(P_hat_h)
                Err_h.append(np.sum((P_normalized_h - P_hat_h) ** 2))

            Err_h = np.array(Err_h)
            best_idx_h = np.argmin(Err_h)
            k_best_h = k_search_grid_h[best_idx_h]

            x_h = CSpeC_py(
                L1_high, cortex1_high, Betta_high, Cs,
                P_e, int(k_best_h), Epsil, pow_flag_h, elec_sub
            )

            X_det_high = np.zeros((p_high,), dtype=np.float64)
            idx_x_h = np.argsort(x_h)
            X_det_high[idx_x_h[:int(k_best_h)]] = 1.0

            plot_source_space_signal_vF_py(
                X_det_high, sulc_high, cortex1_high, cortex1_high
            )
            i_silence = np.where(X_det_high == 1.0)[0]

            silence_center_old_old = silence_center_old.copy()
            silence_center_old = silence_center.copy()
            silence_center = Src_loc[X_det_high == 1.0, :].mean(axis=0)

            d_center = np.sum((silence_center_old - silence_center) ** 2)

            # you can save results as in MATLAB:
            # Result_file = f"Noisy_OT_woB_kk_{kk}_{repp}_indx_center_{indx_center}_i_ref_{i_ref_tot[ref_ind]+1}.npz"
            # np.savez(Result_file, X_det=X_det_high, X_act=X_act, d_center=d_center,
            #          sigma_sq_hat=sigma_sq_hat, gamma_p=gamma_p,
            #          X_det_L=X_det_low, SNR_M=SNR_M, Err=Err_h)

    print("\nDone.")


if __name__ == "__main__":
    run_silencemap_no_baseline()
