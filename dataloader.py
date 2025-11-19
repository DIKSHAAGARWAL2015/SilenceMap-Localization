# dataloader.py
# Load leadfield/headmodel from .mat or create synthetic.

import numpy as np
import mat73


def _walk_py(obj, prefix=""):
    """Recursively walk Python objects (dicts/lists/arrays) from mat73."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _walk_py(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += _walk_py(v, f"{prefix}[{i}]")
    elif isinstance(obj, np.ndarray):
        found.append((prefix, obj))
    return found


def _pick_leadfield_from_py(items):
    """Pick a large 2D array as leadfield (prefer names containing 'L'/'leadfield')."""
    best = None
    for path, arr in items:
        if arr.ndim == 2 and arr.size >= 32 * 256:
            name = path.lower()
            score = arr.size
            if path.split(".")[-1].lower() in ("l",):
                score += 10_000_000
            if "leadfield" in name:
                score += 5_000_000
            if (best is None) or (score > best[0]):
                best = (score, path, arr)
    return None if best is None else best[2]


def _pick_vertices_from_py(items):
    """Pick a (p,3) or (3,p) array for cortex vertices."""
    best = None
    for path, arr in items:
        if arr.ndim == 2 and 3 in arr.shape:
            name = path.lower()
            score = max(arr.shape)
            if "vertices" in name:
                score += 1_000_000
            if "pial" in name:
                score += 500_000
            if "cortex" in name:
                score += 100_000
            if (best is None) or (score > best[0]):
                best = (score, path, arr)
    return None if best is None else best[2]


def load_leadfield_any(leadfield_path, headmodel_path, leadfield_var=None, vertices_var=None):
    """
    Robust loader for MATLAB v7.3 .mat via mat73 (and <= v7).

    Returns
    -------
    L : (n,p) float32
    src_xyz : (p,3) float32
    """
    # Leadfield
    lead = mat73.loadmat(leadfield_path)
    if leadfield_var:
        L = lead
        for part in leadfield_var.split("."):
            L = L[part]
    else:
        items = _walk_py(lead)
        L = _pick_leadfield_from_py(items)
        if L is None:
            raise RuntimeError("Could not find leadfield matrix L in file (mat73).")

    # Vertices
    head = mat73.loadmat(headmodel_path)
    if vertices_var:
        V = head
        for part in vertices_var.split("."):
            V = V[part]
    else:
        items = _walk_py(head)
        V = _pick_vertices_from_py(items)
        if V is None:
            raise RuntimeError("Could not find cortex vertices (Pial) in file (mat73).")

    L = np.array(L, dtype=np.float32)
    V = np.array(V, dtype=np.float32)

    # Normalize vertices shape
    if V.ndim != 2:
        V = V.reshape(V.shape[0], -1)
    if V.shape[1] == 3:
        src_xyz = V
    elif V.shape[0] == 3:
        src_xyz = V.T
    else:
        raise RuntimeError(f"Vertices shape {V.shape} not (p,3) or (3,p).")

    # Normalize leadfield shape
    if L.ndim != 2:
        L = L.reshape(L.shape[0], -1)

    nL, pL = L.shape
    pV = src_xyz.shape[0]
    if pL != pV:
        if nL == pV:
            L = L.T
        else:
            raise RuntimeError(f"Dimension mismatch: L {L.shape}, vertices {src_xyz.shape}")

    return L.astype(np.float32), src_xyz.astype(np.float32)


