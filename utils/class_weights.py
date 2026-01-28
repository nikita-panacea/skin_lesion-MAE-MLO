# utils/class_weights.py
import numpy as np
import pandas as pd
import torch

def compute_class_weights_from_csv(csv_path, class_names=None, eps=1e-6, device="cpu", max_weight=10.0):
    """
    Returns torch.tensor of class weights length = number of classes.
    - Accepts either 'label' column or one-hot class columns (provide class_names)
    - Uses inverse frequency weighting (1 / freq) capped by max_weight
    - Returns a CPU tensor (caller will .to(device) as needed)
    """
    df = pd.read_csv(csv_path)

    if "label" in df.columns and class_names is None:
        labels = df["label"].astype(int).values
        n_classes = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_classes).astype(float)
    else:
        if class_names is None:
            raise ValueError("class_names must be provided when CSV uses one-hot columns")
        cols = list(class_names)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing class columns in CSV: {missing}")
        counts = df[cols].sum(axis=0).astype(float).values

    counts_safe = counts + eps
    freqs = counts_safe / counts_safe.sum()
    weights = 1.0 / (freqs + eps)           # inverse frequency
    # cap extreme weights, keep relative scale
    weights = np.minimum(weights, max_weight)
    # convert to tensor (cpu) — caller should move to device if needed
    w = torch.tensor(weights.astype(np.float32), dtype=torch.float32, device="cpu")
    if not torch.isfinite(w).all():
        raise RuntimeError("Non-finite class weights computed.")
    return w
