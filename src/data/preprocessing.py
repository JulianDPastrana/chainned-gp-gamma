"""
preprocessing.py — Load and preprocess reservoir volume data from Excel.

Pipeline
--------
1. Read `Ejercicio hidrología v2.xlsx`, sheet index 1 (multi-level header).
2. Extract every "Volumen (m3)" column (one per reservoir).
3. Replace non-positive / NaN values with forward-fill then back-fill
   (preserves temporal structure; 0 fill only if still missing after both).
4. Train / test split: first N_TRAIN (= 3554) rows → train; rest → test.
5. Min-max rescale to [0, 1] using ONLY training statistics (causal).
6. Clip to [1e-6, 1] so the Gamma-likelihood constraint y > 0 holds.

Returns
-------
V_train_norm    : torch.Tensor float32  (N_train, D)
V_test_norm     : torch.Tensor float32  (N_test,  D)
scaler          : dict  {"x_min": np.ndarray (1,D), "x_max": np.ndarray (1,D)}
reservoir_names : list[str]
"""

import numpy as np
import pandas as pd
import torch

# First N_TRAIN rows are the training set (paper Sec. 3.1 — N = 3554)
N_TRAIN = 3554


def load_and_preprocess(
    excel_path: str,
    sheet: int = 1,
    n_train: int = N_TRAIN,
) -> tuple:
    """
    Load the reservoir Excel file and return normalised train / test tensors.

    Parameters
    ----------
    excel_path : path to `Ejercicio hidrología v2.xlsx`
    sheet      : sheet index (0-based); default 1 = "Datos generales"
    n_train    : number of leading rows allocated to training

    Returns
    -------
    V_train_norm, V_test_norm, scaler, reservoir_names
    """
    # ── 1. Load Excel (two-level column header) ───────────────────────────
    df = pd.read_excel(excel_path, sheet_name=sheet, header=[0, 1])

    # ── 2. Extract "Volumen (m3)" columns ─────────────────────────────────
    vol_cols = [
        col for col in df.columns
        if isinstance(col[1], str) and col[1].strip().lower() == "volumen (m3)"
    ]
    if not vol_cols:
        raise ValueError(
            "No 'Volumen (m3)' columns found. "
            "Check sheet index and header format."
        )

    df_vol = df[vol_cols].copy()
    df_vol.columns = [c[0] for c in df_vol.columns]   # flat reservoir names
    reservoir_names = list(df_vol.columns)

    # ── 3. Clean: coerce to numeric, fill non-positive / NaN ──────────────
    df_vol = df_vol.replace(",", "", regex=True)       # strip thousands sep
    df_vol = df_vol.apply(pd.to_numeric, errors="coerce")

    # Zero / negative volumes mean "reservoir didn't exist yet" or missing
    df_vol[df_vol <= 0] = np.nan

    # Forward-fill (respect temporal order), then back-fill start gaps
    df_vol = df_vol.ffill().bfill()

    # Any column still all-NaN → fill with 0
    df_vol = df_vol.fillna(0.0)

    values = df_vol.values.astype(np.float32)   # (T_total, D)

    # ── 4. Causal train / test split ──────────────────────────────────────
    V_train_np = values[:n_train]    # first n_train rows — training set
    V_test_np  = values[n_train:]    # remaining rows  — test set (held out)

    # ── 5. Min-max normalisation — fitted on training data ONLY ──────────
    x_min = V_train_np.min(axis=0, keepdims=True)   # (1, D)
    x_max = V_train_np.max(axis=0, keepdims=True)   # (1, D)
    denom = (x_max - x_min) + 1e-8

    V_train_norm = (V_train_np - x_min) / denom
    V_test_norm  = (V_test_np  - x_min) / denom

    # ── 6. Clip — Gamma likelihood requires strictly positive targets ─────
    V_train_norm = np.clip(V_train_norm, 1e-6, 1.0)
    V_test_norm  = np.clip(V_test_norm,  1e-6, 1.0)

    scaler = {"x_min": x_min, "x_max": x_max}

    return (
        torch.tensor(V_train_norm, dtype=torch.float32),
        torch.tensor(V_test_norm,  dtype=torch.float32),
        scaler,
        reservoir_names,
    )
