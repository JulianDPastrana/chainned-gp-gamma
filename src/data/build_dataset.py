"""
build_dataset.py — Sliding-window dataset builder for time-series forecasting.

Notation follows paper Sec. 2.1:
  x_n = [v_{n-T}, ..., v_{n-1}]  ∈ R^{T·D}  (flattened past window)
  y_n = v_{n+H-1}               ∈ R^D       (H steps ahead of last observation)
The function is stateless and can be applied to any temporal slice of V
(train, Optuna-val, or test) to produce the corresponding (X, Y) pairs.
"""

import torch


def build_dataset(V: torch.Tensor, T: int, H: int) -> tuple:
    """
    Build (X, Y) sliding-window pairs from a volume tensor.

    Parameters
    ----------
    V : (time, D) tensor — normalised reservoir volumes
    T : model order — number of past time steps used as input features
    H : forecast horizon — number of steps ahead to predict

    Returns
    -------
    X : (N_samples, T*D) float32 tensor
    Y : (N_samples, D)   float32 tensor

    Notes
    -----
    Indexing: sample n uses V[n-T : n] as input and V[n+H-1] as target.
    "H steps ahead" means H steps from the last observed value V[n-1]:
      H=1 → predict V[n]  (immediate next step)
      H=2 → predict V[n+1], etc.
    Valid range: n ∈ [T, len(V)-H+1), so N_samples = len(V) - T - H + 1.
    """
    if len(V) - T - H < 0:
        raise ValueError(
            f"Tensor too short for T={T}, H={H}: "
            f"need >= {T + H} rows, got {len(V)}."
        )

    X_list, Y_list = [], []
    for n in range(T, len(V) - H + 1):
        X_list.append(V[n - T : n].reshape(-1))   # (T*D,)
        Y_list.append(V[n + H - 1])               # (D,)  — H steps from V[n-1]

    return torch.stack(X_list), torch.stack(Y_list)

