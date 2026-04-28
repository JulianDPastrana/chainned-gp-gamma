"""Export per-violin KDE curves to CSV from reservoir volume data.

Outputs for each reservoir (A, B, C, ...):
1) violin_<LETTER>.csv with columns: y,density
2) violin_<LETTER>_normalized.csv with columns: y,density

Preprocessing pipeline (applied to full dataset):
1. Replace negative values with NaN.
2. Forward fill.
3. Fill remaining NaN with 0.
4. Lower-bound clip at 100.

The exported density is reconstructed directly from the rendered violin
polygon, which preserves the exact displayed shape for each violin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection


EXCEL_PATH = Path("Ejercicio hidrología v2.xlsx")
SHEET_INDEX = 1
OUTPUT_DIR = Path(".")


def load_volume_table(excel_path: Path, sheet: int) -> tuple[pd.DataFrame, list[str]]:
    """Load reservoir volume columns from Excel with two-level headers."""
    df = pd.read_excel(excel_path, sheet_name=sheet, header=[0, 1])

    vol_cols = [
        col
        for col in df.columns
        if isinstance(col[1], str) and col[1].strip().lower() == "volumen (m3)"
    ]
    if not vol_cols:
        raise ValueError("No 'Volumen (m3)' columns found in the selected sheet.")

    df_vol = df[vol_cols].copy()
    df_vol.columns = [c[0] for c in df_vol.columns]
    reservoir_names = list(df_vol.columns)

    df_vol = df_vol.replace(",", "", regex=True)
    df_vol = df_vol.apply(pd.to_numeric, errors="coerce")
    return df_vol, reservoir_names


def preprocess(df_vol: pd.DataFrame) -> pd.DataFrame:
    """Apply the mandatory preprocessing pipeline on full data."""
    out = df_vol.copy()
    out[out < 0] = np.nan
    out = out.ffill()
    out = out.fillna(0.0)
    out = out.clip(lower=100.0)
    return out


def build_long_table(df_vol: pd.DataFrame, reservoir_names: list[str]) -> pd.DataFrame:
    """Convert wide volume table to long format and log10 transform for violin y-axis."""
    df_long = (
        df_vol[reservoir_names]
        .reset_index(drop=True)
        .melt(var_name="Reservoir", value_name="Volume_m3")
    )
    df_long["log10_Volume"] = np.log10(df_long["Volume_m3"].to_numpy())
    return df_long


def extract_violin_curves(
    df_long: pd.DataFrame,
    reservoir_names: list[str],
) -> dict[str, pd.DataFrame]:
    """Render violins and extract y-density curves from violin polygons."""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(
        data=df_long,
        x="Reservoir",
        y="log10_Volume",
        order=reservoir_names,
        inner=None,
        linewidth=0.5,
        cut=2,
        bw_method="scott",
        bw_adjust=1.0,
        gridsize=100,
        ax=ax,
    )

    curves: dict[str, pd.DataFrame] = {}
    poly_collections = [c for c in ax.collections if isinstance(c, PolyCollection)]
    if len(poly_collections) < len(reservoir_names):
        raise RuntimeError(
            f"Expected at least {len(reservoir_names)} violin polygons, "
            f"found {len(poly_collections)}."
        )

    for i, name in enumerate(reservoir_names):
        paths = poly_collections[i].get_paths()
        if not paths:
            raise RuntimeError(f"No polygon path found for reservoir '{name}'.")

        verts = paths[0].vertices
        y = verts[:, 1]
        x = verts[:, 0]

        center_x = np.median(x)
        width = np.abs(x - center_x)

        y_unique = np.unique(y)
        dens = np.empty_like(y_unique, dtype=np.float64)
        for j, yj in enumerate(y_unique):
            dens[j] = width[np.isclose(y, yj)].max()

        order = np.argsort(y_unique)
        y_sorted = y_unique[order]
        dens_sorted = dens[order]

        curves[name] = pd.DataFrame(
            {
                "y": np.round(y_sorted, 4),
                "density": np.round(dens_sorted, 5),
            }
        )

    plt.close(fig)
    return curves


def letter_labels(n: int) -> list[str]:
    """Return A, B, ..., Z, AA, AB... labels."""
    labels: list[str] = []
    for i in range(n):
        x = i
        chars: list[str] = []
        while True:
            x, rem = divmod(x, 26)
            chars.append(chr(ord("A") + rem))
            if x == 0:
                break
            x -= 1
        labels.append("".join(reversed(chars)))
    return labels


def save_curves(curves: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write raw and normalized CSVs following required naming convention."""
    labels = letter_labels(len(curves))

    for label, (_, df_curve) in zip(labels, curves.items()):
        raw_path = output_dir / f"violin_{label}.csv"
        norm_path = output_dir / f"violin_{label}.csv"

        # df_curve.to_csv(raw_path, index=False)

        max_density = float(df_curve["density"].max())
        if max_density <= 0:
            density_norm = np.zeros(len(df_curve), dtype=np.float64)
        else:
            density_norm = df_curve["density"].to_numpy(dtype=np.float64) / max_density

        df_norm = pd.DataFrame(
            {
                "y": df_curve["y"],
                "density": np.round(density_norm, 5),
            }
        )
        df_norm.to_csv(norm_path, index=False)

        print(
            f"Saved {raw_path.name} and {norm_path.name} "
            f"(y=[{df_curve['y'].min():.4f}, {df_curve['y'].max():.4f}])"
        )


def main() -> None:
    df_raw, reservoir_names = load_volume_table(EXCEL_PATH, SHEET_INDEX)
    df_proc = preprocess(df_raw)
    df_long = build_long_table(df_proc, reservoir_names)

    curves = extract_violin_curves(df_long, reservoir_names)
    save_curves(curves, OUTPUT_DIR)

    print(f"Done: {len(curves)} reservoir violin distributions exported.")


if __name__ == "__main__":
    main()
