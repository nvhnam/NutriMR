"""Create a 2D heatmap from raycast hit points on a cylinder surface.

This script reads a CSV with hit points, unwraps the cylinder into (angle, height),
and saves a heatmap image.

Expected columns by default:
- rel_x
- rel_y
- rel_z

Example:
    python participant_data/cylinder_heatmap.py \
        --input participant_data/P04_20260404_212519_eye.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cylinder-unwrapped heatmap from a raycast CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <input_stem>_heatmap.png.",
    )
    parser.add_argument(
        "--x-col", default="rel_x", help="CSV column name for x coordinate."
    )
    parser.add_argument(
        "--y-col",
        default="rel_y",
        help="CSV column name for y coordinate (height if axis=y).",
    )
    parser.add_argument(
        "--z-col", default="rel_z", help="CSV column name for z coordinate."
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="y",
        help="Cylinder axis direction. Default is y.",
    )
    parser.add_argument(
        "--theta-bins", type=int, default=180, help="Number of bins around cylinder angle."
    )
    parser.add_argument(
        "--height-bins", type=int, default=140, help="Number of bins along cylinder axis."
    )
    parser.add_argument(
        "--title", default="Cylinder Raycast Heatmap", help="Plot title."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window in addition to saving PNG.",
    )
    return parser.parse_args()


def load_points(path: Path, x_col: str, y_col: str, z_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = [x_col, y_col, z_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    points = df[[x_col, y_col, z_col]].copy()
    points.columns = ["x", "y", "z"]
    points = points.apply(pd.to_numeric, errors="coerce").dropna()

    if points.empty:
        raise SystemExit("No valid numeric rows found in the selected coordinate columns.")

    return points


def cylinder_coordinates(
    points: pd.DataFrame, axis: str
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = points["x"].to_numpy()
    y = points["y"].to_numpy()
    z = points["z"].to_numpy()

    cx = float(np.mean(x))
    cy = float(np.mean(y))
    cz = float(np.mean(z))

    if axis == "y":
        rx = x - cx
        rz = z - cz
        theta = np.mod(np.arctan2(rz, rx), 2 * np.pi)
        height = y
        center_a, center_b = cx, cz
    elif axis == "x":
        ry = y - cy
        rz = z - cz
        theta = np.mod(np.arctan2(rz, ry), 2 * np.pi)
        height = x
        center_a, center_b = cy, cz
    else:
        rx = x - cx
        ry = y - cy
        theta = np.mod(np.arctan2(ry, rx), 2 * np.pi)
        height = z
        center_a, center_b = cx, cy

    return theta, height, center_a, center_b


def build_heatmap(
    theta: np.ndarray, height: np.ndarray, theta_bins: int, height_bins: int
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    h_min = float(np.min(height))
    h_max = float(np.max(height))
    if h_max <= h_min:
        h_max = h_min + 1e-6

    hist, h_edges, t_edges = np.histogram2d(
        height,
        theta,
        bins=[height_bins, theta_bins],
        range=[[h_min, h_max], [0.0, 2 * np.pi]],
    )

    extent = (0.0, 360.0, float(h_edges[0]), float(h_edges[-1]))
    return hist, extent


def render_and_save(
    heatmap: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
    output: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="hot",
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Angle around cylinder (degrees)")
    ax.set_ylabel("Axis coordinate")
    ax.set_xlim(0, 360)
    fig.colorbar(image, ax=ax, label="Hit count")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved heatmap to: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    points = load_points(args.input, args.x_col, args.y_col, args.z_col)
    theta, height, center_a, center_b = cylinder_coordinates(points, args.axis)
    heatmap, extent = build_heatmap(theta, height, args.theta_bins, args.height_bins)

    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}_heatmap.png")

    axis_plane = {
        "y": "(x,z)",
        "x": "(y,z)",
        "z": "(x,y)",
    }[args.axis]

    title = (
        f"{args.title} | axis={args.axis}, radial plane={axis_plane}, "
        f"center~({center_a:.3f}, {center_b:.3f}), n={len(points)}"
    )

    render_and_save(heatmap, extent, title, output, args.show)


if __name__ == "__main__":
    main()
