"""Create heatmaps from HoloLens raycast CSV files.

Supports two data shapes:
- Cylinder/object hits from eye/head tracking: `rel_x`, `rel_y`, `rel_z`
- Label surface hits from label tracking: `local_x`, `local_y`, `local_z`

Examples:
    python participant_data/cylinder_heatmap.py \
        --input participant_data/P04_20260404_212519_eye.csv

    python participant_data/cylinder_heatmap.py \
        --input participant_data/P05_20260407_070224_label.csv

    python participant_data/cylinder_heatmap.py \
        --input participant_data/P05_20260407_070224_label.csv \
        --split-by-label
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CYLINDER_COLS = ("rel_x", "rel_y", "rel_z")
LABEL_COLS = ("local_x", "local_y", "local_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cylinder or label-surface heatmap from a raycast CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <input_stem>_heatmap.png.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "cylinder", "label"],
        default="auto",
        help="Heatmap mode. Default auto-detects from columns/file name.",
    )
    parser.add_argument(
        "--x-col",
        default=None,
        help="CSV column name for x-like coordinate. Auto-selected when omitted.",
    )
    parser.add_argument(
        "--y-col",
        default=None,
        help="CSV column name for y-like coordinate. Auto-selected when omitted.",
    )
    parser.add_argument(
        "--z-col",
        default=None,
        help="CSV column name for z-like coordinate. Auto-selected when omitted.",
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="y",
        help="Cylinder axis direction. Only used in cylinder mode.",
    )
    parser.add_argument(
        "--theta-bins",
        type=int,
        default=180,
        help="Number of bins around cylinder angle, or x bins for label mode.",
    )
    parser.add_argument(
        "--height-bins",
        type=int,
        default=140,
        help="Number of bins along cylinder height, or y bins for label mode.",
    )
    parser.add_argument(
        "--split-by-label",
        action="store_true",
        help="In label mode, save one heatmap per `label_name` in addition to the combined heatmap.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title. When omitted, a descriptive title is generated.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window in addition to saving PNG.",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "label"


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    return pd.read_csv(path)


def infer_mode(frame: pd.DataFrame, input_path: Path, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    cols = set(frame.columns)
    if set(LABEL_COLS).issubset(cols) and (
        "label_name" in cols or input_path.stem.endswith("_label")
    ):
        return "label"
    if set(CYLINDER_COLS).issubset(cols):
        return "cylinder"

    raise SystemExit(
        "Could not auto-detect mode. "
        f"Found columns: {list(frame.columns)}. "
        "Use --mode cylinder or --mode label explicitly."
    )


def resolve_columns(
    frame: pd.DataFrame,
    mode: str,
    x_col: str | None,
    y_col: str | None,
    z_col: str | None,
) -> tuple[str, str, str | None]:
    if x_col and y_col:
        return x_col, y_col, z_col

    if mode == "label":
        if x_col or y_col or z_col:
            return x_col or "local_x", y_col or "local_y", z_col or "local_z"
        return LABEL_COLS

    if x_col or y_col or z_col:
        return x_col or "rel_x", y_col or "rel_y", z_col or "rel_z"
    return CYLINDER_COLS


def load_points(frame: pd.DataFrame, columns: tuple[str, str, str | None]) -> pd.DataFrame:
    selected = [c for c in columns if c is not None]
    missing = [c for c in selected if c not in frame.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. Found: {list(frame.columns)}")

    points = frame[selected].copy()
    points.columns = ["x", "y"] + (["z"] if len(selected) == 3 else [])
    points = points.apply(pd.to_numeric, errors="coerce").dropna()

    if points.empty:
        raise SystemExit("No valid numeric rows found in the selected coordinate columns.")

    return points


def cylinder_coordinates(
    points: pd.DataFrame, axis: str
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Convert rel vectors (obj.pos - hitPoint) to cylinder surface angles."""
    x = points["x"].to_numpy()
    y = points["y"].to_numpy()
    z = points["z"].to_numpy()

    if axis == "y":
        theta = np.mod(-np.arctan2(-z, -x), 2 * np.pi)
        height = -y
        center_a, center_b = float(np.mean(x)), float(np.mean(z))
    elif axis == "x":
        theta = np.mod(-np.arctan2(-z, -y), 2 * np.pi)
        height = -x
        center_a, center_b = float(np.mean(y)), float(np.mean(z))
    else:
        theta = np.mod(-np.arctan2(-y, -x), 2 * np.pi)
        height = -z
        center_a, center_b = float(np.mean(x)), float(np.mean(y))

    return theta, height, center_a, center_b


def build_cylinder_heatmap(
    theta: np.ndarray, height: np.ndarray, theta_bins: int, height_bins: int
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    h_min = -3.0
    h_max = 3.0

    hist, h_edges, _ = np.histogram2d(
        height,
        theta,
        bins=[height_bins, theta_bins],
        range=[[h_min, h_max], [0.0, 2 * np.pi]],
    )

    extent = (0.0, 360.0, float(h_edges[0]), float(h_edges[-1]))
    return hist, extent


def build_label_heatmap(
    points: pd.DataFrame, x_bins: int, y_bins: int
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = points["x"].to_numpy()
    y = points["y"].to_numpy()

    pad_x = max((float(np.max(x)) - float(np.min(x))) * 0.05, 1e-4)
    pad_y = max((float(np.max(y)) - float(np.min(y))) * 0.05, 1e-4)
    x_min, x_max = float(np.min(x) - pad_x), float(np.max(x) + pad_x)
    y_min, y_max = float(np.min(y) - pad_y), float(np.max(y) + pad_y)

    hist, y_edges, _ = np.histogram2d(
        y,
        x,
        bins=[y_bins, x_bins],
        range=[[y_min, y_max], [x_min, x_max]],
    )

    extent = (x_min, x_max, float(y_edges[0]), float(y_edges[-1]))
    return hist, extent


def render_and_save(
    heatmap: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
    x_label: str,
    y_label: str,
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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(image, ax=ax, label="Hit count")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved heatmap to: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def render_cylinder_mode(
    points: pd.DataFrame,
    args: argparse.Namespace,
    output: Path,
    show: bool,
) -> None:
    theta, height, center_a, center_b = cylinder_coordinates(points, args.axis)
    heatmap, extent = build_cylinder_heatmap(
        theta, height, args.theta_bins, args.height_bins
    )

    axis_plane = {
        "y": "(x,z)",
        "x": "(y,z)",
        "z": "(x,y)",
    }[args.axis]

    title = args.title or (
        f"Cylinder Raycast Heatmap | axis={args.axis}, radial plane={axis_plane}, "
        f"center~({center_a:.3f}, {center_b:.3f}), n={len(points)}"
    )

    render_and_save(
        heatmap,
        extent,
        title,
        x_label="Angle around cylinder (degrees)",
        y_label="Axis coordinate",
        output=output,
        show=show,
    )


def render_label_mode(
    frame: pd.DataFrame,
    points: pd.DataFrame,
    args: argparse.Namespace,
    output: Path,
    show: bool,
) -> None:
    heatmap, extent = build_label_heatmap(points, args.theta_bins, args.height_bins)
    label_names = sorted(frame["label_name"].dropna().astype(str).unique()) if "label_name" in frame.columns else []
    title = args.title or (
        f"Label Surface Heatmap | labels={len(label_names) or 1}, n={len(points)}"
    )
    render_and_save(
        heatmap,
        extent,
        title,
        x_label="Label local x",
        y_label="Label local y",
        output=output,
        show=show,
    )

    if not args.split_by_label or "label_name" not in frame.columns:
        return

    x_col, y_col, z_col = resolve_columns(frame, "label", args.x_col, args.y_col, args.z_col)
    selected = [c for c in (x_col, y_col, z_col) if c is not None]
    filtered = frame[selected + ["label_name"]].copy()

    for label_name, group in filtered.groupby("label_name"):
        group_points = group[selected].copy()
        group_points.columns = ["x", "y"] + (["z"] if len(selected) == 3 else [])
        group_points = group_points.apply(pd.to_numeric, errors="coerce").dropna()
        if group_points.empty:
            continue

        group_heatmap, group_extent = build_label_heatmap(
            group_points, args.theta_bins, args.height_bins
        )
        group_output = output.with_name(
            f"{output.stem}_{sanitize_filename(str(label_name))}{output.suffix}"
        )
        group_title = args.title or f"Label Surface Heatmap | {label_name} | n={len(group_points)}"
        render_and_save(
            group_heatmap,
            group_extent,
            group_title,
            x_label="Label local x",
            y_label="Label local y",
            output=group_output,
            show=False,
        )


def main() -> None:
    args = parse_args()
    frame = load_frame(args.input)
    mode = infer_mode(frame, args.input, args.mode)
    columns = resolve_columns(frame, mode, args.x_col, args.y_col, args.z_col)
    points = load_points(frame, columns)

    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}_heatmap.png")

    if mode == "label":
        render_label_mode(frame, points, args, output, args.show)
    else:
        render_cylinder_mode(points, args, output, args.show)


if __name__ == "__main__":
    main()
