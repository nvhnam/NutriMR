"""Unwrap cylinder hit-point CSV files into a 2D rectangle heatmap.

Each CSV row is interpreted as one 3D hit point: x, y, z.
The cylinder surface is unwrapped by converting (x, y, z) -> (theta, z):
- theta: angle around cylinder, mapped to [0, 360) degrees
- z: axial position along cylinder height

Then a 2D histogram is rendered as a heatmap (theta x z).

Usage:
	python visualize.py
	python visualize.py --output cylinder_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_hit_points(path: Path, x_col: int, y_col: int, z_col: int) -> pd.DataFrame | None:
	"""Load one CSV file and return numeric x,y,z columns.

	Returns None for empty files or files without enough usable rows.
	"""

	try:
		frame = pd.read_csv(path, header=None)
	except pd.errors.EmptyDataError:
		return None

	if frame.empty:
		return None

	max_col = max(x_col, y_col, z_col)
	if frame.shape[1] <= max_col:
		return None

	points = frame.iloc[:, [x_col, y_col, z_col]].copy()
	points.columns = ["x", "y", "z"]
	points = points.apply(pd.to_numeric, errors="coerce").dropna()
	if points.empty:
		return None

	return points


def iter_csv_files(folder: Path) -> Iterable[Path]:
	for csv_file in sorted(folder.glob("*.csv")):
		if csv_file.name != Path(__file__).name:
			yield csv_file


def load_all_points(
	folder: Path,
	x_col: int,
	y_col: int,
	z_col: int,
) -> tuple[pd.DataFrame, list[str]]:
	frames: list[pd.DataFrame] = []
	used_files: list[str] = []

	for csv_file in iter_csv_files(folder):
		points = load_hit_points(csv_file, x_col=x_col, y_col=y_col, z_col=z_col)
		if points is None:
			continue
		frames.append(points)
		used_files.append(csv_file.name)

	if not frames:
		raise SystemExit("No usable CSV hit-point files were found in the current folder.")

	all_points = pd.concat(frames, axis=0, ignore_index=True)
	return all_points, used_files


def estimate_cylinder_parameters(points: pd.DataFrame) -> tuple[float, float, float]:
	"""Estimate cylinder center and radius from XY hit points."""

	cx = float(points["x"].mean())
	cy = float(points["y"].mean())
	radius = float(np.median(np.sqrt((points["x"] - cx) ** 2 + (points["y"] - cy) ** 2)))
	return cx, cy, radius


def unwrap_to_rectangle(
	points: pd.DataFrame,
	cx: float,
	cy: float,
	z_min: float,
	z_max: float,
	theta_bins: int,
	z_bins: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
	"""Convert points to (theta, z) bins and return heatmap and plot extent."""

	dx = points["x"].to_numpy() - cx
	dy = points["y"].to_numpy() - cy
	z = points["z"].to_numpy()

	theta = np.arctan2(dy, dx)
	theta = np.mod(theta, 2 * np.pi)

	hist, z_edges, theta_edges = np.histogram2d(
		z,
		theta,
		bins=[z_bins, theta_bins],
		range=[[z_min, z_max], [0.0, 2 * np.pi]],
	)

	extent = (0.0, 360.0, float(z_edges[0]), float(z_edges[-1]))
	return hist, extent


def plot_heatmap(
	heatmap: np.ndarray,
	extent: tuple[float, float, float, float],
	title: str,
	output: Path | None,
) -> None:
	fig, ax = plt.subplots(figsize=(12, 7))

	image = ax.imshow(
		heatmap,
		origin="lower",
		aspect="auto",
		extent=extent,
		cmap="inferno",
		interpolation="nearest",
	)
	fig.colorbar(image, ax=ax, label="Hit count")

	ax.set_title(title)
	ax.set_xlabel("Unwrapped angle theta (degrees)")
	ax.set_ylabel("Cylinder axis z")
	ax.set_xlim(0, 360)
	plt.tight_layout()

	if output is not None:
		fig.savefig(output, dpi=200, bbox_inches="tight")
		print(f"Saved heatmap to {output}")

	plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Unwrap all cylinder hit-point CSV files into a 2D heatmap."
	)
	parser.add_argument("--x-col", type=int, default=0, help="CSV column index for x.")
	parser.add_argument("--y-col", type=int, default=1, help="CSV column index for y.")
	parser.add_argument("--z-col", type=int, default=2, help="CSV column index for z.")
	parser.add_argument("--theta-bins", type=int, default=180, help="Number of angular bins.")
	parser.add_argument("--z-bins", type=int, default=120, help="Number of z-axis bins.")
	parser.add_argument("--cx", type=float, default=None, help="Optional cylinder center x.")
	parser.add_argument("--cy", type=float, default=None, help="Optional cylinder center y.")
	parser.add_argument("--z-min", type=float, default=None, help="Optional fixed z minimum.")
	parser.add_argument("--z-max", type=float, default=None, help="Optional fixed z maximum.")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional path to save the rendered heatmap image.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	folder = Path(__file__).resolve().parent
	points, used_files = load_all_points(
		folder,
		x_col=args.x_col,
		y_col=args.y_col,
		z_col=args.z_col,
	)

	auto_cx, auto_cy, auto_radius = estimate_cylinder_parameters(points)
	cx = auto_cx if args.cx is None else args.cx
	cy = auto_cy if args.cy is None else args.cy
	z_min = float(points["z"].min()) if args.z_min is None else args.z_min
	z_max = float(points["z"].max()) if args.z_max is None else args.z_max

	if z_max <= z_min:
		raise SystemExit("Invalid z range: z-max must be greater than z-min.")

	heatmap, extent = unwrap_to_rectangle(
		points,
		cx=cx,
		cy=cy,
		z_min=z_min,
		z_max=z_max,
		theta_bins=args.theta_bins,
		z_bins=args.z_bins,
	)

	title = (
		f"Cylinder Surface Heatmap ({len(used_files)} files, "
		f"center=({cx:.3f}, {cy:.3f}), R~{auto_radius:.3f})"
	)

	plot_heatmap(heatmap, extent=extent, title=title, output=args.output)


if __name__ == "__main__":
	main()
