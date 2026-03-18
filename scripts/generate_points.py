#!/usr/bin/env python3
"""
Generate and save the canonical point layout for the Graph Construction Gallery.

This script creates the standard 30-point, two-cluster layout used
across all gallery examples and saves both the data (as .npz) and
a visualization (as .png).

Usage:
    python scripts/generate_points.py
    python scripts/generate_points.py --seed 42 --output-dir assets
    python scripts/generate_points.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the canonical point layout for the gallery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/generate_points.py
  python scripts/generate_points.py --seed 123 --show
  python scripts/generate_points.py --uniform --n-points 50
""",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets",
        help="Directory to save outputs (default: assets/).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Image resolution (default: 200).",
    )

    # Layout variants
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--two-cluster",
        action="store_true",
        default=True,
        help="Standard two-cluster layout (default).",
    )
    group.add_argument(
        "--single-cluster",
        action="store_true",
        help="Single Gaussian cluster.",
    )
    group.add_argument(
        "--uniform",
        action="store_true",
        help="Uniform random layout.",
    )

    parser.add_argument(
        "--n-points",
        type=int,
        default=30,
        help="Total number of points (for single/uniform; default: 30).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from graphgallery.points import (
        make_two_cluster_layout,
        make_single_cluster_layout,
        make_uniform_layout,
    )
    from graphgallery.viz import plot_points_only, save_figure, PlotStyle

    # --- Generate layout ---
    print("=" * 60)
    print("  Graph Construction Gallery — Point Layout Generator")
    print("=" * 60)

    if args.uniform:
        layout = make_uniform_layout(n_points=args.n_points, seed=args.seed)
        layout_name = "uniform"
    elif args.single_cluster:
        layout = make_single_cluster_layout(n_points=args.n_points, seed=args.seed)
        layout_name = "single_cluster"
    else:
        layout = make_two_cluster_layout(seed=args.seed)
        layout_name = "two_cluster"

    # --- Print summary ---
    print(f"\n  Layout type:   {layout_name}")
    print(f"  Total points:  {layout.n_points}")
    print(f"  Clusters:      {layout.n_clusters}")
    print(f"  Seed:          {layout.seed}")

    for c_id in range(layout.n_clusters):
        spec = layout.cluster_specs[c_id]
        pts = layout.cluster_points(c_id)
        print(f"\n  Cluster {c_id}: {spec.label}")
        print(f"    Points:  {pts.shape[0]}")
        print(f"    Center:  ({spec.center[0]:.1f}, {spec.center[1]:.1f})")
        print(f"    Std:     {spec.std}")
        print(f"    X range: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]")
        print(f"    Y range: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]")

    bbox = layout.bounding_box()
    print(f"\n  Bounding box: x=[{bbox[0]:.2f}, {bbox[1]:.2f}], "
          f"y=[{bbox[2]:.2f}, {bbox[3]:.2f}]")

    # --- Save data ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "point_layout.npz"
    np.savez(
        npz_path,
        points=layout.points,
        labels=layout.labels,
        seed=layout.seed,
    )
    print(f"\n  Saved data:  {npz_path}")

    # Also save as CSV for easy inspection
    csv_path = output_dir / "point_layout.csv"
    header = "index,x,y,cluster"
    rows = []
    for i in range(layout.n_points):
        rows.append(f"{i},{layout.points[i, 0]:.6f},{layout.points[i, 1]:.6f},{layout.labels[i]}")
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"  Saved CSV:   {csv_path}")

    # --- Save visualization ---
    style = PlotStyle(dpi=args.dpi, figsize=(8, 5.5))

    title = {
        "two_cluster": "Canonical 30-Point Layout (Two Clusters)",
        "single_cluster": f"Single Cluster Layout ({args.n_points} Points)",
        "uniform": f"Uniform Random Layout ({args.n_points} Points)",
    }[layout_name]

    subtitle = f"Seed={layout.seed}"
    for c_id in range(layout.n_clusters):
        spec = layout.cluster_specs[c_id]
        subtitle += f" | {spec.label}: n={spec.n_points}"
        if spec.std > 0:
            subtitle += f", σ={spec.std}"

    fig = plot_points_only(
        layout,
        title=title,
        style=PlotStyle(
            dpi=args.dpi,
            figsize=(8, 5.5),
            subtitle=subtitle,
            point_size=80,
            show_node_labels=True,
            node_label_fontsize=6,
        ),
        show=args.show,
    )

    png_path = output_dir / "point_layout.png"
    save_figure(fig, str(png_path), dpi=args.dpi)
    print(f"  Saved image: {png_path}")

    print(f"\n{'=' * 60}")
    print("  Done! ✓")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
