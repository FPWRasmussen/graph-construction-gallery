#!/usr/bin/env python3
"""
Generate all 74 graph example images for the gallery.

Iterates over every registered GraphBuilder, constructs the graph
on the canonical point layout, renders it, and saves the image to
the appropriate assets/examples/ subdirectory.

Usage:
    # Generate everything
    python scripts/generate_all_examples.py

    # Generate only one category
    python scripts/generate_all_examples.py --category proximity

    # Generate a single algorithm
    python scripts/generate_all_examples.py --algorithm knn

    # Override parameters
    python scripts/generate_all_examples.py --algorithm knn --params k=7

    # Dry run (list what would be generated)
    python scripts/generate_all_examples.py --dry-run

    # Parallel generation
    python scripts/generate_all_examples.py --workers 4
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Ensure all subpackages are imported so builders register themselves
# ---------------------------------------------------------------------------

_SUBPACKAGES = [
    "graphgallery.proximity",
    "graphgallery.triangulation",
    "graphgallery.spanning",
    "graphgallery.random_models",
    "graphgallery.lattice",
    "graphgallery.spanners",
    "graphgallery.ann",
    "graphgallery.kernel",
    "graphgallery.visibility",
    "graphgallery.data_driven",
    "graphgallery.misc",
]


def _import_all_subpackages() -> None:
    """Import every subpackage to trigger builder registration."""
    for pkg in _SUBPACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError as e:
            print(f"  ⚠ Could not import {pkg}: {e}")


# ---------------------------------------------------------------------------
# Data classes for tracking progress
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Result of generating a single graph image."""
    category: str
    slug: str
    name: str
    success: bool
    image_path: str
    n_nodes: int = 0
    n_edges: int = 0
    build_time_ms: float = 0.0
    render_time_ms: float = 0.0
    error_message: str = ""


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

MAX_TITLE_LENGTH = 60
MAX_SUBTITLE_LENGTH = 60


def _shorten_text(text: str, max_length: int) -> str:
    """Trim overly long strings for on-image titles."""

    clean = text.strip()
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 1].rstrip() + "…"


def generate_single_example(
    builder_class,
    layout,
    output_dir: Path,
    dpi: int,
    param_overrides: dict,
) -> GenerationResult:
    """Build one graph and save its image.

    Args:
        builder_class: The GraphBuilder subclass to instantiate.
        layout: The canonical PointLayout.
        output_dir: Root output directory (e.g. assets/examples/).
        dpi: Image resolution.
        param_overrides: Dict of parameter overrides for the builder.

    Returns:
        A GenerationResult with success/failure info.
    """
    from graphgallery.base import CATEGORY_DIRECTORY_MAP
    from graphgallery.viz import plot_graph, save_figure, PlotStyle

    # Instantiate builder with optional parameter overrides
    try:
        builder = builder_class(**param_overrides)
    except TypeError:
        builder = builder_class()

    category = builder.category
    slug = builder.slug
    name = builder.name

    # Determine output path
    cat_dir = CATEGORY_DIRECTORY_MAP.get(category, category)
    image_dir = output_dir / cat_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{slug}.png"

    try:
        # Build graph
        t0 = time.perf_counter()
        result = builder.build_and_record(layout)
        build_ms = result.build_time_ms
        G = result.graph

        # Determine which layout to use for visualization
        # Structured graphs (lattice, debruijn, cayley) have their own positions
        struct_layout = G.graph.get("structural_layout", None)
        viz_layout = struct_layout if struct_layout is not None else layout

        # Handle conforming Delaunay (may have extra Steiner nodes)
        if "all_points" in G.graph:
            from graphgallery.points import PointLayout, ClusterSpec
            import numpy as np
            all_pts = G.graph["all_points"]
            n_orig = G.graph.get("n_original", layout.n_points)
            # Create a new layout encompassing all points
            labels = np.zeros(len(all_pts), dtype=int)
            if n_orig <= len(layout.labels):
                labels[:n_orig] = layout.labels[:n_orig]
            spec = ClusterSpec(
                n_points=len(all_pts),
                center=(float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean())),
                std=0.0,
                label="With Steiner points",
            )
            viz_layout = PointLayout(
                points=all_pts,
                labels=labels,
                cluster_specs=(spec,),
                seed=layout.seed,
            )

        # Render
        t1 = time.perf_counter()

        # Choose whether to show weighted edges
        weighted = category in ("kernel",)

        # Build title/subtitle with consistent lengths
        title = _shorten_text(builder.name, MAX_TITLE_LENGTH)
        subtitle = _shorten_text(builder.description or "", MAX_SUBTITLE_LENGTH)

        # Consistent tile style across all algorithms
        style = PlotStyle(
            figsize=(5.75, 4.25),
            dpi=dpi,
            title_fontsize=13,
            subtitle_fontsize=9,
            subtitle_offset=0.11,
            point_size=52,
            margin=0.75,
        )

        fig = plot_graph(
            G,
            viz_layout,
            title=title,
            subtitle=subtitle,
            style=style,
            weighted=weighted,
        )

        save_figure(fig, str(image_path), dpi=dpi, bbox_inches=None)
        render_ms = (time.perf_counter() - t1) * 1000.0

        return GenerationResult(
            category=category,
            slug=slug,
            name=name,
            success=True,
            image_path=str(image_path),
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            build_time_ms=build_ms,
            render_time_ms=render_ms,
        )

    except Exception as e:
        return GenerationResult(
            category=category,
            slug=slug,
            name=name,
            success=False,
            image_path=str(image_path),
            error_message=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all graph example images for the gallery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/generate_all_examples.py
  python scripts/generate_all_examples.py --category proximity
  python scripts/generate_all_examples.py --algorithm knn --params k=7
  python scripts/generate_all_examples.py --dry-run
  python scripts/generate_all_examples.py --workers 4
""",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Generate only this category (e.g. 'proximity').",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Generate only this algorithm slug (e.g. 'knn').",
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="*",
        default=[],
        help="Parameter overrides as key=value pairs (e.g. k=7 epsilon=1.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for point layout (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/examples",
        help="Output directory for images (default: assets/examples/).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution (default: 150).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be generated without actually generating.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue generating even if some algorithms fail.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output for each algorithm.",
    )

    return parser.parse_args()


def parse_param_overrides(param_strings: list[str]) -> dict:
    """Parse 'key=value' strings into a dict with type inference."""
    overrides = {}
    for s in param_strings:
        if "=" not in s:
            print(f"  ⚠ Ignoring malformed param: {s!r} (expected key=value)")
            continue
        key, value = s.split("=", 1)
        key = key.strip()

        # Type inference
        try:
            overrides[key] = int(value)
        except ValueError:
            try:
                overrides[key] = float(value)
            except ValueError:
                if value.lower() in ("true", "yes"):
                    overrides[key] = True
                elif value.lower() in ("false", "no"):
                    overrides[key] = False
                else:
                    overrides[key] = value

    return overrides


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Graph Construction Gallery — Example Image Generator")
    print("=" * 70)

    # Import all subpackages to populate the registry
    print("\n  Importing subpackages...")
    _import_all_subpackages()

    from graphgallery.base import (
        registry,
        all_builders,
        get_builder,
        list_categories,
        list_algorithms,
        CATEGORY_DIRECTORY_MAP,
        CATEGORY_DISPLAY_NAMES,
    )
    from graphgallery.points import make_two_cluster_layout

    # --- Resolve which builders to run ---
    param_overrides = parse_param_overrides(args.params)

    if args.algorithm:
        # Find the builder across all categories
        found = []
        for cat in list_categories():
            try:
                cls = get_builder(cat, args.algorithm)
                found.append(cls)
            except KeyError:
                continue
        if not found:
            print(f"\n  ✗ Algorithm '{args.algorithm}' not found.")
            print(f"  Available categories: {list_categories()}")
            sys.exit(1)
        builders_to_run = found

    elif args.category:
        if args.category not in registry():
            print(f"\n  ✗ Category '{args.category}' not found.")
            print(f"  Available: {list_categories()}")
            sys.exit(1)
        builders_to_run = [
            get_builder(args.category, slug)
            for slug in list_algorithms(args.category)
        ]

    else:
        builders_to_run = all_builders()

    n_total = len(builders_to_run)
    print(f"\n  Builders to generate: {n_total}")
    print(f"  Output directory:     {args.output_dir}")
    print(f"  DPI:                  {args.dpi}")
    print(f"  Seed:                 {args.seed}")
    if param_overrides:
        print(f"  Param overrides:      {param_overrides}")
    print()

    # --- Dry run ---
    if args.dry_run:
        print("  DRY RUN — would generate:\n")
        for cls in builders_to_run:
            try:
                b = cls(**param_overrides) if param_overrides else cls()
            except TypeError:
                b = cls()
            cat_dir = CATEGORY_DIRECTORY_MAP.get(b.category, b.category)
            path = f"{args.output_dir}/{cat_dir}/{b.slug}.png"
            display_cat = CATEGORY_DISPLAY_NAMES.get(b.category, b.category)
            print(f"    [{display_cat}] {b.name}")
            print(f"      → {path}")
        print(f"\n  Total: {n_total} images")
        return

    # --- Generate layout ---
    print("  Generating canonical point layout...")
    layout = make_two_cluster_layout(seed=args.seed)
    print(f"  Layout: {layout.n_points} points, {layout.n_clusters} clusters\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate examples ---
    results: list[GenerationResult] = []
    t_start = time.perf_counter()

    if args.workers > 1:
        # Parallel execution
        print(f"  Running with {args.workers} parallel workers...\n")
        # Note: ProcessPoolExecutor requires picklable arguments.
        # For simplicity with matplotlib, we fall back to sequential
        # with a thread-based approach.
        # Matplotlib is not fully thread-safe, so we use sequential
        # with progress reporting.
        print("  ⚠ Matplotlib is not fully process-safe; using sequential.\n")
        args.workers = 1

    # Sequential execution with progress bar
    width = 50
    for idx, builder_class in enumerate(builders_to_run):
        # Progress bar
        progress = (idx + 1) / n_total
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        pct = progress * 100

        result = generate_single_example(
            builder_class,
            layout,
            output_dir,
            args.dpi,
            param_overrides,
        )
        results.append(result)

        # Status character
        status = "✓" if result.success else "✗"

        if args.verbose or not result.success:
            print(f"\r  {bar} {pct:5.1f}% ", end="")
            print()
            if result.success:
                print(
                    f"    {status} [{result.category}/{result.slug}] "
                    f"{result.name}: "
                    f"{result.n_nodes}n/{result.n_edges}e "
                    f"({result.build_time_ms:.0f}ms build, "
                    f"{result.render_time_ms:.0f}ms render)"
                )
            else:
                print(
                    f"    {status} [{result.category}/{result.slug}] "
                    f"{result.name}: FAILED"
                )
                print(f"      Error: {result.error_message}")
                if not args.skip_errors:
                    print("\n  Use --skip-errors to continue past failures.")
                    sys.exit(1)
        else:
            # Compact progress line
            print(
                f"\r  {bar} {pct:5.1f}% "
                f"[{result.category}/{result.slug}]"
                f"{'':40}",
                end="",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start

    # Final newline after progress bar
    print()

    # --- Summary ---
    n_success = sum(1 for r in results if r.success)
    n_failed = sum(1 for r in results if not r.success)
    total_build_ms = sum(r.build_time_ms for r in results)
    total_render_ms = sum(r.render_time_ms for r in results)
    total_edges = sum(r.n_edges for r in results if r.success)

    print()
    print("=" * 70)
    print("  Generation Summary")
    print("=" * 70)
    print(f"\n  Total algorithms:   {n_total}")
    print(f"  Successful:         {n_success} ✓")
    if n_failed > 0:
        print(f"  Failed:             {n_failed} ✗")
    print(f"  Total time:         {elapsed:.1f}s")
    print(f"    Build time:       {total_build_ms / 1000:.1f}s")
    print(f"    Render time:      {total_render_ms / 1000:.1f}s")
    print(f"  Total edges built:  {total_edges:,}")
    print(f"  Avg time/graph:     {elapsed / max(n_total, 1) * 1000:.0f}ms")

    # Per-category summary
    print(f"\n  Per-category breakdown:")
    categories_seen = {}
    for r in results:
        if r.category not in categories_seen:
            categories_seen[r.category] = {"success": 0, "failed": 0, "time": 0.0}
        if r.success:
            categories_seen[r.category]["success"] += 1
        else:
            categories_seen[r.category]["failed"] += 1
        categories_seen[r.category]["time"] += r.build_time_ms + r.render_time_ms

    for cat, info in sorted(categories_seen.items()):
        display = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        status = f"{info['success']}✓"
        if info["failed"]:
            status += f" {info['failed']}✗"
        print(f"    {display:30s}  {status:10s}  {info['time']:.0f}ms")

    # Report failures
    if n_failed > 0:
        print(f"\n  Failed algorithms:")
        for r in results:
            if not r.success:
                print(f"    ✗ [{r.category}/{r.slug}] {r.name}")
                print(f"      {r.error_message}")

    print(f"\n  Output directory: {output_dir.resolve()}")
    print(f"\n{'=' * 70}")
    print(f"  {'Done! ✓' if n_failed == 0 else f'Done with {n_failed} errors.'}")
    print(f"{'=' * 70}\n")

    # Exit with error code if there were failures
    if n_failed > 0 and not args.skip_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
