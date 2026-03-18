#!/usr/bin/env python3
"""
Auto-generate the gallery section of README.md.

Reads the builder registry, checks for existing images, and produces
the Markdown tables and image references for the README gallery.
Can either print to stdout or patch an existing README.md in place.

Usage:
    # Print gallery Markdown to stdout
    python scripts/generate_readme_gallery.py

    # Write gallery section into README.md between markers
    python scripts/generate_readme_gallery.py --patch README.md

    # Generate only a specific category
    python scripts/generate_readme_gallery.py --category proximity

    # Include build statistics from a previous generation run
    python scripts/generate_readme_gallery.py --with-stats
"""

from __future__ import annotations

import argparse
import importlib
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Markers for patching README.md
GALLERY_START_MARKER = "<!-- GALLERY_START -->"
GALLERY_END_MARKER = "<!-- GALLERY_END -->"


# ---------------------------------------------------------------------------
# Import all subpackages
# ---------------------------------------------------------------------------

_SUBPACKAGES = [
    "graphgallery.proximity",
    "graphgallery.triangulation",
    "graphgallery.spanning",
    "graphgallery.random_models",
    "graphgallery.spanners",
    "graphgallery.ann",
    "graphgallery.kernel",
    "graphgallery.visibility",
    "graphgallery.data_driven",
    "graphgallery.misc",
]


def _import_all() -> None:
    for pkg in _SUBPACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Gallery section metadata
# ---------------------------------------------------------------------------

SECTION_META = {
    "proximity": {
        "number": 1,
        "emoji": "🔵",
        "title": "Proximity & Distance-Based Graphs",
        "description": (
            "Graphs built by connecting points based on spatial "
            "distance or neighbor relationships."
        ),
    },
    "triangulation": {
        "number": 2,
        "emoji": "🔺",
        "title": "Triangulation-Based Graphs",
        "description": (
            "Graphs derived from triangulations and Voronoi structures."
        ),
    },
    "spanning": {
        "number": 3,
        "emoji": "🌲",
        "title": "Spanning Tree-Based Graphs",
        "description": (
            "Minimal or constrained trees connecting all nodes."
        ),
    },
    "random_models": {
        "number": 4,
        "emoji": "🎲",
        "title": "Random Graph Models",
        "description": (
            "Probabilistic models that generate edges according to "
            "various distributions."
        ),
    },
    "spanners": {
        "number": 5,
        "emoji": "🧭",
        "title": "Geometric Spanners",
        "description": (
            "Sparse subgraphs that approximately preserve "
            "shortest-path distances."
        ),
    },
    "ann": {
        "number": 6,
        "emoji": "🔎",
        "title": "Approximate Nearest Neighbor Graphs",
        "description": (
            "Graphs optimized for efficient nearest neighbor search."
        ),
    },
    "kernel": {
        "number": 7,
        "emoji": "🌀",
        "title": "Kernel & Similarity-Based Graphs",
        "description": (
            "Graphs where edge weights come from kernel or "
            "similarity functions."
        ),
    },
    "visibility": {
        "number": 8,
        "emoji": "👁️",
        "title": "Visibility Graphs",
        "description": (
            "Graphs derived from geometric visibility or time series.\n\n"
            "*For time-series visibility, we sort points by x-coordinate "
            "and treat y as the series value.*"
        ),
    },
    "data_driven": {
        "number": 9,
        "emoji": "📊",
        "title": "Data-Driven / Learned Graphs",
        "description": (
            "Graphs inferred from statistical relationships between "
            "features.\n\n*Each point's (x, y) coordinates generate "
            "spatially correlated multivariate observations.*"
        ),
    },
    "misc": {
        "number": 10,
        "emoji": "🧩",
        "title": "Miscellaneous",
        "description": (
            "Other notable graph construction methods."
        ),
    },
}


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _builder_cell_html(
    builder,
    image_exists: bool,
    section_number: int,
    builder_index: int,
    cell_width: float,
) -> str:
    image_path = builder.image_path
    if image_exists:
        image_html = (
            f'<img src="{image_path}" alt="{builder.name}" width="100%"/>'
        )
    else:
        image_html = f"<em>{builder.slug}.png missing</em>"

    title_html = f"<strong>{section_number}.{builder_index} {builder.name}</strong>"
    description = (builder.description or "").replace("\n", " ").strip()

    lines = [
        f'    <td align="center" valign="top" width="{cell_width:.0f}%">',
        f"      {image_html}<br/>",
        f"      {title_html}<br/>",
    ]

    if description:
        lines.append(f"      <sub>{description}</sub>")
    else:
        lines.append("      <sub></sub>")

    lines.append("    </td>")
    return "\n".join(lines)


def generate_section_markdown(
    category: str,
    builders: list,
    assets_root: Path,
    ncols: int = 3,
) -> str:
    """Generate the full Markdown for one gallery section.

    Args:
        category: Category slug.
        builders: List of builder instances in this category.
        assets_root: Path to check for existing images.
        ncols: Number of columns in the image grid.

    Returns:
        Markdown string for the section.
    """
    meta = SECTION_META.get(category, {
        "number": 99,
        "emoji": "📌",
        "title": category.replace("_", " ").title(),
        "description": "",
    })

    lines = []
    lines.append(f"### {meta['number']} · {meta['title']}\n")
    lines.append(f"{meta['description']}\n")

    lines.append("<table>")
    cell_width = 100 / ncols

    for row_start in range(0, len(builders), ncols):
        row_builders = builders[row_start:row_start + ncols]
        while len(row_builders) < ncols:
            row_builders.append(None)

        lines.append("  <tr>")
        for col_idx, builder in enumerate(row_builders):
            if builder is None:
                lines.append("    <td></td>")
                continue

            builder_idx = row_start + col_idx + 1
            image_path_full = assets_root / builder.image_path
            image_exists = image_path_full.exists()

            cell_html = _builder_cell_html(
                builder,
                image_exists,
                meta["number"],
                builder_idx,
                cell_width,
            )
            lines.append(cell_html)

        lines.append("  </tr>")

    lines.append("</table>")
    lines.append("")
    lines.append("---\n")

    return "\n".join(lines)


def generate_full_gallery(
    assets_root: Path,
    categories_filter: str | None = None,
    ncols: int = 3,
) -> str:
    """Generate the complete gallery Markdown.

    Args:
        assets_root: Project root to check for images.
        categories_filter: If set, only generate this category.
        ncols: Columns per row.

    Returns:
        Full gallery Markdown string.
    """
    from graphgallery.base import (
        registry,
        list_categories,
        list_algorithms,
        get_builder,
    )

    sections = []
    sections.append("## Gallery\n")

    for category in sorted(
        list_categories(),
        key=lambda c: SECTION_META.get(c, {}).get("number", 99),
    ):
        if categories_filter and category != categories_filter:
            continue

        algo_slugs = list_algorithms(category)
        if not algo_slugs:
            continue

        # Instantiate each builder with defaults
        builders = []
        for slug in algo_slugs:
            cls = get_builder(category, slug)
            try:
                builder = cls()
            except Exception:
                continue
            builders.append(builder)

        if not builders:
            continue

        section_md = generate_section_markdown(
            category, builders, assets_root, ncols
        )
        sections.append(section_md)

    return "\n".join(sections)


def generate_summary_table() -> str:
    """Generate the algorithm comparison summary table."""
    from graphgallery.base import list_categories, list_algorithms

    lines = []
    lines.append("## 📊 Algorithm Comparison Table\n")
    lines.append("| Category | # Algorithms | Spatial? | Deterministic? |")
    lines.append("|---|---|---|---|")

    total = 0
    for category in sorted(
        list_categories(),
        key=lambda c: SECTION_META.get(c, {}).get("number", 99),
    ):
        meta = SECTION_META.get(category, {})
        display = meta.get("title", category)
        n_algos = len(list_algorithms(category))
        total += n_algos

        # Determine spatial/deterministic from section knowledge
        spatial_map = {
            "proximity": "✅",
            "triangulation": "✅",
            "spanning": "✅",
            "random_models": "Some",
            "lattice": "❌",
            "spanners": "✅",
            "ann": "✅",
            "kernel": "✅",
            "visibility": "✅",
            "data_driven": "❌",
            "misc": "Mixed",
        }
        det_map = {
            "proximity": "✅",
            "triangulation": "✅",
            "spanning": "Mostly ✅",
            "random_models": "❌",
            "lattice": "✅",
            "spanners": "✅",
            "ann": "❌",
            "kernel": "✅",
            "visibility": "✅",
            "data_driven": "✅",
            "misc": "Mixed",
        }

        spatial = spatial_map.get(category, "—")
        det = det_map.get(category, "—")

        lines.append(f"| {display} | {n_algos} | {spatial} | {det} |")

    lines.append(f"| **Total** | **{total}** | | |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# README patching
# ---------------------------------------------------------------------------

def patch_readme(readme_path: Path, gallery_md: str) -> bool:
    """Replace the gallery section in an existing README.md.

    Looks for GALLERY_START and GALLERY_END markers and replaces
    everything between them.

    Args:
        readme_path: Path to the README.md file.
        gallery_md: New gallery Markdown content.

    Returns:
        True if successfully patched, False if markers not found.
    """
    if not readme_path.exists():
        print(f"  ✗ README not found: {readme_path}")
        return False

    content = readme_path.read_text(encoding="utf-8")

    start_idx = content.find(GALLERY_START_MARKER)
    end_idx = content.find(GALLERY_END_MARKER)

    if start_idx == -1 or end_idx == -1:
        print(f"  ✗ Gallery markers not found in {readme_path}")
        print(f"    Add these markers to your README.md:")
        print(f"    {GALLERY_START_MARKER}")
        print(f"    ... gallery content ...")
        print(f"    {GALLERY_END_MARKER}")
        return False

    # Replace content between markers (exclusive of markers themselves)
    new_content = (
        content[:start_idx + len(GALLERY_START_MARKER)]
        + "\n\n"
        + gallery_md
        + "\n"
        + content[end_idx:]
    )

    readme_path.write_text(new_content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the gallery section of README.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
To use --patch, add these markers to your README.md:

    {GALLERY_START_MARKER}
    (gallery content will be inserted here)
    {GALLERY_END_MARKER}

Examples:
  python scripts/generate_readme_gallery.py
  python scripts/generate_readme_gallery.py --patch README.md
  python scripts/generate_readme_gallery.py --category proximity
  python scripts/generate_readme_gallery.py --ncols 4
  python scripts/generate_readme_gallery.py --output gallery.md
""",
    )

    parser.add_argument(
        "--patch",
        type=str,
        default=None,
        metavar="README_PATH",
        help="Patch an existing README.md between gallery markers.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="FILE",
        help="Write gallery Markdown to a file instead of stdout.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Generate only one category.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns per row (default: 3).",
    )
    parser.add_argument(
        "--with-summary",
        action="store_true",
        help="Include the algorithm comparison table.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Report which images are missing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Need numpy for builder instantiation
    import numpy as np

    # Make it available in the module scope for generate_builder_cell
    globals()["np"] = np

    print("=" * 60, file=sys.stderr)
    print("  Graph Construction Gallery — README Generator", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Import all subpackages
    print("\n  Importing subpackages...", file=sys.stderr)
    _import_all()

    from graphgallery.base import (
        registry,
        all_builders,
        list_categories,
        list_algorithms,
        get_builder,
        CATEGORY_DIRECTORY_MAP,
    )

    n_builders = len(all_builders())
    n_categories = len(list_categories())
    print(f"  Found {n_builders} builders in {n_categories} categories.",
          file=sys.stderr)

    # Check images
    assets_root = PROJECT_ROOT

    if args.check_images:
        print(f"\n  Image status:", file=sys.stderr)
        n_found = 0
        n_missing = 0
        for cls in all_builders():
            try:
                builder = cls()
            except Exception:
                continue
            img_path = assets_root / builder.image_path
            if img_path.exists():
                n_found += 1
            else:
                n_missing += 1
                print(f"    ✗ MISSING: {builder.image_path}", file=sys.stderr)
        print(f"\n    Found: {n_found}, Missing: {n_missing}", file=sys.stderr)

    # Generate gallery Markdown
    print(f"\n  Generating gallery Markdown (ncols={args.ncols})...",
          file=sys.stderr)

    gallery_md = generate_full_gallery(
        assets_root=assets_root,
        categories_filter=args.category,
        ncols=args.ncols,
    )

    # Optionally add summary table
    if args.with_summary:
        gallery_md += "\n" + generate_summary_table()

    # Output
    if args.patch:
        readme_path = Path(args.patch)
        print(f"\n  Patching {readme_path}...", file=sys.stderr)
        if patch_readme(readme_path, gallery_md):
            print(f"  ✓ Gallery section updated in {readme_path}",
                  file=sys.stderr)
        else:
            print(f"  ✗ Failed to patch {readme_path}", file=sys.stderr)
            sys.exit(1)

    elif args.output:
        output_path = Path(args.output)
        output_path.write_text(gallery_md, encoding="utf-8")
        print(f"\n  ✓ Written to {output_path}", file=sys.stderr)

    else:
        # Print to stdout
        print(gallery_md)

    # Stats
    lines = gallery_md.count("\n")
    images = gallery_md.count("![")
    print(f"\n  Gallery stats:", file=sys.stderr)
    print(f"    Lines:     {lines}", file=sys.stderr)
    print(f"    Images:    {images}", file=sys.stderr)
    print(f"    Sections:  {n_categories if not args.category else 1}",
          file=sys.stderr)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Done! ✓", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
