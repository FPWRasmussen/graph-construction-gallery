"""
Shared visualization utilities for the Graph Construction Gallery.

Provides consistent, publication-quality matplotlib rendering of graphs
overlaid on point layouts. All gallery images are generated through
these functions to ensure visual uniformity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import numpy as np
import networkx as nx

from graphgallery.points import PointLayout


# ---------------------------------------------------------------------------
# Color palette — consistent across all gallery images
# ---------------------------------------------------------------------------

# Cluster point colors (extend as needed for more clusters)
CLUSTER_COLORS = [
    "#4C72B0",  # Steel blue  — cluster 0 (small)
    "#DD8452",  # Warm orange  — cluster 1 (large)
    "#55A868",  # Sage green   — cluster 2
    "#C44E52",  # Muted red    — cluster 3
    "#8172B3",  # Soft purple  — cluster 4
    "#937860",  # Brown        — cluster 5
]

EDGE_COLOR = "#2D2D2D"           # Dark grey, slightly transparent
EDGE_COLOR_LIGHT = "#9E9E9E"     # Lighter grey for dense graphs
EDGE_COLOR_WEIGHTED = "#3A7CA5"  # Blue-ish for weighted edge coloring
BACKGROUND_COLOR = "#FAFAFA"     # Off-white background
POINT_EDGE_COLOR = "#FFFFFF"     # White border on points


@dataclass
class PlotStyle:
    """Visual configuration for graph plots.

    All fields have sensible defaults that produce clean gallery images.
    Override individual fields to customize appearance.
    """

    # Figure
    figsize: tuple[float, float] = (7, 5)
    dpi: int = 150
    background_color: str = BACKGROUND_COLOR
    title_fontsize: int = 14
    title_fontweight: str = "bold"
    title_color: str = "#2D2D2D"

    # Points / nodes
    point_size: float = 60.0
    point_zorder: int = 5
    point_edge_color: str = POINT_EDGE_COLOR
    point_linewidth: float = 0.8
    cluster_colors: list[str] = field(default_factory=lambda: list(CLUSTER_COLORS))

    # Edges
    edge_color: str = EDGE_COLOR
    edge_alpha: float = 0.45
    edge_linewidth: float = 0.8
    edge_zorder: int = 2
    edge_color_light: str = EDGE_COLOR_LIGHT
    dense_edge_threshold: int = 200  # Switch to lighter edges above this count

    # Weighted edges
    weighted_cmap: str = "viridis"
    weighted_linewidth_range: tuple[float, float] = (0.4, 2.5)
    show_weight_colorbar: bool = True

    # Directed edges (arrows)
    arrow_size: float = 8.0
    arrow_style: str = "->"
    arrow_color: str = EDGE_COLOR

    # Layout
    margin: float = 0.6
    axis_off: bool = True
    tight_layout: bool = True

    # Annotations
    show_node_labels: bool = False
    node_label_fontsize: int = 7
    show_edge_count: bool = True
    edge_count_fontsize: int = 9
    edge_count_color: str = "#888888"

    # Subtitle / description
    subtitle: str = ""
    subtitle_fontsize: int = 10
    subtitle_color: str = "#666666"
    subtitle_offset: float = 0.2  # Additional offset above the axes for subtitles


# Default singleton style
DEFAULT_STYLE = PlotStyle()


# ---------------------------------------------------------------------------
# Core plotting functions
# ---------------------------------------------------------------------------

def plot_graph(
    G: nx.Graph,
    layout: PointLayout,
    *,
    title: str = "",
    subtitle: str = "",
    style: PlotStyle | None = None,
    ax: plt.Axes | None = None,
    show: bool = False,
    weighted: bool = False,
    weight_key: str = "weight",
) -> plt.Figure:
    """Render a NetworkX graph on top of a point layout.

    This is the primary visualization entry point. It handles undirected
    and directed graphs, optionally coloring/sizing edges by weight.

    Args:
        G: A NetworkX Graph or DiGraph. Nodes should be integer indices
            corresponding to rows in ``layout.points``.
        layout: The :class:`PointLayout` providing node positions and
            cluster labels.
        title: Title displayed above the plot.
        subtitle: Smaller descriptive text below the title.
        style: A :class:`PlotStyle` instance. Uses ``DEFAULT_STYLE`` if None.
        ax: An existing matplotlib Axes to draw on. If None, a new figure
            is created.
        show: If True, call ``plt.show()`` before returning.
        weighted: If True, map edge weights to color and linewidth.
        weight_key: Edge attribute name for weights (default ``"weight"``).

    Returns:
        The matplotlib Figure containing the plot.
    """
    style = style or PlotStyle()

    fig, ax = _setup_figure(ax, style)

    # --- Edges ---
    if weighted and nx.is_weighted(G, weight=weight_key):
        _draw_weighted_edges(G, layout, ax, style, weight_key)
    elif G.is_directed():
        _draw_directed_edges(G, layout, ax, style)
    else:
        _draw_edges(G, layout, ax, style)

    # --- Nodes ---
    _draw_nodes(layout, ax, style)

    # --- Annotations ---
    _add_title(ax, title, subtitle or style.subtitle, style)

    if style.show_edge_count:
        _add_edge_count(ax, G, style)

    if style.show_node_labels:
        _add_node_labels(layout, ax, style)

    # --- Layout ---
    _set_bounds(layout, ax, style)

    if style.axis_off:
        ax.set_axis_off()

    if style.tight_layout:
        top = 0.9 if (title or subtitle or style.subtitle) else 0.97
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))

    if show:
        plt.show()

    return fig


def plot_graph_comparison(
    graphs: Sequence[tuple[str, nx.Graph]],
    layout: PointLayout,
    *,
    ncols: int = 3,
    style: PlotStyle | None = None,
    suptitle: str = "",
    show: bool = False,
) -> plt.Figure:
    """Plot multiple graphs side-by-side on the same layout.

    Useful for comparing algorithms directly.

    Args:
        graphs: Sequence of (title, nx.Graph) pairs.
        layout: Shared point layout.
        ncols: Number of columns in the grid.
        style: Plot style (applied to every subplot).
        suptitle: Overall title for the figure.
        show: If True, call ``plt.show()``.

    Returns:
        The matplotlib Figure.
    """
    style = style or PlotStyle()
    n = len(graphs)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(style.figsize[0] * ncols * 0.55, style.figsize[1] * nrows * 0.55),
        dpi=style.dpi,
        facecolor=style.background_color,
    )
    axes = np.asarray(axes).flatten()

    for idx, (title, G) in enumerate(graphs):
        plot_graph(G, layout, title=title, style=style, ax=axes[idx])

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=style.title_fontsize + 2,
            fontweight=style.title_fontweight,
            color=style.title_color,
            y=1.02,
        )

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_points_only(
    layout: PointLayout,
    *,
    title: str = "Point Layout",
    style: PlotStyle | None = None,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render just the point layout without any graph edges.

    Used to generate the base ``point_layout.png`` image.

    Args:
        layout: The point layout to render.
        title: Plot title.
        style: Plot style.
        ax: Optional existing axes.
        show: If True, call ``plt.show()``.

    Returns:
        The matplotlib Figure.
    """
    style = style or PlotStyle()
    fig, ax = _setup_figure(ax, style)

    _draw_nodes(layout, ax, style)
    _add_title(ax, title, style.subtitle, style)
    _set_bounds(layout, ax, style)

    if style.axis_off:
        ax.set_axis_off()
    if style.tight_layout:
        top = 0.9 if (title or style.subtitle) else 0.97
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    path: str,
    *,
    dpi: int | None = None,
    transparent: bool = False,
    bbox_inches: str | None = "tight",
) -> None:
    """Save a figure to disk and close it to free memory.

    Args:
        fig: The matplotlib Figure.
        path: Output file path (e.g., ``"assets/examples/01_proximity/knn.png"``).
        dpi: Resolution override. Uses the figure's DPI if None.
        transparent: If True, save with a transparent background.
    """
    fig.savefig(
        path,
        dpi=dpi or fig.dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        facecolor=fig.get_facecolor() if not transparent else "none",
        edgecolor="none",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _setup_figure(
    ax: plt.Axes | None,
    style: PlotStyle,
) -> tuple[plt.Figure, plt.Axes]:
    """Create or retrieve a figure and axes."""
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(
        figsize=style.figsize,
        dpi=style.dpi,
        facecolor=style.background_color,
    )
    ax.set_facecolor(style.background_color)
    return fig, ax


def _draw_nodes(
    layout: PointLayout,
    ax: plt.Axes,
    style: PlotStyle,
) -> None:
    """Draw scatter points colored by cluster membership."""
    for cluster_id in range(layout.n_clusters):
        mask = layout.labels == cluster_id
        color = style.cluster_colors[cluster_id % len(style.cluster_colors)]
        ax.scatter(
            layout.points[mask, 0],
            layout.points[mask, 1],
            s=style.point_size,
            c=color,
            edgecolors=style.point_edge_color,
            linewidths=style.point_linewidth,
            zorder=style.point_zorder,
            label=layout.cluster_specs[cluster_id].label or f"Cluster {cluster_id}",
        )


def _draw_edges(
    G: nx.Graph,
    layout: PointLayout,
    ax: plt.Axes,
    style: PlotStyle,
) -> None:
    """Draw undirected edges as a LineCollection for efficiency."""
    edges = list(G.edges())
    if not edges:
        return

    n_edges = len(edges)
    is_dense = n_edges > style.dense_edge_threshold
    color = style.edge_color_light if is_dense else style.edge_color
    alpha = style.edge_alpha * 0.6 if is_dense else style.edge_alpha
    lw = style.edge_linewidth * 0.6 if is_dense else style.edge_linewidth

    segments = [
        [layout.points[u], layout.points[v]]
        for u, v in edges
    ]

    lc = mcollections.LineCollection(
        segments,
        colors=color,
        linewidths=lw,
        alpha=alpha,
        zorder=style.edge_zorder,
    )
    ax.add_collection(lc)


def _draw_directed_edges(
    G: nx.DiGraph,
    layout: PointLayout,
    ax: plt.Axes,
    style: PlotStyle,
) -> None:
    """Draw directed edges with arrowheads."""
    for u, v in G.edges():
        ax.annotate(
            "",
            xy=layout.points[v],
            xytext=layout.points[u],
            arrowprops=dict(
                arrowstyle=style.arrow_style,
                color=style.arrow_color,
                alpha=style.edge_alpha,
                lw=style.edge_linewidth,
                shrinkA=3,
                shrinkB=3,
            ),
            zorder=style.edge_zorder,
        )


def _draw_weighted_edges(
    G: nx.Graph,
    layout: PointLayout,
    ax: plt.Axes,
    style: PlotStyle,
    weight_key: str,
) -> None:
    """Draw edges colored and sized by weight using a colormap."""
    edges = list(G.edges(data=weight_key, default=1.0))
    if not edges:
        return

    weights = np.array([w for _, _, w in edges])

    # Normalize weights to [0, 1]
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-12:
        w_norm = (weights - w_min) / (w_max - w_min)
    else:
        w_norm = np.ones_like(weights) * 0.5

    cmap = plt.get_cmap(style.weighted_cmap)
    lw_min, lw_max = style.weighted_linewidth_range

    segments = [
        [layout.points[u], layout.points[v]]
        for u, v, _ in edges
    ]

    colors = cmap(w_norm)
    linewidths = lw_min + w_norm * (lw_max - lw_min)

    lc = mcollections.LineCollection(
        segments,
        colors=colors,
        linewidths=linewidths,
        alpha=style.edge_alpha,
        zorder=style.edge_zorder,
    )
    ax.add_collection(lc)

    if style.show_weight_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=w_min, vmax=w_max),
        )
        sm.set_array([])
        cbar = ax.get_figure().colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Edge weight", fontsize=8)


def _add_title(
    ax: plt.Axes,
    title: str,
    subtitle: str,
    style: PlotStyle,
) -> None:
    """Add title and optional subtitle."""
    if title:
        ax.set_title(
            title,
            fontsize=style.title_fontsize,
            fontweight=style.title_fontweight,
            color=style.title_color,
            pad=34,
        )
    if subtitle:
        # Place subtitle just below the title area
        ax.text(
            0.5,
            1.02 + min(0.2, style.subtitle_offset),
            subtitle,
            transform=ax.transAxes,
            fontsize=style.subtitle_fontsize,
            color=style.subtitle_color,
            ha="center",
            va="bottom",
        )


def _add_edge_count(
    ax: plt.Axes,
    G: nx.Graph,
    style: PlotStyle,
) -> None:
    """Show edge/node count in the bottom-right corner."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    text = f"{n_nodes} nodes · {n_edges} edges"
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=style.edge_count_fontsize,
        color=style.edge_count_color,
        ha="right",
        va="bottom",
        family="monospace",
    )


def _add_node_labels(
    layout: PointLayout,
    ax: plt.Axes,
    style: PlotStyle,
) -> None:
    """Add small numeric labels to each point."""
    for i, (x, y) in enumerate(layout.points):
        ax.text(
            x,
            y + 0.15,
            str(i),
            fontsize=style.node_label_fontsize,
            ha="center",
            va="bottom",
            color=style.title_color,
            zorder=style.point_zorder + 1,
        )


def _set_bounds(
    layout: PointLayout,
    ax: plt.Axes,
    style: PlotStyle,
) -> None:
    """Set axis limits to the layout bounding box."""
    xmin, xmax, ymin, ymax = layout.bounding_box(padding=style.margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
