"""
Abstract base class and registry for graph construction algorithms.

Every graph builder in the gallery inherits from :class:`GraphBuilder`,
ensuring a uniform interface for construction, visualization, parameter
introspection, and batch image generation.

The module also provides an automatic **registry** so that scripts like
``generate_all_examples.py`` can discover every available algorithm
without hard-coded imports.
"""

from __future__ import annotations

import abc
import inspect
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Optional, Sequence, Type

import networkx as nx
import numpy as np

from graphgallery.points import PointLayout


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Global mapping: category_slug → {algorithm_slug → builder_class}
_REGISTRY: dict[str, dict[str, Type[GraphBuilder]]] = {}


def registry() -> dict[str, dict[str, Type[GraphBuilder]]]:
    """Return a read-only view of the global builder registry.

    Returns:
        Nested dict mapping ``category_slug → algorithm_slug → class``.

    Example:
        >>> from graphgallery.base import registry
        >>> for cat, algos in registry().items():
        ...     print(f"{cat}: {list(algos.keys())}")
    """
    return dict(_REGISTRY)


def get_builder(category: str, algorithm: str) -> Type[GraphBuilder]:
    """Look up a builder class by category and algorithm slug.

    Args:
        category: Category slug, e.g. ``"proximity"``.
        algorithm: Algorithm slug, e.g. ``"knn"``.

    Returns:
        The registered :class:`GraphBuilder` subclass.

    Raises:
        KeyError: If the category or algorithm is not found.
    """
    try:
        return _REGISTRY[category][algorithm]
    except KeyError:
        available = {
            cat: list(algos.keys()) for cat, algos in _REGISTRY.items()
        }
        raise KeyError(
            f"Builder '{category}/{algorithm}' not found.\n"
            f"Available builders: {available}"
        ) from None


def all_builders() -> list[Type[GraphBuilder]]:
    """Return a flat list of every registered builder class.

    Returns:
        List of :class:`GraphBuilder` subclasses, sorted by
        (category, algorithm) slug.
    """
    builders = []
    for cat_slug in sorted(_REGISTRY):
        for algo_slug in sorted(_REGISTRY[cat_slug]):
            builders.append(_REGISTRY[cat_slug][algo_slug])
    return builders


def list_categories() -> list[str]:
    """Return sorted list of registered category slugs."""
    return sorted(_REGISTRY.keys())


def list_algorithms(category: str) -> list[str]:
    """Return sorted list of algorithm slugs in a category.

    Args:
        category: Category slug.

    Returns:
        List of algorithm slug strings.

    Raises:
        KeyError: If the category doesn't exist.
    """
    if category not in _REGISTRY:
        raise KeyError(
            f"Category '{category}' not found. "
            f"Available: {list_categories()}"
        )
    return sorted(_REGISTRY[category].keys())


# ---------------------------------------------------------------------------
# Build result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildResult:
    """Container for the output of a graph builder.

    Bundles the constructed graph with metadata about the build process.

    Attributes:
        graph: The constructed NetworkX graph.
        builder_name: Human-readable name of the algorithm.
        builder_slug: Machine-friendly algorithm identifier.
        category: Category slug.
        params: Dict of parameters used for the build.
        build_time_ms: Wall-clock time for the build in milliseconds.
        layout: The PointLayout that was used (if spatial).
    """

    graph: nx.Graph
    builder_name: str
    builder_slug: str
    category: str
    params: dict[str, Any]
    build_time_ms: float
    layout: PointLayout | None = None

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def is_directed(self) -> bool:
        return self.graph.is_directed()

    @property
    def density(self) -> float:
        """Edge density in [0, 1]."""
        return nx.density(self.graph)

    def summary(self) -> str:
        """One-line summary string."""
        kind = "directed" if self.is_directed else "undirected"
        return (
            f"[{self.category}/{self.builder_slug}] "
            f"{self.builder_name}: "
            f"{self.n_nodes} nodes, {self.n_edges} edges ({kind}), "
            f"density={self.density:.4f}, "
            f"built in {self.build_time_ms:.1f}ms"
        )

    def __repr__(self) -> str:
        return (
            f"BuildResult(name={self.builder_name!r}, "
            f"nodes={self.n_nodes}, edges={self.n_edges}, "
            f"time={self.build_time_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Parameter descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamInfo:
    """Metadata describing a single algorithm parameter.

    Used for automatic documentation, CLI argument generation,
    and README table rendering.

    Attributes:
        name: Parameter name (matches the constructor kwarg).
        description: Human-readable description.
        type_str: Type as a display string (e.g. ``"int"``, ``"float"``).
        default: Default value.
        constraints: Optional description of valid ranges/values.
    """

    name: str
    description: str
    type_str: str = "float"
    default: Any = None
    constraints: str = ""

    def as_markdown_row(self) -> str:
        """Format as a Markdown table row."""
        default_str = f"`{self.default}`" if self.default is not None else "—"
        return (
            f"| `{self.name}` | {self.description} "
            f"| {self.type_str} | {default_str} | {self.constraints} |"
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class GraphBuilder(abc.ABC):
    """Abstract base class for all graph construction algorithms.

    Subclasses **must** implement:
        - :meth:`build` — construct a graph from a point layout.
        - :attr:`name` — human-readable algorithm name.
        - :attr:`slug` — short, URL-safe identifier.
        - :attr:`category` — category slug (e.g. ``"proximity"``).

    Subclasses **should** implement:
        - :attr:`description` — one-sentence description.
        - :meth:`params_info` — list of :class:`ParamInfo` descriptors.
        - :attr:`is_directed` — whether the algorithm produces directed graphs.
        - :attr:`is_spatial` — whether the algorithm requires point coordinates.
        - :attr:`complexity` — big-O complexity string.

    Registration is automatic: any concrete subclass that defines ``slug``
    and ``category`` is added to the global registry upon class creation.

    Example::

        class KNNGraph(GraphBuilder):
            def __init__(self, k: int = 5):
                self.k = k

            @property
            def name(self) -> str:
                return "k-Nearest Neighbors"

            @property
            def slug(self) -> str:
                return "knn"

            @property
            def category(self) -> str:
                return "proximity"

            def build(self, layout: PointLayout) -> nx.Graph:
                ...
    """

    # --- Registration hook ---------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register concrete subclasses in the global registry."""
        super().__init_subclass__(**kwargs)

        # Only register classes that are fully concrete (all ABCs resolved)
        if inspect.isabstract(cls):
            return

        # Attempt to read slug and category from a temporary instance-less
        # check.  We inspect whether the class defines them as plain
        # properties or class-level attributes.
        slug = _try_get_class_attr(cls, "slug")
        category = _try_get_class_attr(cls, "category")

        if slug and category:
            _REGISTRY.setdefault(category, {})[slug] = cls

    # --- Abstract interface ---------------------------------------------------

    @abc.abstractmethod
    def build(self, layout: PointLayout) -> nx.Graph:
        """Construct a graph from a point layout.

        Implementations should create node indices ``0 .. n-1`` matching
        the rows of ``layout.points``.

        Args:
            layout: A :class:`PointLayout` with n points.

        Returns:
            A NetworkX ``Graph`` (or ``DiGraph``).
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name (e.g. ``'k-Nearest Neighbors'``)."""
        ...

    @property
    @abc.abstractmethod
    def slug(self) -> str:
        """Short URL-safe identifier (e.g. ``'knn'``).

        Used for file paths and registry keys. Should be lowercase
        with underscores, no spaces.
        """
        ...

    @property
    @abc.abstractmethod
    def category(self) -> str:
        """Category slug (e.g. ``'proximity'``, ``'spanning'``).

        Must match one of the subpackage directory names.
        """
        ...

    # --- Optional overrides (sensible defaults) ------------------------------

    @property
    def description(self) -> str:
        """One-sentence description of the algorithm."""
        return ""

    @property
    def is_directed(self) -> bool:
        """Whether this algorithm produces a directed graph."""
        return False

    @property
    def is_spatial(self) -> bool:
        """Whether this algorithm requires spatial point coordinates.

        If ``False``, the builder only uses node count (e.g. lattice graphs,
        random models).
        """
        return True

    @property
    def is_deterministic(self) -> bool:
        """Whether the algorithm produces identical output for the same input."""
        return True

    @property
    def complexity(self) -> str:
        """Big-O time complexity string (e.g. ``'O(n² log n)'``)."""
        return ""

    def params_info(self) -> list[ParamInfo]:
        """Return a list of parameter descriptors for this builder.

        Override to provide rich metadata for documentation and CLI tools.
        The default implementation introspects ``__init__`` parameters.

        Returns:
            List of :class:`ParamInfo` objects.
        """
        return _auto_params_info(self)

    @property
    def params(self) -> dict[str, Any]:
        """Current parameter values as a dict.

        Automatically collected from ``__init__`` argument names matched
        against instance attributes.
        """
        return _collect_params(self)

    # --- Convenience methods -------------------------------------------------

    def build_and_record(self, layout: PointLayout) -> BuildResult:
        """Build the graph and wrap it in a :class:`BuildResult` with timing.

        Args:
            layout: Point layout to build from.

        Returns:
            A :class:`BuildResult` containing the graph and metadata.
        """
        t0 = time.perf_counter()
        G = self.build(layout)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return BuildResult(
            graph=G,
            builder_name=self.name,
            builder_slug=self.slug,
            category=self.category,
            params=self.params,
            build_time_ms=elapsed_ms,
            layout=layout,
        )

    def build_and_plot(
        self,
        layout: PointLayout,
        *,
        show: bool = False,
        weighted: bool = False,
        **plot_kwargs: Any,
    ):
        """Build a graph and immediately visualize it.

        This is a convenience shortcut that combines :meth:`build` with
        :func:`graphgallery.viz.plot_graph`.

        Args:
            layout: Point layout.
            show: If True, display the plot interactively.
            weighted: If True, color edges by weight.
            **plot_kwargs: Additional keyword arguments passed to
                :func:`~graphgallery.viz.plot_graph`.

        Returns:
            ``(BuildResult, matplotlib.figure.Figure)`` tuple.
        """
        # Lazy import to avoid circular dependency at module load
        from graphgallery.viz import plot_graph

        result = self.build_and_record(layout)

        title = plot_kwargs.pop("title", self._default_title())
        subtitle = plot_kwargs.pop("subtitle", self._default_subtitle())

        fig = plot_graph(
            result.graph,
            layout,
            title=title,
            subtitle=subtitle,
            show=show,
            weighted=weighted,
            **plot_kwargs,
        )
        return result, fig

    def validate_layout(self, layout: PointLayout) -> None:
        """Check that a layout is valid for this builder.

        Override for algorithm-specific validation (e.g. minimum point
        count). The base implementation checks for basic sanity.

        Args:
            layout: Layout to validate.

        Raises:
            ValueError: If the layout is invalid.
        """
        if layout.n_points < 2:
            raise ValueError(
                f"{self.name} requires at least 2 points, "
                f"got {layout.n_points}."
            )
        if layout.points.ndim != 2 or layout.points.shape[1] < 2:
            raise ValueError(
                f"{self.name} requires points with shape (n, d) where d >= 2, "
                f"got shape {layout.points.shape}."
            )

    # --- Display methods -----------------------------------------------------

    def info(self) -> str:
        """Rich multi-line description of this builder."""
        lines = [
            f"{'─' * 50}",
            f"  {self.name}",
            f"  Category:      {self.category}",
            f"  Slug:          {self.slug}",
        ]
        if self.description:
            wrapped = textwrap.fill(self.description, width=44)
            lines.append(f"  Description:   {wrapped}")
        lines.append(f"  Directed:      {'yes' if self.is_directed else 'no'}")
        lines.append(f"  Spatial:       {'yes' if self.is_spatial else 'no'}")
        lines.append(f"  Deterministic: {'yes' if self.is_deterministic else 'no'}")
        if self.complexity:
            lines.append(f"  Complexity:    {self.complexity}")

        params = self.params
        if params:
            lines.append(f"  Parameters:")
            for k, v in params.items():
                lines.append(f"    {k} = {v!r}")

        lines.append(f"{'─' * 50}")
        return "\n".join(lines)

    def params_table_markdown(self) -> str:
        """Render the parameters as a Markdown table.

        Returns:
            A Markdown-formatted table string, or empty string if no params.
        """
        infos = self.params_info()
        if not infos:
            return ""
        header = "| Parameter | Description | Type | Default | Constraints |\n"
        sep = "|---|---|---|---|---|\n"
        rows = "\n".join(p.as_markdown_row() for p in infos)
        return header + sep + rows

    @property
    def image_filename(self) -> str:
        """Canonical filename for the gallery image (without directory).

        Returns:
            e.g. ``"knn.png"``
        """
        return f"{self.slug}.png"

    @property
    def image_path(self) -> str:
        """Relative path for the gallery image from repo root.

        Returns:
            e.g. ``"assets/examples/01_proximity/knn.png"``
        """
        # Map category slugs to numbered directories
        cat_dir = CATEGORY_DIRECTORY_MAP.get(self.category, self.category)
        return f"assets/examples/{cat_dir}/{self.image_filename}"

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        cls_name = type(self).__name__
        return f"{cls_name}({params_str})" if params_str else f"{cls_name}()"

    def __str__(self) -> str:
        return f"{self.name} [{self.category}/{self.slug}]"

    # --- Private helpers -----------------------------------------------------

    def _default_title(self) -> str:
        """Generate a default plot title with parameter info."""
        params = self.params
        if not params:
            return self.name

        def _is_simple(value: Any) -> bool:
            return isinstance(value, (int, float, bool, str))

        filtered = []
        for key, value in params.items():
            if value is None or not _is_simple(value):
                continue
            value_str = str(value).strip()
            if not value_str or len(value_str) > 20:
                continue
            filtered.append(f"{key}={value_str}")

        if not filtered:
            return self.name

        param_str = ", ".join(filtered)
        return f"{self.name} ({param_str})"

    def _default_subtitle(self) -> str:
        """Generate a default subtitle from the description."""
        return self.description


# ---------------------------------------------------------------------------
# Category → directory mapping
# ---------------------------------------------------------------------------

CATEGORY_DIRECTORY_MAP: dict[str, str] = {
    "proximity":      "01_proximity",
    "triangulation":  "02_triangulation",
    "spanning":       "03_spanning",
    "random_models":  "04_random_models",
    "lattice":        "05_lattice",
    "spanners":       "06_spanners",
    "ann":            "07_ann",
    "kernel":         "08_kernel",
    "visibility":     "09_visibility",
    "data_driven":    "10_data_driven",
    "misc":           "11_misc",
}

CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "proximity":      "Proximity & Distance-Based",
    "triangulation":  "Triangulation-Based",
    "spanning":       "Spanning Tree-Based",
    "random_models":  "Random Graph Models",
    "lattice":        "Lattice & Structured",
    "spanners":       "Geometric Spanners",
    "ann":            "Approximate Nearest Neighbor",
    "kernel":         "Kernel & Similarity-Based",
    "visibility":     "Visibility Graphs",
    "data_driven":    "Data-Driven / Learned",
    "misc":           "Miscellaneous",
}


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _try_get_class_attr(cls: type, attr_name: str) -> str | None:
    """Try to get a string attribute from a class without instantiation.

    Works for plain attributes, cached properties defined on the class,
    and classlevel defaults. Returns None if the attribute is still
    abstract or cannot be retrieved.
    """
    # Check if it's defined directly on the class dict (not inherited ABC stub)
    for klass in cls.__mro__:
        if attr_name in klass.__dict__:
            val = klass.__dict__[attr_name]
            # Skip attributes that would require instantiation (properties)
            # or are abstract placeholders (decorated via abc.abstractmethod).
            if isinstance(val, property):
                return None
            if getattr(val, "__isabstractmethod__", False):
                return None
            if isinstance(val, str):
                return val
            break
    return None


def _collect_params(builder: GraphBuilder) -> dict[str, Any]:
    """Introspect ``__init__`` to collect current parameter values.

    Matches ``__init__`` parameter names (excluding ``self``) against
    instance attributes.

    Returns:
        Ordered dict of param_name → current_value.
    """
    sig = inspect.signature(type(builder).__init__)
    params: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if hasattr(builder, param_name):
            params[param_name] = getattr(builder, param_name)
    return params


def _auto_params_info(builder: GraphBuilder) -> list[ParamInfo]:
    """Auto-generate :class:`ParamInfo` list from ``__init__`` signature.

    This provides a reasonable fallback when a subclass doesn't override
    :meth:`~GraphBuilder.params_info`.
    """
    sig = inspect.signature(type(builder).__init__)
    infos: list[ParamInfo] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        # Infer type string from annotation
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            type_str = "any"
        elif isinstance(ann, type):
            type_str = ann.__name__
        else:
            type_str = str(ann)

        # Default value
        default = (
            param.default
            if param.default is not inspect.Parameter.empty
            else None
        )

        infos.append(
            ParamInfo(
                name=param_name,
                description="",
                type_str=type_str,
                default=default,
            )
        )

    return infos
