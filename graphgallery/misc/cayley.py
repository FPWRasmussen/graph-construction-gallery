"""
Cayley Graph.

A Cayley graph Cay(Γ, S) represents a group Γ with a generating
set S as a graph:

    - Nodes: elements of Γ
    - Edges: g — g·s  for all g ∈ Γ, s ∈ S

If S is closed under inverses (s ∈ S ⟹ s⁻¹ ∈ S), the graph is
undirected.

Cayley graphs are vertex-transitive (every vertex looks the same)
and provide a geometric/topological view of algebraic structure.
Many named graph families are Cayley graphs:

    - Cycle C_n = Cay(ℤ_n, {1, -1})
    - Hypercube Q_d = Cay(ℤ₂^d, {e₁, ..., e_d})
    - Complete graph K_n = Cay(ℤ_n, ℤ_n \ {0})
    - Petersen graph = Cay(ℤ₅ ⋊ ℤ₂, ...)

This builder implements Cayley graphs for several concrete groups:

    - **Cyclic group ℤ_n**: The integers mod n under addition.
    - **Dihedral group D_n**: Symmetries of a regular n-gon.
    - **Symmetric group S_n**: All permutations of n elements.

Reference:
    Cayley, A. (1878). "Desiderata and suggestions: No. 2. The
    theory of groups: graphical representation." American Journal
    of Mathematics, 1(2), 174–176.
"""

from __future__ import annotations

import itertools
from typing import Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.lattice._layout_utils import (
    ring_positions,
    make_structural_layout,
)


class CayleyGraph(GraphBuilder):
    """Graph representation of a group with its generating set.

    Parameters:
        group: Which group to use.
            - ``"cyclic"``: ℤ_n (integers mod n).
            - ``"dihedral"``: D_n (symmetries of n-gon).
            - ``"symmetric"``: S_n (permutations of n elements).
        n: Group parameter (mod value, polygon sides, or perm size).
        generators: Optional explicit generator set.  If None,
            canonical generators are used.
    """

    slug = "cayley"
    category = "misc"

    def __init__(
        self,
        group: Literal["cyclic", "dihedral", "symmetric"] = "cyclic",
        n: int = 15,
        generators: list | None = None,
    ):
        self.group = group
        self.n = n
        self.generators = generators

    @property
    def name(self) -> str:
        return "Cayley Graph"

    @property
    def description(self) -> str:
        if self.group == "cyclic":
            return f"Cay(ℤ_{self.n}, generators). Algebraic structure as a graph."
        elif self.group == "dihedral":
            return f"Cay(D_{self.n}, {{r, s}}). Symmetries of {self.n}-gon."
        elif self.group == "symmetric":
            return f"Cay(S_{self.n}, transpositions). {self.n}! nodes."
        return f"Cayley graph of {self.group} group."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(|Γ| · |S|)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "group", "Group type",
                "str", "cyclic", "cyclic | dihedral | symmetric",
            ),
            ParamInfo("n", "Group parameter", "int", 15, "≥ 2"),
            ParamInfo(
                "generators", "Generator set (None = canonical)",
                "list | None", None,
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        if self.group == "cyclic":
            return self._build_cyclic()
        elif self.group == "dihedral":
            return self._build_dihedral()
        elif self.group == "symmetric":
            return self._build_symmetric()
        else:
            raise ValueError(f"Unknown group: {self.group}")

    # --- Cyclic group ℤ_n --------------------------------------------------

    def _build_cyclic(self) -> nx.Graph:
        """Cayley graph of the cyclic group ℤ_n.

        Default generators: {1, 2} mod n (giving a circulant graph).
        """
        n = self.n

        if self.generators is not None:
            gens = [int(g) % n for g in self.generators]
        else:
            gens = [1, 2] if n > 4 else [1]

        # Close under inverses for undirected graph
        gen_set = set()
        for g in gens:
            gen_set.add(g % n)
            gen_set.add((-g) % n)
        gen_set.discard(0)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for elem in range(n):
            G.nodes[elem]["label"] = str(elem)
            for g in gen_set:
                neighbor = (elem + g) % n
                if not G.has_edge(elem, neighbor):
                    G.add_edge(elem, neighbor, generator=g)

        positions = ring_positions(n, radius=2.5)

        G.graph["algorithm"] = "cayley"
        G.graph["group"] = "cyclic"
        G.graph["n"] = n
        G.graph["generators"] = sorted(gen_set)
        G.graph["positions"] = positions
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

    # --- Dihedral group D_n ------------------------------------------------

    def _build_dihedral(self) -> nx.Graph:
        """Cayley graph of the dihedral group D_n.

        D_n has 2n elements: n rotations and n reflections.
        Generators: r (rotation by 2π/n) and s (reflection).

        Elements represented as (rotation_index, is_reflected):
            - (k, 0): rotation by 2πk/n
            - (k, 1): reflection after rotation by 2πk/n
        """
        n = self.n
        order = 2 * n  # |D_n| = 2n

        # Enumerate elements: (rot, flip) where rot ∈ {0,...,n-1}, flip ∈ {0,1}
        elements = [(r, f) for f in range(2) for r in range(n)]
        elem_to_id = {e: i for i, e in enumerate(elements)}

        # Group operation: (r1, f1) * (r2, f2)
        def multiply(e1, e2):
            r1, f1 = e1
            r2, f2 = e2
            if f1 == 0:
                return ((r1 + r2) % n, f2)
            else:
                return ((r1 - r2) % n, 1 - f2)

        # Generators: r = (1, 0) and s = (0, 1)
        if self.generators is not None:
            gens = self.generators
        else:
            gens = [(1, 0), (0, 1)]

        # Close under inverses
        gen_set = set()
        for g in gens:
            gen_set.add(g)
            # Inverse of (r, 0) is (-r mod n, 0)
            # Inverse of (r, 1) is (r, 1) (reflections are self-inverse)
            r, f = g
            if f == 0:
                gen_set.add(((-r) % n, 0))
            else:
                gen_set.add((r, 1))
        gen_set.discard((0, 0))  # Remove identity

        G = nx.Graph()
        G.add_nodes_from(range(order))

        for elem in elements:
            eid = elem_to_id[elem]
            r, f = elem
            label = f"r{r}" if f == 0 else f"r{r}s"
            G.nodes[eid]["label"] = label
            G.nodes[eid]["rotation"] = r
            G.nodes[eid]["reflected"] = bool(f)

            for g in gen_set:
                product = multiply(elem, g)
                nid = elem_to_id[product]
                if not G.has_edge(eid, nid):
                    G.add_edge(eid, nid)

        # Layout: rotations on outer ring, reflections on inner ring
        outer = ring_positions(n, radius=2.5)
        inner = ring_positions(n, radius=1.2)
        positions = np.vstack([outer, inner])

        G.graph["algorithm"] = "cayley"
        G.graph["group"] = "dihedral"
        G.graph["n"] = n
        G.graph["order"] = order
        G.graph["positions"] = positions
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

    # --- Symmetric group S_n -----------------------------------------------

    def _build_symmetric(self) -> nx.Graph:
        """Cayley graph of the symmetric group S_n.

        S_n has n! elements (all permutations of {0, ..., n-1}).
        Default generators: adjacent transpositions {(0 1), (1 2), ..., (n-2 n-1)}.

        Warning: n! grows fast — n=4 gives 24 nodes, n=5 gives 120.
        """
        n = self.n
        if n > 5:
            import warnings
            warnings.warn(
                f"S_{n} has {np.math.factorial(n)} elements — "
                f"capping at S_5 (120 nodes) for performance.",
                stacklevel=2,
            )
            n = 5

        # Generate all permutations
        all_perms = list(itertools.permutations(range(n)))
        perm_to_id = {p: i for i, p in enumerate(all_perms)}
        order = len(all_perms)

        # Generators: adjacent transpositions
        if self.generators is not None:
            gens = [tuple(g) for g in self.generators]
        else:
            gens = []
            for i in range(n - 1):
                # Transposition (i, i+1) as a permutation
                perm = list(range(n))
                perm[i], perm[i + 1] = perm[i + 1], perm[i]
                gens.append(tuple(perm))

        # Compose permutations: (p ∘ g)(i) = p[g[i]]
        def compose(p, g):
            return tuple(p[g[i]] for i in range(n))

        G = nx.Graph()
        G.add_nodes_from(range(order))

        for perm in all_perms:
            pid = perm_to_id[perm]
            G.nodes[pid]["label"] = "".join(str(c) for c in perm)
            G.nodes[pid]["permutation"] = perm

            for gen in gens:
                product = compose(perm, gen)
                gid = perm_to_id[product]
                if not G.has_edge(pid, gid):
                    G.add_edge(pid, gid)

                # Also apply inverse (transpositions are self-inverse)
                inv_product = compose(perm, gen)  # Same for transpositions
                inv_id = perm_to_id.get(inv_product)
                if inv_id is not None and not G.has_edge(pid, inv_id):
                    G.add_edge(pid, inv_id)

        # Layout: use spring layout for permutation groups
        # (no natural geometric embedding)
        try:
            pos_dict = nx.kamada_kawai_layout(G)
        except Exception:
            pos_dict = nx.spring_layout(G, seed=42)

        positions = np.array([pos_dict[i] for i in range(order)]) * 3.0

        G.graph["algorithm"] = "cayley"
        G.graph["group"] = "symmetric"
        G.graph["n"] = n
        G.graph["order"] = order
        G.graph["positions"] = positions
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G
