"""
De Bruijn Graph.

A De Bruijn graph B(k, n) is a directed graph representing overlaps
between sequences.  Nodes are all possible strings of length n over
an alphabet of size k.  A directed edge connects string s to string
t if the last (n-1) characters of s equal the first (n-1) characters
of t.

    Nodes:  k^n  (all n-length strings over alphabet {0, ..., k-1})
    Edges:  k^n  (each node has exactly k out-edges and k in-edges)

The graph has remarkable combinatorial properties:
    - Every Eulerian circuit spells out a De Bruijn sequence: a cyclic
      string of length k^n that contains every n-length substring
      exactly once.
    - The graph is Eulerian (every node has equal in/out degree).
    - It is Hamiltonian.
    - Used in DNA sequence assembly (overlap-layout-consensus).
    - Used in pseudo-random number generation.

For the gallery, we visualize the undirected version (ignoring
direction) with ~30 nodes, using parameters like B(2, 5) = 32 nodes
or B(3, 3) = 27 nodes.

Reference:
    de Bruijn, N.G. (1946). "A combinatorial problem." Koninklijke
    Nederlandse Akademie van Wetenschappen, 49, 758–764.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.lattice._layout_utils import (
    ring_positions,
    make_structural_layout,
)


class DeBruijnGraph(GraphBuilder):
    """De Bruijn graph B(k, n): sequence overlaps over a k-letter alphabet.

    Parameters:
        k: Alphabet size.
        n: Sequence length (word length).
        as_undirected: If True, return undirected graph.
    """

    slug = "debruijn"
    category = "misc"

    def __init__(
        self,
        k: int = 2,
        n: int = 5,
        as_undirected: bool = True,
    ):
        self.k = k
        self.n = n
        self.as_undirected = as_undirected

    @property
    def name(self) -> str:
        return f"De Bruijn Graph"

    @property
    def description(self) -> str:
        num_nodes = self.k ** self.n
        return (
            f"B({self.k},{self.n}): {num_nodes} nodes from "
            f"{self.n}-length strings over {self.k}-letter alphabet. "
            f"Edges represent sequence overlaps."
        )

    @property
    def is_directed(self) -> bool:
        return not self.as_undirected

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return f"O(k^n · k) = O(k^(n+1))"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Alphabet size", "int", 2, "k ≥ 2"),
            ParamInfo("n", "Word length", "int", 5, "n ≥ 1"),
            ParamInfo("as_undirected", "Return undirected graph", "bool", True),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        k, n = self.k, self.n
        n_nodes = k ** n

        # Generate all n-length strings over alphabet {0, ..., k-1}
        # Map each string to an integer node id
        node_labels: dict[tuple, int] = {}
        id_to_label: dict[int, tuple] = {}

        for idx, combo in enumerate(itertools.product(range(k), repeat=n)):
            node_labels[combo] = idx
            id_to_label[idx] = combo

        # Create graph
        if self.as_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        G.add_nodes_from(range(n_nodes))

        # Label each node with its string
        for idx, label in id_to_label.items():
            G.nodes[idx]["label"] = "".join(str(c) for c in label)

        # Add edges: s → t if s[1:] == t[:-1]
        for s_tuple, s_id in node_labels.items():
            suffix = s_tuple[1:]  # Last (n-1) characters of s

            for c in range(k):
                t_tuple = suffix + (c,)  # Append each possible character
                t_id = node_labels[t_tuple]

                if self.as_undirected:
                    if s_id != t_id and not G.has_edge(s_id, t_id):
                        G.add_edge(s_id, t_id)
                else:
                    G.add_edge(s_id, t_id)

        # Layout: circular arrangement
        positions = ring_positions(n_nodes, radius=3.0)

        G.graph["algorithm"] = "debruijn"
        G.graph["k"] = k
        G.graph["n"] = n
        G.graph["positions"] = positions
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G
