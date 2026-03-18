"""
Random Graph Model Construction Algorithms.

This subpackage contains 13 graph builders based on probabilistic
or generative models:

1.  Erdős–Rényi G(n, p)
2.  Erdős–Rényi G(n, m)
3.  Watts–Strogatz Small World
4.  Barabási–Albert Preferential Attachment
5.  Random Geometric Graph
6.  Stochastic Block Model
7.  Configuration Model
8.  Chung–Lu Model
9.  Kronecker Graph
10. Forest Fire Model
11. Price's Model (directed)
12. Holme–Kim Model
13. Random Regular Graph

Note: Random models generate topology probabilistically.  Node
positions come from the point layout for visualization only (except
Random Geometric, which uses spatial coordinates directly).
All builders accept a ``seed`` parameter for reproducibility.
"""

from graphgallery.random_models.erdos_renyi_gnp import ErdosRenyiGnpGraph
from graphgallery.random_models.erdos_renyi_gnm import ErdosRenyiGnmGraph
from graphgallery.random_models.watts_strogatz import WattsStrogatzGraph
from graphgallery.random_models.barabasi_albert import BarabasiAlbertGraph
from graphgallery.random_models.random_geometric import RandomGeometricGraph
from graphgallery.random_models.sbm import StochasticBlockModelGraph
from graphgallery.random_models.configuration import ConfigurationModelGraph
from graphgallery.random_models.chung_lu import ChungLuGraph
from graphgallery.random_models.kronecker import KroneckerGraph
from graphgallery.random_models.forest_fire import ForestFireGraph
from graphgallery.random_models.price import PriceGraph
from graphgallery.random_models.holme_kim import HolmeKimGraph
from graphgallery.random_models.random_regular import RandomRegularGraph

__all__ = [
    "ErdosRenyiGnpGraph",
    "ErdosRenyiGnmGraph",
    "WattsStrogatzGraph",
    "BarabasiAlbertGraph",
    "RandomGeometricGraph",
    "StochasticBlockModelGraph",
    "ConfigurationModelGraph",
    "ChungLuGraph",
    "KroneckerGraph",
    "ForestFireGraph",
    "PriceGraph",
    "HolmeKimGraph",
    "RandomRegularGraph",
]


def all_random_model_builders():
    """Return instances of every random model builder with default params."""
    return [
        ErdosRenyiGnpGraph(),
        ErdosRenyiGnmGraph(),
        WattsStrogatzGraph(),
        BarabasiAlbertGraph(),
        RandomGeometricGraph(),
        StochasticBlockModelGraph(),
        ConfigurationModelGraph(),
        ChungLuGraph(),
        KroneckerGraph(),
        ForestFireGraph(),
        PriceGraph(),
        HolmeKimGraph(),
        RandomRegularGraph(),
    ]
