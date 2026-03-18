<p align="center">
  <img src="assets/banner.png" alt="Graph Construction Gallery Banner" width="800"/>
</p>

<h1 align="center">📊 Graph Construction Gallery</h1>

<p align="center">
  <strong>A visual encyclopedia of 74 graph construction algorithms, all demonstrated on the same point layout.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-the-canonical-point-layout">Point Layout</a> •
  <a href="#-gallery">Gallery</a> •
  <a href="#-api-usage">API</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/algorithms-74-green" alt="74 algorithms"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License"/>
</p>

---

## 🤔 What Is This?

This repository is a **comprehensive, visual reference** for graph construction algorithms. Every algorithm builds a graph from the **same set of 30 points** arranged in two clusters (a small cluster of 10 and a large cluster of 20), so you can directly compare how different methods connect the same data.

Whether you're studying **spectral clustering**, building **approximate nearest neighbor** indices, designing **mesh generation** pipelines, or just curious about graph theory — this gallery gives you side-by-side intuition.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/graph-construction-gallery.git
cd graph-construction-gallery

# Install dependencies
pip install -e ".[dev]"

# Generate the canonical point layout
python scripts/generate_points.py

# Generate ALL example images (saves to assets/examples/)
python scripts/generate_all_examples.py
```

### Dependencies

```
numpy >= 1.24
scipy >= 1.11
scikit-learn >= 1.3
networkx >= 3.1
matplotlib >= 3.8
shapely >= 2.0
```

---

## 📍 The Canonical Point Layout

Every graph in this gallery is constructed on **the same 30 points**: two Gaussian clusters with a clear separation.

- **Cluster A (small):** 10 points, centered at (-2.0, 0.0), σ = 0.5
- **Cluster B (large):** 20 points, centered at (2.0, 0.0), σ = 0.8
- **Random seed:** 42 (fully reproducible)

<p align="center">
  <img src="assets/point_layout.png" alt="Canonical 30-point layout" width="500"/>
</p>

> **Note:** Lattice/structured graphs (Section 5) use their own canonical node positions since their topology *defines* the layout. They are shown with ~30 nodes for visual consistency.

> **Note:** Visibility graphs (Section 9) interpret the x-coordinates of the points as a time series, using the y-coordinates as values.

---

<!-- GALLERY_START -->

## 🖼️ Gallery

### 1 · Proximity & Distance-Based Graphs

Graphs built by connecting points based on spatial distance or neighbor relationships.

|:---:|:---:|:---:|
| ![β-Skeleton](assets/examples/01_proximity/beta_skeleton.png) | ![Complete Graph](assets/examples/01_proximity/complete.png) | ![ε-Neighborhood](assets/examples/01_proximity/epsilon.png) |
| **1.1 β-Skeleton** (beta=1.5) | **1.2 Complete Graph** (weighted=True) | **1.3 ε-Neighborhood** (epsilon=1.2) |
| Generalized proximity graph with β=1.5. β=1 → Gabriel, β=2 → RNG. | Every node connected to every other node. O(n²) edges. | Connect all pairs within Euclidean distance ε=1.2. |

|:---:|:---:|:---:|
| ![Gabriel Graph](assets/examples/01_proximity/gabriel.png) | ![Sphere of Influence Graph](assets/examples/01_proximity/influence.png) | ![k-Nearest Neighbors](assets/examples/01_proximity/knn.png) |
| **1.4 Gabriel Graph** | **1.5 Sphere of Influence Graph** | **1.6 k-Nearest Neighbors** (k=5) |
| Connect if no other point lies inside the diametral disk. Equivalent to β-skeleton with β=1. | Each point's radius = nearest-neighbor distance. Connect if spheres overlap. | Directed graph — each node has k outgoing edges to its closest neighbors. |

|:---:|:---:|:---:|
| ![Mutual k-NN](assets/examples/01_proximity/mutual_knn.png) | ![Relative Neighborhood Graph](assets/examples/01_proximity/rng.png) | ![Symmetric k-NN](assets/examples/01_proximity/symmetric_knn.png) |
| **1.7 Mutual k-NN** (k=5) | **1.8 Relative Neighborhood Graph** | **1.9 Symmetric k-NN** (k=5) |
| Undirected edge only if both nodes are in each other's k-NN. | Connect if no third point is closer to both endpoints. Equivalent to β-skeleton with β=2. | Undirected edge if either node is in the other's k-NN (union). |

|:---:|:---:|:---:|
| ![Urquhart Graph](assets/examples/01_proximity/urquhart.png) |  |  |
| **1.10 Urquhart Graph** |  |  |
| Delaunay triangulation minus the longest edge of each triangle. Approximates the RNG. |  |  |

---

### 2 · Triangulation-Based Graphs

Graphs derived from triangulations and Voronoi structures.

|:---:|:---:|:---:|
| ![Conforming Delaunay](assets/examples/02_triangulation/conforming_delaunay.png) | ![Constrained Delaunay](assets/examples/02_triangulation/constrained_delaunay.png) | ![Delaunay Triangulation](assets/examples/02_triangulation/delaunay.png) |
| **2.1 Conforming Delaunay** (min_angle=20, max_area=0) | **2.2 Constrained Delaunay** | **2.3 Delaunay Triangulation** |
| Adds Steiner points to achieve full Delaunay property with constraint edges. Supports quality refinement. | Delaunay triangulation with required edges that must appear. Uses Shewchuk's Triangle when available. | Triangulation maximizing the minimum angle. Dual of the Voronoi diagram. |

|:---:|:---:|:---:|
| ![Voronoi Dual Graph](assets/examples/02_triangulation/voronoi_dual.png) | ![Weighted (Regular) Triangulation](assets/examples/02_triangulation/weighted_triangulation.png) |  |
| **2.4 Voronoi Dual Graph** (store_voronoi=True, finite_only=False) | **2.5 Weighted (Regular) Triangulation** (weight_scale=0.3) |  |
| Connect points whose Voronoi cells share a boundary edge. | Delaunay generalization with per-point weights via the paraboloid lifting method. |  |

---

### 3 · Spanning Tree-Based Graphs

Minimal or constrained trees connecting all nodes.

|:---:|:---:|:---:|
| ![Euclidean MST](assets/examples/03_spanning/emst.png) | ![k-MST Overlay](assets/examples/03_spanning/k_mst_overlay.png) | ![MST (Borůvka's)](assets/examples/03_spanning/mst_boruvka.png) |
| **3.1 Euclidean MST** | **3.2 k-MST Overlay** (k=3, penalty=2, noise_scale=0.1) | **3.3 MST (Borůvka's)** |
| Minimum spanning tree via Delaunay triangulation. O(n log n) instead of O(n²). | Union of 3 diverse spanning trees. Denser than a single MST but much sparser than complete. | Parallel-friendly MST: each round merges components via their cheapest outgoing edge. O(log n) rounds. |

|:---:|:---:|:---:|
| ![MST (Kruskal's)](assets/examples/03_spanning/mst_kruskal.png) | ![MST (Prim's)](assets/examples/03_spanning/mst_prim.png) | ![Random Spanning Tree](assets/examples/03_spanning/random_spanning.png) |
| **3.4 MST (Kruskal's)** | **3.5 MST (Prim's)** (start_vertex=0) | **3.6 Random Spanning Tree** (use_weights=False) |
| Minimum spanning tree via sorted edge insertion with Union-Find cycle detection. | Minimum spanning tree grown from a start vertex. Greedy vertex-centric approach with a priority queue. | Uniformly random spanning tree via Wilson's loop-erased random walk algorithm. |

---

### 4 · Random Graph Models

Probabilistic models that generate edges according to various distributions.

|:---:|:---:|:---:|
| ![Barabási–Albert](assets/examples/04_random_models/barabasi_albert.png) | ![Chung–Lu](assets/examples/04_random_models/chung_lu.png) | ![Configuration Model](assets/examples/04_random_models/configuration.png) |
| **4.1 Barabási–Albert** (m=2) | **4.2 Chung–Lu** | **4.3 Configuration Model** (remove_self_loops=True, remove_multi_edges=True) |
| Preferential attachment: each new node adds m=2 edges. Produces power-law degree distribution. | Random graph with specified expected degree sequence. | Random graph with a prescribed degree sequence via stub matching. |

|:---:|:---:|:---:|
| ![Erdős–Rényi G(n, m)](assets/examples/04_random_models/erdos_renyi_gnm.png) | ![Erdős–Rényi G(n, p)](assets/examples/04_random_models/erdos_renyi_gnp.png) | ![Forest Fire](assets/examples/04_random_models/forest_fire.png) |
| **4.4 Erdős–Rényi G(n, m)** (m=60) | **4.5 Erdős–Rényi G(n, p)** (p=0.15) | **4.6 Forest Fire** (p_forward=0.35, p_backward=0.2) |
| Exactly 60 edges chosen uniformly at random. | Each edge included independently with probability p=0.15. | Nodes burn through neighbors (p_fwd=0.35, p_bwd=0.2). Produces densification. |

|:---:|:---:|:---:|
| ![Holme–Kim](assets/examples/04_random_models/holme_kim.png) | ![Kronecker Graph](assets/examples/04_random_models/kronecker.png) | ![Price's Model](assets/examples/04_random_models/price.png) |
| **4.7 Holme–Kim** (m=2, p=0.5) | **4.8 Kronecker Graph** | **4.9 Price's Model** (m=3, a=1) |
| BA + triad formation: m=2, p_triad=0.5. Power-law degrees with tunable clustering. | Recursive Kronecker product of a 2×2 initiator matrix. Produces realistic heavy-tailed networks. | Directed preferential attachment (citations). m=3 out-edges, attractiveness a=1.0. |

|:---:|:---:|:---:|
| ![Random Geometric](assets/examples/04_random_models/random_geometric.png) | ![Random Regular](assets/examples/04_random_models/random_regular.png) | ![Stochastic Block Model](assets/examples/04_random_models/sbm.png) |
| **4.10 Random Geometric** (r=1) | **4.11 Random Regular** (d=3, max_retries=100) | **4.12 Stochastic Block Model** (p_within=0.4, p_between=0.05) |
| Connect nodes within Euclidean distance r=1.0. | Every node has exactly degree d=3. | Community structure: p_within=0.4, p_between=0.05. |

|:---:|:---:|:---:|
| ![Watts–Strogatz](assets/examples/04_random_models/watts_strogatz.png) |  |  |
| **4.13 Watts–Strogatz** (k=4, p=0.3) |  |  |
| Small-world model: ring lattice (k=4) with p=0.3 rewiring probability. |  |  |

---

### 5 · Lattice & Structured Graphs

Deterministic, regular graph topologies. *These use their own canonical node positions.*

|:---:|:---:|:---:|
| ![Complete Bipartite](assets/examples/05_lattice/complete_bipartite.png) | ![Grid Graph (4-connected)](assets/examples/05_lattice/grid_2d.png) | ![Hexagonal Lattice](assets/examples/05_lattice/hexagonal.png) |
| **5.1 Complete Bipartite** | **5.2 Grid Graph (4-connected)** (eight_connected=False) | **5.3 Hexagonal Lattice** |
| Two sets: every node in A connects to every node in B. | Regular 2D rectangular lattice with cardinal (and optional diagonal) neighbors. | Honeycomb tiling: degree-3 interior nodes. Planar and bipartite. |

|:---:|:---:|:---:|
| ![Hypercube Graph](assets/examples/05_lattice/hypercube.png) | ![Petersen Graph](assets/examples/05_lattice/petersen.png) | ![Ring / Cycle Graph](assets/examples/05_lattice/ring.png) |
| **5.4 Hypercube Graph** | **5.5 Petersen Graph** (n=5, k=2) | **5.6 Ring / Cycle Graph** |
| Nodes are d-bit strings; adjacent iff Hamming distance = 1. | The classic Petersen graph: 10 nodes, 15 edges, 3-regular. Famous counterexample in graph theory. | Each node connected to exactly two neighbors on a circle. |

|:---:|:---:|:---:|
| ![Star Graph](assets/examples/05_lattice/star.png) | ![Torus Graph](assets/examples/05_lattice/torus.png) | ![Triangular Lattice](assets/examples/05_lattice/triangular_lattice.png) |
| **5.7 Star Graph** | **5.8 Torus Graph** | **5.9 Triangular Lattice** |
| One hub connected to all n-1 leaf nodes. Diameter = 2. | 2D grid with wrap-around edges (periodic boundaries). Degree 4 everywhere. | Grid + diagonals: degree-6 interior nodes. Dual of hexagonal lattice. |

---

### 6 · Geometric Spanners

Sparse subgraphs that approximately preserve shortest-path distances.

|:---:|:---:|:---:|
| ![Greedy Spanner](assets/examples/06_spanners/greedy_spanner.png) | ![t-Spanner](assets/examples/06_spanners/t_spanner.png) | ![Theta (Θ) Graph](assets/examples/06_spanners/theta.png) |
| **6.1 Greedy Spanner** (t=2) | **6.2 t-Spanner** (t=2) | **6.3 Theta (Θ) Graph** (k=6) |
| Greedy geometric spanner (t=2.0). Near-optimal sparsity among all t-spanners. | Sparse subgraph with stretch factor t=2.0. All-pairs greedy edge filtering. | Projection-based spanner with 6 cones. Stretch ≤ 2.73. |

|:---:|:---:|:---:|
| ![WSPD Spanner](assets/examples/06_spanners/wspd_spanner.png) | ![Yao Graph](assets/examples/06_spanners/yao.png) |  |
| **6.4 WSPD Spanner** (s=4) | **6.5 Yao Graph** (k=6) |  |
| Well-Separated Pair Decomposition spanner (s=4.0). Theoretical stretch ≤ 3.00. | Nearest neighbor in each of 6 cones (θ=60°). |  |

---

### 7 · Approximate Nearest Neighbor Graphs

Graphs optimized for efficient nearest neighbor search.

|:---:|:---:|:---:|
| ![HNSW](assets/examples/07_ann/hnsw.png) | ![LSH-Based Graph](assets/examples/07_ann/lsh.png) | ![NN-Descent](assets/examples/07_ann/nn_descent.png) |
| **7.1 HNSW** (M=5, M0=10, ef_construction=32, mL=0.62) | **7.2 LSH-Based Graph** (k=5, n_tables=10, n_bits=8) | **7.3 NN-Descent** (k=5, max_iterations=20, delta=0.001, rho=1) |
| Hierarchical NSW: M=5, ef=32. Multi-layer skip-list-inspired ANN graph. | k-NN via 10 hash tables × 8 bits. Random hyperplane LSH. | Iterative k-NN refinement (k=5). 'Neighbor of neighbor is likely a neighbor.' |

|:---:|:---:|:---:|
| ![Navigable Small World](assets/examples/07_ann/nsw.png) | ![RP-Forest Graph](assets/examples/07_ann/rp_forest.png) | ![Vamana (DiskANN)](assets/examples/07_ann/vamana.png) |
| **7.4 Navigable Small World** (f=5, ef_construction=16) | **7.5 RP-Forest Graph** (k=5, n_trees=10, leaf_size=5) | **7.6 Vamana (DiskANN)** (R=5, alpha=1.2, L=20, n_passes=2) |
| Incremental insertion with greedy search. f=5 friends per node. | k-NN via 10 random projection trees (leaf_size=5). | Degree-bounded graph (R=5) with robust pruning (α=1.2). Medoid entry point. |

---

### 8 · Kernel & Similarity-Based Graphs

Graphs where edge weights come from kernel or similarity functions.

|:---:|:---:|:---:|
| ![Adaptive Bandwidth Kernel](assets/examples/08_kernel/adaptive_bandwidth.png) | ![Cosine Similarity](assets/examples/08_kernel/cosine.png) | ![Gaussian (RBF) Kernel](assets/examples/08_kernel/gaussian_rbf.png) |
| **8.1 Adaptive Bandwidth Kernel** (k_bandwidth=7, threshold=0.05, sparsify_knn=0) | **8.2 Cosine Similarity** (threshold=0.8, weighted=True) | **8.3 Gaussian (RBF) Kernel** (threshold=0.1) |
| Gaussian kernel with per-point σ from 7-th neighbor distance. Self-tuning for multi-scale data. | Connect pairs with cosine similarity > 0.8. Measures angular closeness. | Edge weights from exp(-‖x-y‖²/2σ²). σ=median, threshold=0.1. |

|:---:|:---:|:---:|
| ![Jaccard Similarity](assets/examples/08_kernel/jaccard.png) | ![Thresholded Similarity](assets/examples/08_kernel/thresholded.png) |  |
| **8.4 Jaccard Similarity** (method=spatial_bin, threshold=0.3, method_params={}) | **8.5 Thresholded Similarity** (measure=laplacian, threshold=0.3, measure_params={}) |  |
| Set-overlap similarity via spatial_bin features. J(A,B) = |A∩B|/|A∪B| ≥ 0.3. | Binarize laplacian similarity at threshold=0.3. |  |

---

### 9 · Visibility Graphs

Graphs derived from geometric visibility or time series.

*For time-series visibility, we sort points by x-coordinate and treat y as the series value.*

|:---:|:---:|:---:|
| ![Geometric Visibility](assets/examples/09_visibility/geometric_visibility.png) | ![Horizontal Visibility Graph](assets/examples/09_visibility/horizontal_visibility.png) | ![Natural Visibility Graph](assets/examples/09_visibility/natural_visibility.png) |
| **9.1 Geometric Visibility** (n_auto_obstacles=5, obstacle_radius=0.3, seed=42) | **9.2 Horizontal Visibility Graph** (directed=False, use_original_indices=True) | **9.3 Natural Visibility Graph** (directed=False, use_original_indices=True) |
| Connect points with unobstructed line-of-sight. Used for shortest-path planning with obstacles. | Simplified visibility: intermediate points must lie below min(y_i, y_j). Always a subgraph of the NVG. | Time series → graph: connect points with unobstructed line-of-sight over intermediate values. |

---

### 10 · Data-Driven / Learned Graphs

Graphs inferred from statistical relationships between features.

*Each point's (x, y) coordinates generate spatially correlated multivariate observations.*

|:---:|:---:|:---:|
| ![Correlation Graph](assets/examples/10_data_driven/correlation.png) | ![Expansion Graph](assets/examples/10_data_driven/expansion.png) | ![Graphical LASSO](assets/examples/10_data_driven/glasso.png) |
| **10.1 Correlation Graph** (threshold=0.5, n_samples=500, length_scale=1, use_absolute=True) | **10.2 Expansion Graph** (method=diffusion, percentile=20, k_initial=5, diffusion_steps=3, diffusion_threshold=0.01) | **10.3 Graphical LASSO** (alpha=0.1, n_samples=500, length_scale=1, max_iter=200, tol=0.0001) |
| Connect variables with Pearson |ρ| ≥ 0.5. From 500 synthetic observations. | Diffusion affinity graph: 3-step random walk on initial k-NN. | Sparse inverse covariance estimation (α=0.1). L1 penalty auto-selects graph structure. |

|:---:|:---:|:---:|
| ![Mutual Information](assets/examples/10_data_driven/mutual_information.png) | ![Partial Correlation](assets/examples/10_data_driven/partial_correlation.png) |  |
| **10.4 Mutual Information** (threshold=0.1, n_samples=500, length_scale=1, estimator=ksg, n_bins=20, ksg_k=5, nonlinear=False) | **10.5 Partial Correlation** (threshold=0.15, n_samples=500, length_scale=1, regularization=0.01) |  |
| MI-based edges (ksg estimator). Detects all dependencies, not just linear. | Gaussian graphical model: direct linear relationships via precision matrix. |ρ_partial| ≥ 0.15. |  |

---

### 11 · Miscellaneous

Other notable graph construction methods.

|:---:|:---:|:---:|
| ![Ball Tree Neighbor Graph](assets/examples/11_misc/balltree_neighbor.png) | ![Cayley Graph](assets/examples/11_misc/cayley.png) | ![De Bruijn Graph](assets/examples/11_misc/debruijn.png) |
| **11.1 Ball Tree Neighbor Graph** (k=5, radius=1, mode=knn, leaf_size=10) | **11.2 Cayley Graph** (group=cyclic, n=15) | **11.3 De Bruijn Graph** (k=2, n=5, as_undirected=True) |
| k-NN (k=5) via ball tree. Handles arbitrary metrics and higher dimensions. | Cay(ℤ_15, generators). Algebraic structure as a graph. | B(2,5): 32 nodes from 5-length strings over 2-letter alphabet. Edges represent sequence overlaps. |

|:---:|:---:|:---:|
| ![Disk Graph](assets/examples/11_misc/disk.png) | ![Intersection Graph](assets/examples/11_misc/intersection.png) | ![KD-Tree Neighbor Graph](assets/examples/11_misc/kdtree_neighbor.png) |
| **11.4 Disk Graph** (r=0.5, mode=uniform, adaptive_k=3, adaptive_scale=0.5) | **11.5 Intersection Graph** (shape=circle, radius_mean=0.7, radius_std=0.3) | **11.6 KD-Tree Neighbor Graph** (k=5, leaf_size=10) |
| Unit disk model: connect if distance ≤ 1.00 (radius r=0.5). | Connect nodes whose circles overlap. Mean radius=0.7. | k-NN (k=5) via KD-tree. Same result as brute-force but O(n log n) construction. |

|:---:|:---:|:---:|
| ![Power Diagram Graph](assets/examples/11_misc/power_diagram.png) |  |  |
| **11.7 Power Diagram Graph** (weight_scale=0.5) |  |  |
| Dual of the weighted Voronoi diagram (Laguerre tessellation). Generalizes Voronoi dual with per-site weights. |  |  |

---

## 📊 Algorithm Comparison Table

| Category | # Algorithms | Spatial? | Deterministic? |
|---|---|---|---|
| Proximity & Distance-Based Graphs | 10 | ✅ | ✅ |
| Triangulation-Based Graphs | 5 | ✅ | ✅ |
| Spanning Tree-Based Graphs | 6 | ✅ | Mostly ✅ |
| Random Graph Models | 13 | Some | ❌ |
| Lattice & Structured Graphs | 9 | ❌ | ✅ |
| Geometric Spanners | 5 | ✅ | ✅ |
| Approximate Nearest Neighbor Graphs | 6 | ✅ | ❌ |
| Kernel & Similarity-Based Graphs | 5 | ✅ | ✅ |
| Visibility Graphs | 3 | ✅ | ✅ |
| Data-Driven / Learned Graphs | 5 | ❌ | ✅ |
| Miscellaneous | 7 | Mixed | Mixed |
| **Total** | **74** | | |

<!-- GALLERY_END -->

## ⚙️ API Usage

Every algorithm follows a consistent interface:

```python
from graphgallery.points import make_two_cluster_layout
from graphgallery.proximity import KNNGraph
from graphgallery.viz import plot_graph

# Generate the canonical 30-point layout
points = make_two_cluster_layout(seed=42)

# Build a k-NN graph
builder = KNNGraph(k=5)
G = builder.build(points)

# Visualize
fig = plot_graph(G, points, title="k-NN Graph (k=5)")
fig.savefig("knn_example.png", dpi=150)
```

All builders inherit from `GraphBuilder` and implement:

```python
class GraphBuilder(ABC):
    @abstractmethod
    def build(self, points: np.ndarray) -> nx.Graph:
        """Build a graph from an (n, d) array of points."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Category slug (e.g., 'proximity', 'spanning')."""
        ...
```

### Batch Generation

```bash
# Generate all 74 graphs and save images
python scripts/generate_all_examples.py

# Generate only a specific category
python scripts/generate_all_examples.py --category proximity

# Generate a single algorithm
python scripts/generate_all_examples.py --algorithm knn --k 5
```

---

## 🗂️ Project Structure

```
graphgallery/          Core library — one subpackage per category
├── points.py          Canonical point layout generator
├── viz.py             Shared matplotlib visualization
├── base.py            Abstract GraphBuilder base class
├── proximity/         10 algorithms
├── triangulation/     5 algorithms
├── spanning/          6 algorithms
├── random_models/     13 algorithms
├── lattice/           9 algorithms
├── spanners/          5 algorithms
├── ann/               6 algorithms
├── kernel/            5 algorithms
├── visibility/        3 algorithms
├── data_driven/       5 algorithms
└── misc/              7 algorithms

scripts/               CLI tools for generation
tests/                 pytest test suite
notebooks/             Interactive Jupyter exploration
assets/                Generated images and banner
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific category
pytest tests/test_proximity.py -v

# Run with coverage
pytest tests/ --cov=graphgallery --cov-report=html
```

---

## 📊 Algorithm Comparison Table

| Category | # Algorithms | Spatial? | Deterministic? |
|---|---|---|---|
| Proximity & Distance | 10 | ✅ | ✅ |
| Triangulation | 5 | ✅ | ✅ |
| Spanning Trees | 6 | ✅ | Mostly ✅ |
| Random Models | 13 | Some | ❌ |
| Lattice & Structured | 9 | ❌ | ✅ |
| Geometric Spanners | 5 | ✅ | ✅ |
| ANN Graphs | 6 | ✅ | ❌ |
| Kernel & Similarity | 5 | ✅ | ✅ |
| Visibility | 3 | ✅ | ✅ |
| Data-Driven | 5 | ❌ | ✅ |
| Miscellaneous | 7 | Mixed | Mixed |
| **Total** | **74** | | |

---

## 🤝 Contributing

Contributions are welcome! To add a new graph algorithm:

1. **Fork** the repo and create a feature branch
2. **Add** your builder in the appropriate `graphgallery/<category>/` subpackage
3. **Inherit** from `GraphBuilder` and implement `build()`, `name`, and `category`
4. **Add tests** in `tests/test_<category>.py`
5. **Run** `python scripts/generate_all_examples.py --algorithm your_algo` to generate the image
6. **Submit** a PR with the new code, test, and image

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with 🔗 and 🐍
</p>