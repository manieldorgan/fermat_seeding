# The function fermat_kmeanspp is adapted from scikit-learn's _kmeans_plusplus
# (version 1.9.1, https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/cluster/_kmeans.py)
# and is distributed under the following license. The rest of the file is original.


# BSD 3-Clause License

# Copyright (c) 2007-2026 The scikit-learn developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.csgraph import (
    connected_components,
    dijkstra,
    minimum_spanning_tree,
)
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn import cluster, neighbors
from sklearn.datasets import make_blobs, make_moons

import helpers as hp

points_per_clust = 500
clust_centers = [
    (0, 0),
    (0, 5),
    (4.33, -2.5),
    (-4.33, -2.5),
]
configs = [
    ("unequal", 4),
    ("unequal_noise", 4),
    ("moons", 4),
    ("moons", 2),
]
p_values = [1, 2, 4, 8, 16]
metrics = ["euclidean", *p_values]
k_nn = 10
seed = 0


def add_noise(
    X: np.ndarray,
    y: np.ndarray,
    noise_frac: float,
    rng: np.random.Generator,
    pad: float = 1.0,
):
    """Add uniform background noise to a 2-D point cloud.

    Noise points are drawn uniformly from the bounding box of `X`,
    padded by `pad` on each side, and labeled -1.

    Parameters
    ----------
    X : ndarray of shape (n, 2)
        Clean point coordinates.
    y : ndarray of shape (n,)
        Labels for the clean points. Must be a signed integer type.
    noise_frac : float
        Number of noise points as a fraction of the clean points,
        m = int(noise_frac * n).
    rng : numpy.random.Generator
        Random generator used to draw the noise.
    pad : float, default=1.0
        Distance the bounding box extends past the data on each side.

    Returns
    -------
    X_noisy : ndarray of shape (n + m, 2)
        Clean points followed by the noise points.
    y_noisy : ndarray of shape (n + m,)
        Labels followed by m entries of -1.
    """
    low = X.min(axis=0) - pad
    high = X.max(axis=0) + pad
    m = int(noise_frac * len(X))

    noise = rng.uniform(low, high, size=(m, 2))

    # append noise points and their -1 labels to the end
    X_noisy = np.concatenate((X, noise), axis=0)
    y_noisy = np.concatenate((y, np.full(m, -1, dtype=y.dtype)))

    return X_noisy, y_noisy


def make_dataset(
    kind: str, noise_frac: float | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2-D test dataset for the seeding experiments.

    "unequal" is four Gaussian blobs (500 points each) with standard
    deviations 0.3, 0.5, 0.7 and 1.0, centered at `clust_centers`.
    "unequal_noise" is the same blobs plus uniform background noise.
    "moons" is scikit-learn's two interleaving half-circles (2000 points,
    noise=0.1). Noise is only added for "unequal_noise".

    Parameters
    ----------
    kind : {"unequal", "unequal_noise", "moons"}
        Which dataset to generate.
    noise_frac : float or None, optional
        Number of noise points as a fraction of the clean points, so 0.1
        with 2000 clean points adds 200 (about 9.1% of the final dataset).
        Noise is drawn uniformly from the data's bounding box, padded by
        1.0 on each side. Defaults to 0.1 for "unequal_noise"; ignored
        for other kinds.
    seed : int, default=0
        Random seed for both the base dataset and the noise.

    Returns
    -------
    X : ndarray of shape (n, 2)
        Point coordinates. n is 2000, plus int(noise_frac * 2000) for
        "unequal_noise".
    y : ndarray of shape (n,)
        Labels: blob or moon index for clean points, -1 for noise.

    Raises
    ------
    ValueError
        If `kind` is not one of the three options.
    """

    rng = np.random.default_rng(seed)

    if kind in ("unequal", "unequal_noise"):
        X, y = make_blobs(
            n_samples=[points_per_clust] * len(clust_centers),
            centers=clust_centers,
            cluster_std=[0.3, 0.5, 0.7, 1.0],
            random_state=seed,
        )
    elif kind == "moons":
        X, y = make_moons(n_samples=2000, noise=0.1, random_state=seed)
    else:
        raise ValueError(f"Invalid kind: '{kind}'")

    # Adding uniform noise around the dataset in feature space.
    if kind == "unequal_noise":
        if noise_frac is None:
            noise_frac = 0.1

        X, y = add_noise(X, y, noise_frac, rng)

    return X, y


def build_graph(X: np.ndarray, k: int = 10) -> tuple[sp.csr_array, int, int]:
    """Build a connected, symmetric graph: kNN graph union Euclidean MST.

    The kNN graph keeps the graph local and sparse; the minimum spanning
    tree of the complete Euclidean graph guarantees connectivity by adding
    the shortest edges needed to join any disconnected pieces.

    Parameters
    ----------
    X : ndarray of shape (n, 2)
        Point coordinates.
    k : int, default=10
        Number of neighbors per point in the kNN graph.

    Returns
    -------
    G : scipy.sparse.csr_array of shape (n, n)
        Symmetric adjacency matrix whose stored entries are Euclidean
        distances. Raising the entries to the power p gives the Fermat
        cost graph.
    n_components : int
        Number of connected components of the kNN graph alone.
    n_added : int
        Number of edges the MST contributed that the kNN graph lacked.
    """
    # kNN graph, symmetrized by keeping an edge if either point is a neighbor of the other
    knn = sp.csr_array(neighbors.kneighbors_graph(X, k, mode="distance"))
    knn = knn.maximum(knn.T)
    n_components, _ = connected_components(knn, directed=False)

    # MST of euclidean graph, also symmetrized
    mst = sp.csr_array(minimum_spanning_tree(squareform(pdist(X))))
    mst = mst.maximum(mst.T)

    G = knn.maximum(mst)
    n_added = (G.nnz - knn.nnz) // 2  # Undirected edges are stored twice
    # ^^^ counts ALL mst edges missing, not just bridges between clusters

    return G, n_components, n_added


def fermat_kmeanspp(
    G: sp.csr_array,
    n_clusters: int,
    rng: np.random.Generator,
    n_local_trials: int | None = None,
) -> np.ndarray:
    """Choose k-means++ seeds using shortest-path distances on a graph.

    Adapted from scikit-learn's _kmeans_plusplus, with Euclidean distances
    replaced by shortest-path (Dijkstra) distances on G. When G's edge
    weights are ||x - y||^p, these are Fermat distances. Seeds are sampled
    with probability proportional to squared distance to the nearest
    existing seed (the D^2 rule), and all points have unit weight.

    Parameters
    ----------
    G : scipy.sparse.csr_array of shape (n, n)
        Symmetric, connected, weighted adjacency matrix. Edge weights are
        path costs (larger means farther).
    n_clusters : int
        The number of seeds to choose.
    rng : numpy.random.Generator
        Random generator used for all sampling.
    n_local_trials : int or None, default=None
        The number of candidate seeds sampled at each step (except the
        first); the one that most reduces the total squared distance is
        kept. If None, uses 2 + int(log(n_clusters)), scikit-learn's
        default.

    Returns
    -------
    indices : ndarray of shape (n_clusters,)
        Indices of the chosen seeds, as rows of G (equivalently, rows of
        the data array G was built from).
    """
    n_samples = G.shape[0]

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # scikit-learn's default
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = rng.choice(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = dijkstra(G, directed=False, indices=[center_id]) ** 2
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = rng.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(
            candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids
        )

        # Compute distances to center candidates
        distance_to_candidates = (
            dijkstra(G, directed=False, indices=candidate_ids) ** 2
        )

        # update closest distances squared and potential for each candidate
        np.minimum(
            closest_dist_sq, distance_to_candidates, out=distance_to_candidates
        )
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        indices[c] = candidate_ids[best_candidate]

    return indices


def config_label(kind, n_seeds):
    """Label used for a config in plots, checks, and the table."""
    return f"{kind} (k={n_seeds})"


def metric_label(metric):
    """Display name for a metric: "Euclidean" or "p = <p>"."""
    return "Euclidean" if metric == "euclidean" else f"p = {metric}"


def distances_to_seeds(
    seeds: np.ndarray,
    X: np.ndarray | None = None,
    G: sp.csr_array | None = None,
):
    """(n_seeds, n) distances: shortest paths on G if given, else Euclidean on X."""
    if G is not None:
        return dijkstra(G, directed=False, indices=seeds)
    return cdist(X[seeds], X)


def seed_points(metric, X, G, n_seeds, run):
    """k-means++ seed indices for one run, in the given metric."""
    if metric == "euclidean":
        return cluster.kmeans_plusplus(X, n_seeds, random_state=run)[1]
    return fermat_kmeanspp(G, n_seeds, np.random.default_rng(run))


def random_seeds(n, n_seeds, run):
    """Uniformly random distinct seed indices for one run (the baseline)."""
    return np.random.default_rng([run, 1]).choice(n, n_seeds, replace=False)


def representation_cost(dist_to_seeds: np.ndarray) -> tuple[np.ndarray, float]:
    """Assign each point to its nearest seed and total the distances.

    The cost is the sum, over all points, of the distance to the nearest
    seed. The function is metric-agnostic: pass Fermat distances
    (Dijkstra on the p graph) or Euclidean distances (cdist), and the
    cost is in that metric.

    Parameters
    ----------
    dist_to_seeds : ndarray of shape (n_seeds, n)
        Entry [i, j] is the distance from seed i to point j.

    Returns
    -------
    regions : ndarray of shape (n,)
        Index (0 to n_seeds - 1) of the nearest seed for each point.
    cost : float
        Sum of each point's distance to its nearest seed.
    """
    nearest = dist_to_seeds.min(axis=0)
    regions = dist_to_seeds.argmin(axis=0)
    cost = nearest.sum()
    return regions, cost


def seed_accuracy(
    regions: np.ndarray, seeds: np.ndarray, y: np.ndarray
) -> float:
    """Fraction of clean points whose nearest seed has their true label.

    Each point is assigned the label of its nearest seed. Accuracy is
    measured only on clean points (y != -1). A seed on a noise point
    has label -1, so every clean point in its region counts as wrong.

    Parameters
    ----------
    regions : ndarray of shape (n,)
        Index of the nearest seed for each point, from representation_cost.
    seeds : ndarray of shape (n_seeds,)
        Seed indices into the dataset.
    y : ndarray of shape (n,)
        True labels; noise points are labeled -1.

    Returns
    -------
    float
        Accuracy in [0, 1] over points with y != -1.
    """
    predicted = y[seeds][regions]  # (n,): label of each point's nearest seed
    clean = y != -1
    return float((predicted[clean] == y[clean]).mean())


# TODO: Move most random dev diagostics to scratch file before adding sweep
if __name__ == "__main__":
    # DATASETS:
    kinds = list(dict.fromkeys(kind for kind, _ in configs))
    datasets = {kind: make_dataset(kind, seed=seed) for kind in kinds}

    hp.run_setup_checks(
        datasets, centers=clust_centers, save_path="figures/datasets.png"
    )
    os.makedirs("data", exist_ok=True)
    for kind, (X, y) in datasets.items():
        np.savez(f"data/{kind}.npz", X=X, y=y)

    # GRAPHS:
    graphs, graph_info = {}, {}
    for kind, (X, y) in datasets.items():
        G, n_comp, n_added = build_graph(X, k=k_nn)
        graph_info[kind] = {
            "knn_components": n_comp,
            "mst_edges_added": n_added,
        }
        graphs[kind] = {p: G.power(p) for p in p_values}

        n_after, _ = connected_components(G, directed=False)
        print(
            f"{kind}: kNN components = {n_comp}, MST edges added = {n_added}, "
            f"components after = {n_after}, avg degree = {G.nnz / G.shape[0]:.1f}, "
            f"symmetric = {(G != G.T).nnz == 0}"
        )

    # SEED CHECKS (run 0, every metric):
    for metric in metrics:
        panels, seeds = {}, {}
        for kind, n_seeds in configs:
            X, y = datasets[kind]
            G = None if metric == "euclidean" else graphs[kind][metric]
            label = config_label(kind, n_seeds)

            a = seed_points(metric, X, G, n_seeds, run=0)
            b = seed_points(metric, X, G, n_seeds, run=0)
            print(
                f"{metric_label(metric)} {label}: reproducible = {np.array_equal(a, b)}"
            )

            panels[label] = datasets[kind]
            seeds[label] = a

        hp.run_setup_checks(
            panels,
            centers=clust_centers,
            seeds=seeds,
            title=f"Seeds, {metric_label(metric)}",
            save_path=f"figures/seeds_{metric}.png",
        )

    # SCORING CHECKS (run 0, both methods, every metric):
    all_ok = True
    for kind, n_seeds in configs:
        X, y = datasets[kind]
        for metric in metrics:
            G = None if metric == "euclidean" else graphs[kind][metric]
            for method, s in [
                ("kmeans++", seed_points(metric, X, G, n_seeds, run=0)),
                ("random", random_seeds(len(X), n_seeds, run=0)),
            ]:
                D = distances_to_seeds(s, X=X, G=G)
                regions, cost = representation_cost(D)
                label = f"{config_label(kind, n_seeds)} | {metric_label(metric)} | {method}"
                all_ok &= hp.check_scoring(label, D, s, regions, cost)

    print("All scoring checks passed:", all_ok)

    # DIAGNOSTIC: how often does each blob get seeded?
    n_runs = 100
    for kind in ["unequal", "unequal_noise"]:
        X, y = datasets[kind]
        print(
            f"\n{kind}: fraction of {n_runs} runs with >= 1 seed in blobs 0-3, mean noise picks"
        )
        for metric in metrics:
            G = None if metric == "euclidean" else graphs[kind][metric]
            coverage, noise = hp.seed_coverage(
                y, lambda r: seed_points(metric, X, G, 4, r), n_runs
            )
            print(
                f"  {metric_label(metric):>9}: {np.round(coverage, 2)}  noise picks = {noise:.2f}"
            )

    print(
        "\nlabel of the Fermat run-0 first seed (index 1871) on unequal_noise:",
        datasets["unequal_noise"][1][1871],
    )

    # ACCURACY CHECKS:
    # 1. Clean unequal, p = 2, run 0: one seed per blob, so about 100%
    X, y = datasets["unequal"]
    G = graphs["unequal"][2]
    s = seed_points(2, X, G, 4, run=0)
    regions, _ = representation_cost(distances_to_seeds(s, G=G))
    acc = seed_accuracy(regions, s, y)
    print(
        f"unequal, p = 2, run 0: seeds per blob = {np.bincount(y[s], minlength=4)}, "
        f"accuracy = {acc:.3f} (expect ~1)"
    )

    # 2. All seeds on noise points: every region gets label -1, so exactly 0%
    X, y = datasets["unequal_noise"]
    s = np.flatnonzero(y == -1)[:4]
    regions, _ = representation_cost(distances_to_seeds(s, X=X))
    acc = seed_accuracy(regions, s, y)
    print(f"all-noise seeds: accuracy = {acc:.3f} (expect 0)")
    assert acc == 0.0

    # 3. Moons (k=2), Euclidean, run 0: if both seeds land in one moon, exactly 50%
    X, y = datasets["moons"]
    s = seed_points("euclidean", X, None, 2, run=0)
    regions, _ = representation_cost(distances_to_seeds(s, X=X))
    acc = seed_accuracy(regions, s, y)
    print(
        f"moons (k=2), Euclidean, run 0: seeds per moon = {np.bincount(y[s], minlength=2)}, "
        f"accuracy = {acc:.3f}"
    )
