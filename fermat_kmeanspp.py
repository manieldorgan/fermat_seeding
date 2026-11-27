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
from scipy.sparse.csgraph import connected_components, dijkstra
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
seed = 0


def config_label(kind, n_seeds):
    """Label used for a config in plots, checks, and the table."""
    return f"{kind} (k={n_seeds})"


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


def build_connected_knn(
    X: np.ndarray, k_start: int = 10, step: int = 5
) -> tuple[sp.csr_array, int]:
    """Build a symmetric, connected kNN graph with Euclidean edge lengths.

    Starts from each point's `k_start` nearest neighbors and increases k
    by `step` until the graph has a single connected component. An edge
    is kept if either endpoint is among the other's k nearest neighbors.

    Parameters
    ----------
    X : ndarray of shape (n, 2)
        Point coordinates.
    k_start : int, default=10
        Initial number of neighbors per point.
    step : int, default=5
        Amount k increases by after each disconnected attempt.

    Returns
    -------
    G : scipy.sparse.csr_array of shape (n, n)
        Symmetric adjacency matrix whose stored entries are Euclidean
        distances between neighboring points. Raising the entries to the
        power p gives the Fermat cost graph.
    k : int
        Number of neighbors used in the final, connected graph.
    """
    k = k_start
    D = sp.csr_array(neighbors.kneighbors_graph(X, k, mode="distance"))
    # Symmetrize by keeping an edge if either point is one of the other's k nearest
    G = D.maximum(D.T)

    n_components, _ = connected_components(G, directed=False)
    while n_components > 1:  # keep increasing k until the graph is connected
        k += step
        D = sp.csr_array(neighbors.kneighbors_graph(X, k, mode="distance"))
        G = D.maximum(D.T)
        n_components, _ = connected_components(G, directed=False)

    return G, k


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


if __name__ == "__main__":
    # DATASETS:
    kinds = list(dict.fromkeys(kind for kind, _ in configs))
    datasets = {kind: make_dataset(kind, seed=seed) for kind in kinds}

    hp.run_checks(
        datasets, centers=clust_centers, save_path="figures/datasets.png"
    )
    os.makedirs("data", exist_ok=True)
    for kind, (X, y) in datasets.items():
        np.savez(f"data/{kind}.npz", X=X, y=y)

    # GRAPHS:
    graphs, k_nn_used = {}, {}
    for kind, (X, y) in datasets.items():
        knn, k_nn = build_connected_knn(X)
        k_nn_used[kind] = k_nn
        graphs[kind] = {p: knn.power(p) for p in p_values}
        print(f"{kind}: k_nn = {k_nn}")

    # SEED CHECKS:
    for p in p_values:
        panels, seeds = {}, {}
        for kind, n_seeds in configs:
            label = config_label(kind, n_seeds)
            G = graphs[kind][p]

            a = fermat_kmeanspp(G, n_seeds, np.random.default_rng(seed))
            b = fermat_kmeanspp(G, n_seeds, np.random.default_rng(seed))
            print(f"p={p} {label}: reproducible = {np.array_equal(a, b)}")

            panels[label] = datasets[kind]
            seeds[label] = a

        hp.run_checks(
            panels,
            centers=clust_centers,
            seeds=seeds,
            title=f"Fermat seeds, p = {p}",
            save_path=f"figures/seeds_p{p}.png",
        )
