import numpy as np
from scipy import sparse as sp
from sklearn import datasets, neighbors, cluster
import pandas as pd
import helpers as hp
import os


points_per_clust = 500                                          # per blob; 4 blobs -> 2000 points
clust_centers = [(0, 0), (0, 5), (4.33, -2.5), (-4.33, -2.5)]   # 120 degrees between radial lines (arbitrary choice)
seed = 0                                                        # for reproducibility


def make_dataset(kind, noise_frac=None, seed=0):
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
    noise_frac : float, optional
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
        X, y = datasets.make_blobs(
            n_samples=[points_per_clust] * len(clust_centers),   # list = count per center
            centers=clust_centers,
            cluster_std=[0.3, 0.5, 0.7, 1.0],
            random_state=seed,
            )
    elif kind == "moons":
        X, y = datasets.make_moons(
            n_samples=2000,
            noise=0.1,
            random_state=seed
            )
    else:
        raise ValueError(f"Invalid kind: '{kind}'")

    # Adding uniform noise around the dataset in feature space.
    if kind == "unequal_noise":
        if noise_frac is None: 
            noise_frac = 0.1 

        pad = 1.0
        X, y = hp.add_noise(X, y, noise_frac, rng, pad)

    return X, y


def build_connected_knn(X, k_start=10, step=5):
    """
    Returns sparse, symmetric, connected, UNWEIGHTED knn graph
    """
    k = k_start
    D = sp.csr_array(neighbors.kneighbors_graph(X, k, mode="distance"))
    G = D.maximum(D.T)                                                   # Symmetrize by keeping an edge if either point is one of the other's k nearest
    
    connected = sp.csgraph.connected_components(G, directed=False)
    while not connected:                                                 # keep increasing k until the graph is connected
        k += step
        D = sp.csr_array(neighbors.kneighbors_graph(X, k, mode="distance"))
        G = D.maximum(D.T)
        connected = sp.csgraph.connected_components(G, directed=False)

    return G, k


if __name__ == "__main__":
    datasets = {
                kind: make_dataset(kind, seed=seed)
                for kind in ["unequal", "unequal_noise", "moons"]
                }

    hp.check_datasets(datasets, centers=clust_centers)
    os.makedirs("data", exist_ok=True)
    for kind, (X, y) in datasets.items():
        np.savez(f"data/{kind}.npz", X=X, y=y)

    for dataset in datasets:
        print(build_connected_knn(datasets[dataset][0]))
    
    
    
    """LOAD CHECK
    with np.load("data/unequal.npz") as data:
        X_loaded, y_loaded = data["X"], data["y"]
    print(np.array_equal(X_loaded, datasets["unequal"][0]), np.array_equal(y_loaded, datasets["unequal"][1]))
    """