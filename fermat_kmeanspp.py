import numpy as np
from scipy import sparse as sp
from sklearn import datasets, neighbors, cluster
# import pandas as pd  (not yet installed in the glearning env)


points_per_clust = 500                                          # per blob; 4 blobs -> 2000 points
clust_centers = [(0, 0), (0, 5), (4.33, -2.5), (-4.33, -2.5)]   # 120 degrees between radial lines (arbitrary choice)
seed = 0                                                        # for reproducibility


def make_dataset(kind, noise_frac=0.0, seed=0):
    """
    parameter documentation to be added.
    Returns: X shape (n, 2), y shape (n,) labels (noise points labeled -1)
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
        X, y = datasets.make_moons(n_samples=2000, noise=0.1, random_state=seed)
    else:
        raise ValueError(f"Invalid kind: '{kind}'")

    if kind == "unequal_noise":
        pad = 1.0
        low = X.min(axis=0) - pad             # padded bounding box, shape (2,)
        high = X.max(axis=0) + pad
        m = int(noise_frac * len(X))          # noise = noise_frac of the clean points
        noise = rng.uniform(low, high, size=(m, 2))

        # TODO: stack `noise` onto X and m labels of -1 onto y

    return X, y


if __name__ == "__main__":
    ...
