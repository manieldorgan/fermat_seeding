import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp


def add_noise(dataset, labels, noise_frac, rng, pad=1.0):
    """Add noise points with label -1 throughout a 2D point cloud dataset with padding around dataset boundary.
    
    Parameters
    ----------
    dataset: ...
    labels: ...
    noise_frac: ...
    rng: ...
    pad: ...

    Returns
    -------
    X: ndarray of shape (len(dataset) + len(dataset) * noise_frac, 2)
    Y: ndarray of shape (len(dataset) + len(dataset) * noise_frac, )
    """

    low = dataset.min(axis=0) - pad             
    high = dataset.max(axis=0) + pad
    m = int(noise_frac * len(dataset))                                        # noise = noise_frac of the clean points
    noise = rng.uniform(low, high, size=(m, 2))
    X = np.concatenate((dataset, noise), axis=0)                              # adds noise points at the end of dataset
    y = np.concatenate((labels, np.full((m, ), -1, dtype=labels.dtype)), axis=0)  # adds noise point labels to label array
    
    return X, y


def check_datasets(datasets, centers=None, tol=0.1):
    """Plot each dataset colored by label and print basic checks.

    Parameters
    ----------
    datasets : dict
        Maps a dataset name to its (X, y) pair, with X of shape (n, 2)
        and y of shape (n,). Noise points are labeled -1.
    centers : array-like of shape (k, 2), optional
        Expected cluster centers. The means check runs only on datasets
        whose non-noise labels are exactly 0, ..., k-1.
    tol : float, default=0.1
        Absolute tolerance for comparing cluster means to `centers`.
    """

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 10), squeeze=False)
    fig.subplots_adjust(wspace=0.25)

    for (kind, (X, y)), ax in zip(datasets.items(), axes[0]):
        clean = y != -1
        noise = y == -1

        # PLOT:
        ax.scatter(X[clean, 0], X[clean, 1], c=y[clean], cmap="tab10", s=5)
        ax.scatter(X[noise, 0], X[noise, 1], color="gray", s=5, alpha=0.6)
        ax.set_title(kind)
        ax.set_aspect("equal")

        # CHECKS:
        labels, counts = np.unique(y, return_counts=True)
        print(f"{kind}: X {X.shape}, y {y.shape}")
        print("  labels:", labels, " counts:", counts)

        clean_labels = np.unique(y[clean])
        if centers is not None and np.array_equal(clean_labels, np.arange(len(centers))):
            means = np.array([X[y == i].mean(axis=0) for i in clean_labels])
            close = np.allclose(means, centers, atol=tol)
            print(f"  cluster means within {tol} of centers: {close}")
            if not close:
                print("  means:\n", means)

    plt.show()
