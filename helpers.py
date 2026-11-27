import os

import matplotlib.pyplot as plt
import numpy as np


def run_checks(
    datasets,
    centers=None,
    seeds=None,
    title=None,
    tol=0.1,
    save_path=None,
):

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.subplots_adjust(wspace=0.25)

    for (kind, (X, y)), ax in zip(datasets.items(), axes[0]):
        clean = y != -1
        noise = y == -1

        # PLOT POINTS
        ax.scatter(X[clean, 0], X[clean, 1], c=y[clean], cmap="tab10", s=5)
        ax.scatter(X[noise, 0], X[noise, 1], color="gray", s=5, alpha=0.6)
        ax.set_title(kind)
        ax.set_aspect("equal")

        # DATASET CHECKS
        labels, counts = np.unique(y, return_counts=True)
        print(f"{kind}: X {X.shape}, y {y.shape}")
        print("  labels:", labels, " counts:", counts)

        clean_labels = np.unique(y[clean])
        if centers is not None and np.array_equal(
            clean_labels, np.arange(len(centers))
        ):
            means = np.array([X[y == i].mean(axis=0) for i in clean_labels])
            close = np.allclose(means, centers, atol=tol)
            print(f"  cluster means within {tol} of centers: {close}")
            if not close:
                print("  means:\n", means)

        # SEEDS PLOT AND CHECKS
        if seeds is not None and kind in seeds:
            s = seeds[kind]
            ax.scatter(
                X[s, 0],
                X[s, 1],
                marker="x",
                color="black",
                s=120,
                linewidths=2.5,
            )

            seed_labels = y[s]
            n_true = len(clean_labels)
            print("  seeds:", s, " distinct:", len(np.unique(s)) == len(s))
            print(
                "  seeds per cluster:",
                np.bincount(seed_labels[seed_labels >= 0], minlength=n_true),
            )
            if noise.any():
                print("  noise picks:", (seed_labels == -1).sum())

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
