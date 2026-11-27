# Fermat k-means++ seeding

k-means++ seeding using Fermat distances (shortest paths on a connected kNN graph with edge weights ‖x − y‖^p), compared with Euclidean k-means++ on three 2-D datasets for p = 1, 2, 4, 8.

## Setup

```
conda env create -f environment.yml
conda activate fermat-seeding
```

Tested with: Python [x.y], NumPy [x.y], SciPy [x.y], scikit-learn [x.y], matplotlib [x.y], pandas [x.y].

## Run

```
python fermat_kmeanspp.py
```

Outputs `data/*.npz` (datasets), `results/figure.png`, and `results/table.md`. Full run takes about [runtime].

## Score

Representation cost: the sum of distances from each point to its nearest seed, divided by the same sum for uniformly random seeds, measured in the same metric. Reported as mean ± std over 20 runs. Values below 1 mean k-means++ beats random seeding.

## Notes

- Noise in `unequal_noise`: [noise_frac × number of clean points] points drawn uniformly from the padded bounding box, labeled −1.
- kNN graph k needed for connectivity: [per dataset].
- `fermat_kmeanspp` is adapted from scikit-learn's `_kmeans_plusplus` (version [x.y], BSD 3-Clause license; notice included in the source file). That function is private and changes between versions.


## temp
$p = 1$ column measures path length along the data, not Euclidean distance (not necessarily clear)

changes made from scikit-learns k means++:
- dijkstra on symmetric knn graph with weights $||x_ij-y_ij||^p$ is the same as fermat distance on graph so we just use dijkstra in place of euclidean distances
- every point has weight 1 so sample_weight andm x_squared norms are unneeded
- we track seed indices instead of coordinates since the coordinates are less meaningful with fermat distances
- kept the default number of local trials for now.

add 4th configuration to readme

each p starts from the same point, so differences come from the metric rather than the starting draw.

NN ∪ MST to guarantee connectivity without inflating k