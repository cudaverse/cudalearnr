# cudalearnr

`cudalearnr` is the reusable numerical algorithm layer of the **cudaverse**.
It provides GPU-aware implementations with explicit device reporting and
portable CPU fallbacks.

## Current algorithms

- Singular value decomposition (`cuda_svd()`).
- Principal component analysis (`cuda_pca()`).
- Euclidean and cosine distances (`cuda_distance()`).
- k-nearest neighbours (`cuda_knn()`).
- Lloyd k-means with a GPU-capable distance step (`cuda_kmeans()`).

## Installation

```r
# install.packages("pak")
pak::pak("cudaverse/cudalearnr")
```

## Example

```r
library(cudalearnr)

x <- scale(iris[, 1:4])

pca <- cuda_pca(x, n_components = 2)
knn <- cuda_knn(pca$x, k = 10, batch_size = 128)
clusters <- cuda_kmeans(pca$x, centers = 3, seed = 1)

pca
knn
clusters
```

## Backend semantics

When a CUDA-enabled R torch installation is available, SVD/PCA, pairwise
distances, and kNN distance blocks execute through libtorch on the GPU. k-means
uses GPU distance calculations while updating centres in R. Without CUDA, the
same APIs use base R and `stats`.

## Exact kNN without a full distance matrix

`cuda_knn()` compares every observation with every other observation, but works
on query batches:

```r
neighbors <- cuda_knn(
  x,
  k = 15,
  metric = "cosine",
  batch_size = 256
)
```

At most `min(batch_size, nrow(x)) * nrow(x)` distances are held at once,
instead of an `nrow(x) * nrow(x)` distance matrix. The returned `index` and
`distance` matrices require only `nrow(x) * k` entries. A smaller batch uses
less memory; a larger batch can improve throughput. This is still an exact
quadratic-time algorithm.

Each observation excludes itself. When multiple candidates have exactly the
same distance, the candidate with the smaller input row number is selected
first, so changing `batch_size` does not change the result.

On CUDA, the validated input is uploaded once, distance blocks are computed
with torch, and each block is returned to the CPU for deterministic ordering.
Neighbour selection is therefore not yet fully device-resident. Use
`cuda_distance()` only when the complete dense pairwise matrix is actually
needed.

The first release prioritizes correctness and a stable R API. Approximate
neighbour search and fully device-resident neighbour selection remain future
milestones.

For installation, device verification, memory advice, and common failures, see
the cudaverse
[GPU setup and troubleshooting guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li
