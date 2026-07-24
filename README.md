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
knn <- cuda_knn(pca$x, k = 10)
clusters <- cuda_kmeans(pca$x, centers = 3, seed = 1)

pca
knn
clusters
```

## Backend semantics

When a CUDA-enabled R torch installation is available, SVD/PCA and pairwise
distance calculations execute through libtorch on the GPU. kNN consumes the GPU
distance matrix, and k-means uses GPU distance calculations while updating
centres in R. Without CUDA, the same APIs use base R and `stats`.

The first release prioritizes correctness and a stable R API. Larger-than-memory
batching and fully device-resident neighbour selection are future milestones.

## License

MIT © Yaoxiang Li
