# cudalearnr

`cudalearnr` is the reusable numerical algorithm layer of the
**cudaverse**. It provides GPU-aware implementations with explicit
device reporting and portable CPU fallbacks.

## Current algorithms

- Singular value decomposition
  ([`cuda_svd()`](https://cudaverse.github.io/cudalearnr/reference/cuda_svd.md)).
- Principal component analysis
  ([`cuda_pca()`](https://cudaverse.github.io/cudalearnr/reference/cuda_pca.md)).
- Euclidean and cosine distances
  ([`cuda_distance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_distance.md)).
- k-nearest neighbours
  ([`cuda_knn()`](https://cudaverse.github.io/cudalearnr/reference/cuda_knn.md)).
- Lloyd k-means with a GPU-capable distance step
  ([`cuda_kmeans()`](https://cudaverse.github.io/cudalearnr/reference/cuda_kmeans.md)).
- Post-fit PCA projection and k-means assignment through standard
  [`predict()`](https://rdrr.io/r/stats/predict.html) methods.

## Installation

``` r

# install.packages("pak")
pak::pak("cudaverse/cudalearnr")
```

## Example

``` r

library(cudalearnr)

features <- as.matrix(iris[, 1:4])
train <- features[1:120, ]
holdout <- features[121:150, ]

pca <- cuda_pca(train, n_components = 2)
knn <- cuda_knn(pca$x, k = 10, batch_size = 128)
clusters <- cuda_kmeans(pca$x, centers = 3, seed = 1)

holdout_pca <- predict(pca, holdout)
holdout_cluster <- predict(clusters, holdout_pca)

pca
knn
clusters
head(holdout_cluster)
```

## Predict new observations safely

[`predict()`](https://rdrr.io/r/stats/predict.html) reuses the fitted
PCA preprocessing and loadings, or assigns observations to the closest
fitted k-means centre. A fitted model’s actual device is reused by
default; set `device = "cpu"`, `"cuda"`, or `"auto"` to make a different
request. This override also lets a model fitted and saved on a CUDA
machine be used later on a CPU-only machine. Provenance records the
default as `requested_device = "inherited"` with
`selection_reason = "model_device"`; it does not mislabel that choice as
an explicit request.

Feature names are part of the prediction contract. Columns supplied in a
different order are aligned automatically, while missing, unexpected, or
unnamed features are rejected when the model was fitted with named
columns:

``` r

shuffled <- holdout[, rev(colnames(holdout)), drop = FALSE]
holdout_pca <- predict(pca, shuffled, device = "cpu")

center_distance <- predict(
  clusters,
  holdout_pca,
  type = "distance",
  device = "cpu"
)
```

Both methods accept a one-row matrix. Prediction results retain
observation and component or centre names. Recomputed results also carry
stage-level provenance, so `cuda_provenance(holdout_pca)` reports where
projection actually ran. Calling `predict(pca)` or `predict(clusters)`
without `newdata` validates and returns the stored training values
unchanged; retrieval does not create new compute provenance.

## Backend semantics

When a CUDA-enabled R torch installation is available, SVD/PCA, pairwise
distances, and kNN distance blocks execute through libtorch on the GPU.
k-means uses GPU distance calculations while updating centres in R.
Without CUDA, the same APIs use base R and `stats`.

Every result records the requested device, actual stage devices,
concrete backend, output device, and any automatic fallback. Inspect
those fields as a table with
[`cuda_provenance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_provenance.md):

``` r

cudatensr::cuda_diagnostics()
cuda_provenance(pca)
```

| Function | Device-selected work | Always-CPU work | CUDA aggregate |
|----|----|----|----|
| [`cuda_svd()`](https://cudaverse.github.io/cudalearnr/reference/cuda_svd.md) | decomposition | R result materialization | `cuda` |
| [`cuda_pca()`](https://cudaverse.github.io/cudalearnr/reference/cuda_pca.md) | preprocessing and decomposition | R result materialization | `cuda` |
| [`cuda_distance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_distance.md) | distance calculation | R result materialization | `cuda` |
| [`cuda_knn()`](https://cudaverse.github.io/cudalearnr/reference/cuda_knn.md) | distance blocks | deterministic neighbour selection | `hybrid` |
| [`cuda_kmeans()`](https://cudaverse.github.io/cudalearnr/reference/cuda_kmeans.md) | distance calculation | initialization, assignment, centre updates | `hybrid` |
| `predict(cuda_pca)` | projection | R result materialization | `cuda` |
| `predict(cuda_kmeans)` | distance calculation | closest-centre assignment | `hybrid` |

Here, “CUDA aggregate” describes a successful explicit CUDA run. An
automatic request that cannot use CUDA records a CPU fallback instead.
See [Backend provenance and CUDA
diagnostics](https://cudaverse.github.io/cudalearnr/articles/backend-provenance.html)
for a runnable CPU tutorial, the complete stage contract, memory
guidance, and the hardware-CI gate.

## Exact kNN without a full distance matrix

[`cuda_knn()`](https://cudaverse.github.io/cudalearnr/reference/cuda_knn.md)
compares every observation with every other observation, but works on
query batches:

``` r

neighbors <- cuda_knn(
  x,
  k = 15,
  metric = "cosine",
  batch_size = 256
)
```

At most `min(batch_size, nrow(x)) * nrow(x)` distances are held at once,
instead of an `nrow(x) * nrow(x)` distance matrix. The returned `index`
and `distance` matrices require only `nrow(x) * k` entries. A smaller
batch uses less memory; a larger batch can improve throughput. This is
still an exact quadratic-time algorithm.

CPU Euclidean distances use a common translation and global scaling
before a vectorized calculation. Numerically risky pairs are recomputed
from direct observation differences with a scale-first norm. This
retains nearby distances when values share a large common offset and
avoids avoidable overflow or underflow at extreme finite magnitudes. The
same implementation is used by
[`cuda_distance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_distance.md),
CPU kNN blocks, and the distance steps in CPU k-means.

Each observation excludes itself. When multiple candidates have exactly
the same distance, the candidate with the smaller input row number is
selected first, so changing `batch_size` does not change the result.

## Identifier preservation

Observation and feature names are part of the result contract. PCA
scores, distances, neighbour queries, singular vectors, and cluster
assignments retain the corresponding input names on both CPU and CUDA
backends. PCA components use stable `PC1`, `PC2`, … names, and kNN
neighbour identities can be mapped without relying on an external
reordered table:

``` r

rownames(neighbors$index)[neighbors$index]
```

On CUDA, the validated input is uploaded once, distance blocks are
computed with torch, and each block is returned to the CPU for
deterministic ordering. Neighbour selection is therefore not yet fully
device-resident. Use
[`cuda_distance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_distance.md)
only when the complete dense pairwise matrix is actually needed.

The first release prioritizes correctness and a stable R API.
Approximate neighbour search and fully device-resident neighbour
selection remain future milestones.

For installation, device verification, memory advice, and common
failures, see the cudaverse [GPU setup and troubleshooting
guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li
