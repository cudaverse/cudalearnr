# Changelog

## cudalearnr 0.2.0

- SVD, PCA, distance, exact kNN, and k-means results now expose the
  shared stage-level provenance schema, their original device request,
  concrete backend, effective parameters, source metadata, and aggregate
  compute device.
- CUDA kNN and k-means now report their real hybrid execution: distance
  kernels run on CUDA while deterministic neighbour ordering,
  assignment, and centroid updates run on CPU.
- Added a concise `cuda_svd` print method and made all algorithm print
  methods disclose backend and aggregate compute device.
- Explicit CUDA unavailability now uses the shared classed condition;
  automatic CPU selection retains its structured reason.

## cudalearnr 0.1.2

- SVD, PCA, distance, kNN, and k-means results now preserve observation
  and feature identifiers consistently on CPU and CUDA backends. PCA and
  SVD components receive stable names.
- [`cuda_pca()`](https://cudaverse.github.io/cudalearnr/reference/cuda_pca.md)
  now rejects non-logical `center` and `scale.` values consistently
  before backend dispatch.

## cudalearnr 0.1.1

- [`cuda_knn()`](https://cudaverse.github.io/cudalearnr/reference/cuda_knn.md)
  now performs exact neighbour search in configurable query batches,
  avoiding allocation of the complete pairwise distance matrix.
  Self-neighbours are excluded explicitly and equal distances are
  resolved by input row number, independently of batch size.
- CUDA kNN uploads the validated input once and computes distance blocks
  with torch while retaining deterministic CPU neighbour ordering.
- [`cuda_kmeans()`](https://cudaverse.github.io/cudalearnr/reference/cuda_kmeans.md)
  now recomputes assignments against its returned centers and derives
  `withinss` from that final assignment, including when `iter.max` is
  reached.
- Cosine distance now rejects zero rows consistently before CPU/CUDA
  dispatch and uses scale-first normalization for extreme finite values.
