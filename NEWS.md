# cudalearnr 0.1.2

- SVD, PCA, distance, kNN, and k-means results now preserve observation and
  feature identifiers consistently on CPU and CUDA backends. PCA and SVD
  components receive stable names.
- `cuda_pca()` now rejects non-logical `center` and `scale.` values consistently
  before backend dispatch.

# cudalearnr 0.1.1

- `cuda_knn()` now performs exact neighbour search in configurable query
  batches, avoiding allocation of the complete pairwise distance matrix.
  Self-neighbours are excluded explicitly and equal distances are resolved by
  input row number, independently of batch size.
- CUDA kNN uploads the validated input once and computes distance blocks with
  torch while retaining deterministic CPU neighbour ordering.
- `cuda_kmeans()` now recomputes assignments against its returned centers and
  derives `withinss` from that final assignment, including when `iter.max` is
  reached.
- Cosine distance now rejects zero rows consistently before CPU/CUDA dispatch
  and uses scale-first normalization for extreme finite values.
