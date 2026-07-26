# cudalearnr 0.2.0

- Fitted `cuda_pca` and `cuda_kmeans` objects now support standard
  `predict()` workflows for new observations. Prediction aligns named features
  safely, accepts a single observation, preserves identifiers, supports
  explicit CPU/CUDA selection, and records the projection or assignment stages.
- Prediction with `device = "model"` now records an inherited model-device
  decision instead of misreporting an explicit CPU or CUDA request. Stored
  training scores and assignments are validated before retrieval and are
  documented separately from recomputed predictions.
- `cuda_distance()` now accepts one-row inputs, enabling single-observation
  scoring without special casing.
- CPU Euclidean distance, exact kNN, and k-means now combine a translated,
  globally scaled vectorized path with targeted direct, scale-first difference
  norms. This avoids catastrophic cancellation for nearby values with large
  common offsets and avoids avoidable overflow or underflow without giving up
  the fast common case.
- SVD, PCA, distance, exact kNN, and k-means results now expose the shared
  stage-level provenance schema, their original device request, concrete
  backend, effective parameters, source metadata, and aggregate compute device.
- CUDA kNN and k-means now report their real hybrid execution: distance kernels
  run on CUDA while deterministic neighbour ordering, assignment, and centroid
  updates run on CPU.
- Added a concise `cuda_svd` print method and made all algorithm print methods
  disclose backend and aggregate compute device.
- Explicit CUDA unavailability now uses the shared classed condition; automatic
  CPU selection retains its structured reason.

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
