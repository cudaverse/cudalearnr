# cudalearnr 0.1.1

- `cuda_kmeans()` now recomputes assignments against its returned centers and
  derives `withinss` from that final assignment, including when `iter.max` is
  reached.
- Cosine distance now rejects zero rows consistently before CPU/CUDA dispatch
  and uses scale-first normalization for extreme finite values.
