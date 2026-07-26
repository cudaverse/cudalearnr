# Backend provenance and CUDA diagnostics

`cudalearnr` uses one API on CPU and CUDA, but it does not treat a
CUDA-aware function name as proof that every stage ran on a GPU. Version
0.2.0 records the device request and actual execution of every stage.

This tutorial first runs entirely on the CPU. The CUDA section is
optional and executes only when diagnostics prove that a usable device
exists.

## A reproducible CPU workflow

Use `device = "cpu"` when you need a portable, guaranteed CPU run. This
path does not inspect or initialize the optional CUDA runtime.

``` r

library(cudalearnr)

x <- scale(as.matrix(iris[, 1:4]))
rownames(x) <- paste0("flower_", seq_len(nrow(x)))

decomposition <- cuda_svd(
  x,
  nu = 2,
  nv = 2,
  device = "cpu"
)
pca <- cuda_pca(
  x,
  n_components = 2,
  device = "cpu"
)
distance <- cuda_distance(
  pca$x,
  device = "cpu"
)
neighbors <- cuda_knn(
  pca$x,
  k = 5,
  batch_size = 32,
  device = "cpu"
)
clusters <- cuda_kmeans(
  pca$x,
  centers = 3,
  seed = 1,
  device = "cpu"
)
```

The numerical results remain ordinary R objects. Provenance is separate,
stable metadata:

``` r

cuda_provenance(decomposition)
#> <cuda_provenance schema=cudaverse-stage/1 stages=1 compute=cpu>
#>          stage requested_device device backend selection_reason fallback
#>  decomposition              cpu    cpu    base     explicit_cpu    FALSE
#>  output_device
#>            cpu
cuda_provenance(pca)
#> <cuda_provenance schema=cudaverse-stage/1 stages=2 compute=cpu>
#>          stage requested_device device backend selection_reason fallback
#>  preprocessing              cpu    cpu   stats     explicit_cpu    FALSE
#>  decomposition              cpu    cpu   stats     explicit_cpu    FALSE
#>  output_device
#>            cpu
#>            cpu
cuda_provenance(distance)
#> <cuda_provenance schema=cudaverse-stage/1 stages=1 compute=cpu>
#>     stage requested_device device backend selection_reason fallback
#>  distance              cpu    cpu    base     explicit_cpu    FALSE
#>  output_device
#>            cpu
cuda_provenance(neighbors)
#> <cuda_provenance schema=cudaverse-stage/1 stages=2 compute=cpu>
#>               stage requested_device device backend   selection_reason fallback
#>            distance              cpu    cpu    base       explicit_cpu    FALSE
#>  neighbor_selection        fixed-cpu    cpu    base algorithm_cpu_only    FALSE
#>  output_device
#>            cpu
#>            cpu
cuda_provenance(clusters)
#> <cuda_provenance schema=cudaverse-stage/1 stages=4 compute=cpu>
#>           stage requested_device device backend   selection_reason fallback
#>  initialization        fixed-cpu    cpu    base algorithm_cpu_only    FALSE
#>        distance              cpu    cpu    base       explicit_cpu    FALSE
#>      assignment        fixed-cpu    cpu    base algorithm_cpu_only    FALSE
#>   center_update        fixed-cpu    cpu    base algorithm_cpu_only    FALSE
#>  output_device
#>            cpu
#>            cpu
#>            cpu
#>            cpu
```

[`cuda_provenance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_provenance.md)
returns one row per stage. Its columns have deliberately different
meanings:

| Column | Meaning |
|----|----|
| `requested_device` | What selected the stage: `"cpu"`, `"cuda"`, `"auto"`, a fixed CPU algorithm, or an inherited choice. |
| `device` | Where that stage actually computed: `"cpu"` or `"cuda"`. |
| `backend` | The concrete implementation, such as `base`, `stats`, or `torch`. |
| `selection_reason` | Why the device was selected, including explicit requests, fixed CPU work, input transfers, or CUDA unavailability. |
| `fallback` | `TRUE` only when an `"auto"` request selected CPU because CUDA was unavailable. |
| `output_device` | Where the stage result was materialized. Returned R matrices and lists normally live on CPU even after CUDA computation. |

The table has a `compute_device` attribute. It is `"cpu"` when all
stages are CPU, `"cuda"` when all compute stages are CUDA, and
`"hybrid"` when both occur:

``` r

pca_provenance <- cuda_provenance(pca)
attr(pca_provenance, "compute_device")
#> [1] "cpu"
```

An output on CPU is not evidence of a fallback. CUDA kernels commonly
return an R matrix on CPU. Conversely, `fallback = TRUE` records a
device-selection decision; it is not used to hide a CUDA execution
error.

## Stage-by-stage backend semantics

The following table describes successful CPU and explicit CUDA requests.

| Function | Stage | CPU backend | CUDA backend | Output |
|----|----|----|----|----|
| [`cuda_svd()`](https://cudaverse.github.io/cudalearnr/reference/cuda_svd.md) | `decomposition` | `base` | `torch` | CPU R list |
| [`cuda_pca()`](https://cudaverse.github.io/cudalearnr/reference/cuda_pca.md) | `preprocessing` | `stats` | `torch` | selected device |
| [`cuda_pca()`](https://cudaverse.github.io/cudalearnr/reference/cuda_pca.md) | `decomposition` | `stats` | `torch` | CPU R list |
| [`cuda_distance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_distance.md) | `distance` | `base` | `torch` | CPU dense matrix |
| [`cuda_knn()`](https://cudaverse.github.io/cudalearnr/reference/cuda_knn.md) | `distance` | `base` | `torch` | CPU distance block |
| [`cuda_knn()`](https://cudaverse.github.io/cudalearnr/reference/cuda_knn.md) | `neighbor_selection` | `base` | `base` | CPU R list |
| [`cuda_kmeans()`](https://cudaverse.github.io/cudalearnr/reference/cuda_kmeans.md) | `initialization` | `base` | `base` | CPU |
| [`cuda_kmeans()`](https://cudaverse.github.io/cudalearnr/reference/cuda_kmeans.md) | `distance` | `base` | `torch` | CPU distance matrix |
| [`cuda_kmeans()`](https://cudaverse.github.io/cudalearnr/reference/cuda_kmeans.md) | `assignment` | `base` | `base` | CPU |
| [`cuda_kmeans()`](https://cudaverse.github.io/cudalearnr/reference/cuda_kmeans.md) | `center_update` | `base` | `base` | CPU R list |

Thus SVD, PCA, and distance have a CUDA aggregate when explicitly run on
CUDA. kNN and k-means are intentionally hybrid: CUDA accelerates
distance calculation, while deterministic selection, assignments, and
centre updates remain in R on the CPU.

If a CUDA `cudatensor` must first be materialized as an R matrix,
provenance also includes an `input_materialization` stage. This makes a
device transfer visible instead of presenting the subsequent kernel as
the whole workflow.

## Runtime diagnostics and device requests

Diagnostics are non-destructive: they inspect the optional torch runtime
but never install or download it.

``` r

diagnostics <- cudatensr::cuda_diagnostics()
diagnostics
#> <cuda_diagnostics available=FALSE devices=0 torch=0.17.0 reason=backend_error>
```

Device requests follow these rules:

- `"cpu"` is explicit and always uses the portable backend.
- `"cuda"` is strict. If CUDA is unavailable, the call raises a
  `cudaverse_cuda_unavailable` error; it does not fall back.
- `"auto"` uses CUDA when diagnostics report a usable device. Otherwise
  it selects CPU and records `fallback = TRUE` plus a stable reason such
  as `torch_not_installed` or `cuda_unavailable`.
- Once CUDA has been selected, an execution error is reported as an
  error. It is not retried silently on CPU.

## Optional CUDA run

The same data can be used for a small CUDA smoke test. The conditional
keeps the vignette runnable on machines without a GPU.

``` r

if (isTRUE(diagnostics$cuda_available)) {
  cuda_pca_fit <- cuda_pca(
    x,
    n_components = 2,
    device = "cuda"
  )
  cuda_neighbors <- cuda_knn(
    cuda_pca_fit$x,
    k = 5,
    batch_size = 32,
    device = "cuda"
  )

  cuda_provenance(cuda_pca_fit)
  cuda_provenance(cuda_neighbors)
} else {
  message("CUDA example skipped: ", diagnostics$reason)
}
#> CUDA example skipped: backend_error
```

For kNN, a CUDA distance stage followed by CPU neighbour selection
produces `compute_device = "hybrid"`. This is expected and is stronger
evidence than checking only the legacy `$device` field.

## Dense-memory boundaries

[`cuda_pca()`](https://cudaverse.github.io/cudalearnr/reference/cuda_pca.md)
accepts a dense observation-by-feature matrix. A double matrix alone
uses roughly `8 * nrow(x) * ncol(x)` bytes, and centring, scaling, SVD,
and CPU/GPU copies require additional working memory. Reduce the number
of observations or features before PCA when the dense representation is
too large.

[`cuda_distance()`](https://cudaverse.github.io/cudalearnr/reference/cuda_distance.md)
intentionally returns the complete dense pairwise matrix, requiring
about `8 * nrow(x) * nrow(y)` bytes for doubles.

[`cuda_knn()`](https://cudaverse.github.io/cudalearnr/reference/cuda_knn.md)
is exact and quadratic in time, but `batch_size` bounds its resident
distance block to approximately `8 * min(batch_size, nrow(x)) * nrow(x)`
bytes. Its returned distance matrix uses about `8 * nrow(x) * k` bytes,
in addition to the integer index matrix. Lower `batch_size` to reduce
peak memory; it does not change selected neighbours.

## Hardware-enforced parity gate

Ordinary package checks use portable CPU paths and may conditionally
skip GPU examples. That is not accepted as CUDA coverage. The cudaverse
hardware workflow runs on a self-hosted runner labelled `cuda`, sets
`CUDAVERSE_REQUIRE_CUDA=true`, proves NVIDIA visibility with
`nvidia-smi`, and requires both
[`torch::cuda_is_available()`](https://torch.mlverse.org/docs/reference/cuda_is_available.html)
and a positive CUDA device count. It then compares the public CPU and
CUDA paths and verifies their provenance.

Projects reusing the same convention can turn the environment marker
into a hard local gate:

``` r

require_cuda <- identical(
  tolower(Sys.getenv("CUDAVERSE_REQUIRE_CUDA", unset = "false")),
  "true"
)
if (require_cuda && !isTRUE(diagnostics$cuda_available)) {
  stop("CUDA hardware is required, but no usable device was detected.")
}
```

The package-level CUDA workflow runs manually, or automatically when the
repository variable `CUDAVERSE_NVIDIA_CI` is set to `enabled`. A missing
or broken GPU on that required job is a failure, never a successful
skip.
