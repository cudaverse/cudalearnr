# Pairwise distances with an optional CUDA backend

Pairwise distances with an optional CUDA backend

## Usage

``` r
cuda_distance(
  x,
  y = NULL,
  metric = c("euclidean", "cosine"),
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x, y:

  Numeric matrices with observations in rows. When `y` is `NULL`,
  computes all pairwise distances within `x`.

- metric:

  `"euclidean"` or `"cosine"`.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A dense numeric distance matrix with a `device` attribute.

## Examples

``` r
cuda_distance(matrix(1:12, 4, 3), device = "cpu")
#>          [,1]     [,2]     [,3]     [,4]
#> [1,] 0.000000 1.732051 3.464102 5.196152
#> [2,] 1.732051 0.000000 1.732051 3.464102
#> [3,] 3.464102 1.732051 0.000000 1.732051
#> [4,] 5.196152 3.464102 1.732051 0.000000
#> attr(,"device")
#> [1] "cpu"
```
