# GPU-aware singular value decomposition

GPU-aware singular value decomposition

## Usage

``` r
cuda_svd(
  x,
  nu = min(nrow(x), ncol(x)),
  nv = min(nrow(x), ncol(x)),
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x:

  A finite numeric matrix or `cudatensor`.

- nu, nv:

  Number of left and right singular vectors to return.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A list with `d`, `u`, `v`, and the actual `device`. Matrix row and
column names are retained on the corresponding singular vectors.

## Examples

``` r
cuda_svd(matrix(rnorm(30), 10, 3), device = "cpu")
#> $d
#>     SVD1     SVD2     SVD3 
#> 2.869171 1.661290 1.475526 
#> 
#> $u
#>              SVD1        SVD2        SVD3
#>  [1,] -0.04150433  0.22856936 -0.48180687
#>  [2,] -0.39082093 -0.61741638  0.15778587
#>  [3,]  0.42570053  0.06499382  0.63448576
#>  [4,] -0.52175072  0.43293582  0.03641741
#>  [5,]  0.05996450 -0.03078349 -0.22963435
#>  [6,] -0.57675901 -0.03777884  0.35893887
#>  [7,] -0.13318252  0.09639454 -0.11080516
#>  [8,] -0.19352735  0.13419311 -0.01327715
#>  [9,] -0.02226106  0.55864409  0.35131843
#> [10,] -0.01260843  0.18203753 -0.14703610
#> 
#> $v
#>            SVD1        SVD2       SVD3
#> [1,]  0.5081674  0.08897574  0.8566500
#> [2,] -0.3074937 -0.91035209  0.2769597
#> [3,]  0.8044958 -0.40415639 -0.4352518
#> 
#> $device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_svd"
```
