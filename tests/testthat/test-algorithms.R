test_matrix <- function() {
  matrix(c(
    1, 2, 3,
    2, 3, 4,
    4, 2, 1,
    5, 3, 2,
    8, 7, 9,
    9, 8, 8
  ), ncol = 3, byrow = TRUE)
}

test_that("SVD reconstructs the input", {
  x <- test_matrix()
  fit <- cuda_svd(x, device = "cpu")
  reconstructed <- fit$u %*% diag(fit$d) %*% t(fit$v)

  expect_equal(reconstructed, x, tolerance = 1e-10)
  expect_identical(fit$device, "cpu")
})

test_that("PCA matches prcomp variance and dimensions", {
  x <- test_matrix()
  fit <- cuda_pca(x, n_components = 2, device = "cpu")
  expected <- stats::prcomp(x, rank. = 2)

  expect_s3_class(fit, "cuda_pca")
  expect_identical(dim(fit$x), c(6L, 2L))
  expect_equal(fit$sdev, expected$sdev[1:2])
})

test_that("distance supports Euclidean and cosine metrics", {
  x <- test_matrix()
  euclidean <- cuda_distance(x, device = "cpu")
  cosine <- cuda_distance(x, metric = "cosine", device = "cpu")

  expect_equal(euclidean, as.matrix(stats::dist(x)), tolerance = 1e-10,
               ignore_attr = TRUE)
  expect_equal(diag(cosine), rep(0, nrow(x)), tolerance = 1e-10)
})

test_that("k-nearest neighbours exclude each observation", {
  fit <- cuda_knn(test_matrix(), k = 2, device = "cpu")

  expect_identical(dim(fit$index), c(6L, 2L))
  expect_false(any(fit$index == row(fit$index)))
  expect_true(all(fit$distance >= 0))
})

test_that("k-means returns coherent clusters", {
  set.seed(1)
  x <- rbind(
    matrix(rnorm(40, 0, 0.2), 20, 2),
    matrix(rnorm(40, 5, 0.2), 20, 2)
  )
  fit <- cuda_kmeans(x, 2, seed = 1, device = "cpu")

  expect_s3_class(fit, "cuda_kmeans")
  expect_length(fit$cluster, 40)
  expect_identical(dim(fit$centers), c(2L, 2L))
  expect_true(all(fit$cluster %in% 1:2))
})

test_that("seeded k-means does not mutate the caller RNG state", {
  x <- test_matrix()
  set.seed(99)
  before <- .Random.seed

  cuda_kmeans(x, 2, seed = 1, device = "cpu")

  expect_identical(.Random.seed, before)
  expect_error(
    cuda_kmeans(x, 2, seed = 1.5, device = "cpu"),
    "whole number"
  )
})

test_that("invalid algorithm inputs fail clearly", {
  x <- test_matrix()
  expect_error(cuda_pca(x, n_components = 10, device = "cpu"), "between")
  expect_error(cuda_knn(x, k = nrow(x), device = "cpu"), "nrow")
  expect_error(
    cuda_distance(x, matrix(1:8, 4, 2), device = "cpu"),
    "same number of columns"
  )
})
