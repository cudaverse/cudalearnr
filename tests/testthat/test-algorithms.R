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

test_that("cosine distance rejects zero rows before backend dispatch", {
  zero_x <- rbind(c(0, 0), c(1, 0))
  zero_y <- rbind(c(1, 0), c(0, 0))
  valid <- rbind(c(1, 0), c(0, 1))

  expect_error(
    cuda_distance(zero_x, metric = "cosine", device = "cpu"),
    "zero-length rows"
  )
  expect_error(
    cuda_distance(valid, zero_y, metric = "cosine", device = "cpu"),
    "zero-length rows"
  )
  expect_error(
    cuda_distance(zero_x, metric = "cosine", device = "cuda"),
    "zero-length rows"
  )
})

test_that("cosine normalization is stable across extreme finite scales", {
  x <- rbind(
    c(1e300, 1e300),
    c(1e-300, 0)
  )
  distance <- cuda_distance(x, metric = "cosine", device = "cpu")

  expect_true(all(is.finite(distance)))
  expect_equal(diag(distance), c(0, 0), tolerance = 1e-12)
  expect_equal(
    distance[1, 2],
    1 - 1 / sqrt(2),
    tolerance = 1e-12
  )
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

test_that("k-means final assignments and sums match returned centers", {
  set.seed(1)
  x <- matrix(rnorm(60), 30, 2)
  fit <- cuda_kmeans(
    x,
    centers = 3,
    iter.max = 1,
    seed = 1001,
    device = "cpu"
  )
  distances <- cuda_distance(x, fit$centers, device = "cpu")
  expected_cluster <- max.col(-distances, ties.method = "first")
  expected_withinss <- vapply(
    seq_len(nrow(fit$centers)),
    function(group) {
      members <- which(expected_cluster == group)
      sum(distances[cbind(members, rep.int(group, length(members)))]^2)
    },
    numeric(1)
  )

  expect_identical(fit$cluster, expected_cluster)
  expect_equal(fit$withinss, expected_withinss)
  expect_equal(fit$tot.withinss, sum(expected_withinss))
  expect_identical(fit$iter, 1L)
  expect_false(fit$converged)
})

test_that("k-means handles ties and empty clusters deterministically", {
  tied <- matrix(c(-2, 0, 1), ncol = 1)
  tied_fit <- cuda_kmeans(
    tied,
    centers = matrix(c(-1, 1), ncol = 1),
    device = "cpu"
  )

  expect_identical(tied_fit$cluster, c(1L, 1L, 2L))

  duplicated <- matrix(1, nrow = 3, ncol = 1)
  empty_fit <- cuda_kmeans(
    duplicated,
    centers = matrix(c(1, 1), ncol = 1),
    device = "cpu"
  )

  expect_identical(empty_fit$cluster, rep(1L, 3))
  expect_equal(empty_fit$centers, matrix(c(1, 1), ncol = 1))
  expect_identical(empty_fit$withinss, c(0, 0))
  expect_true(empty_fit$converged)
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
