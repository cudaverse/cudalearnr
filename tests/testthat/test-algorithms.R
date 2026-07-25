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
  expect_equal(unname(fit$sdev), expected$sdev[1:2])
  expect_identical(names(fit$sdev), c("PC1", "PC2"))
})

test_that("algorithms preserve observation and feature identifiers", {
  x <- test_matrix()
  rownames(x) <- paste0("cell_", seq_len(nrow(x)))
  colnames(x) <- paste0("feature_", seq_len(ncol(x)))

  decomposition <- cuda_svd(x, nu = 2, nv = 2, device = "cpu")
  expect_identical(rownames(decomposition$u), rownames(x))
  expect_identical(rownames(decomposition$v), colnames(x))
  expect_identical(colnames(decomposition$u), c("SVD1", "SVD2"))

  pca <- cuda_pca(x, n_components = 2, device = "cpu")
  expect_identical(rownames(pca$x), rownames(x))
  expect_identical(rownames(pca$rotation), colnames(x))
  expect_identical(colnames(pca$x), c("PC1", "PC2"))
  expect_identical(names(pca$center), colnames(x))

  distance <- cuda_distance(x, device = "cpu")
  expect_identical(dimnames(distance), list(rownames(x), rownames(x)))

  neighbors <- cuda_knn(x, k = 2, device = "cpu", batch_size = 2)
  expect_identical(rownames(neighbors$index), rownames(x))
  neighbor_labels <- matrix(
    rownames(neighbors$index)[as.vector(neighbors$index)],
    nrow = nrow(neighbors$index),
    dimnames = dimnames(neighbors$index)
  )
  expect_identical(dim(neighbor_labels), dim(neighbors$index))
  expect_true(all(neighbor_labels %in% rownames(x)))
  expect_identical(dimnames(neighbors$distance), dimnames(neighbors$index))

  clusters <- cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
  expect_identical(names(clusters$cluster), rownames(x))
  expect_identical(colnames(clusters$centers), colnames(x))
  expect_identical(rownames(clusters$centers), c("cluster_1", "cluster_2"))
  expect_identical(names(clusters$withinss), rownames(clusters$centers))
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

  expect_named(fit, c("index", "distance", "metric", "device"))
  expect_identical(dim(fit$index), c(6L, 2L))
  expect_false(any(fit$index == row(fit$index)))
  expect_true(all(fit$distance >= 0))
})

test_that("batched exact neighbours match full pairwise distances", {
  x <- test_matrix()
  reference_index <- seq_len(nrow(x))

  for (metric in c("euclidean", "cosine")) {
    full_distance <- cuda_distance(x, metric = metric, device = "cpu")
    expected_index <- vapply(
      reference_index,
      function(i) {
        candidates <- reference_index[-i]
        ordering <- order(
          full_distance[i, candidates],
          candidates,
          method = "radix"
        )
        candidates[ordering[1:2]]
      },
      integer(2)
    )
    expected_index <- t(expected_index)
    expected_distance <- matrix(
      full_distance[cbind(
        rep(reference_index, each = 2L),
        as.vector(t(expected_index))
      )],
      nrow = nrow(x),
      byrow = TRUE
    )

    for (batch_size in c(1L, 2L, 100L)) {
      fit <- cuda_knn(
        x,
        k = 2,
        metric = metric,
        device = "cpu",
        batch_size = batch_size
      )

      expect_identical(fit$index, expected_index)
      expect_equal(fit$distance, expected_distance, tolerance = 1e-12)
      expect_identical(fit$metric, metric)
      expect_identical(fit$device, "cpu")
    }
  }
})

test_that("distance blocks never exceed the requested query batch", {
  x <- test_matrix()
  state <- cudalearnr:::.knn_distance_state(
    x,
    metric = "euclidean",
    device = "cpu"
  )
  blocks <- list(1:2, 3:4, 5:6)
  distance <- lapply(
    blocks,
    function(rows) cudalearnr:::.knn_distance_block(state, rows)
  )

  expect_true(all(vapply(distance, nrow, integer(1)) <= 2L))
  expect_true(all(vapply(distance, ncol, integer(1)) == nrow(x)))
  expect_equal(
    do.call(rbind, distance),
    cuda_distance(x, device = "cpu"),
    tolerance = 1e-12,
    ignore_attr = TRUE
  )
})

test_that("kNN ties and self exclusion are deterministic", {
  tied <- matrix(c(0, 2, 4), ncol = 1)
  fit <- cuda_knn(
    tied,
    k = 1,
    device = "cpu",
    batch_size = 1
  )

  expect_identical(as.vector(fit$index), c(2L, 1L, 2L))
  expect_equal(as.vector(fit$distance), c(2, 2, 2))

  duplicated <- matrix(c(0, 0, 1), ncol = 1)
  duplicate_fit <- cuda_knn(
    duplicated,
    k = 1,
    device = "cpu",
    batch_size = 2
  )

  expect_identical(as.vector(duplicate_fit$index), c(2L, 1L, 1L))
  expect_false(any(duplicate_fit$index == row(duplicate_fit$index)))
})

test_that("kNN validates batch sizes and cosine rows clearly", {
  x <- test_matrix()

  for (batch_size in list(0, -1, 1.5, Inf, NA_real_, numeric())) {
    expect_error(
      cuda_knn(x, k = 2, batch_size = batch_size, device = "cpu"),
      "positive whole number"
    )
  }
  expect_error(
    cuda_knn(x, k = Inf, device = "cpu"),
    "between 1 and nrow"
  )

  zero <- rbind(c(0, 0), c(1, 0), c(0, 1))
  expect_error(
    cuda_knn(
      zero,
      k = 1,
      metric = "cosine",
      device = "cuda",
      batch_size = 1
    ),
    "zero-length rows"
  )
})

test_that("CUDA batched neighbours agree with CPU reference", {
  skip_if_not(cudatensr::cuda_available())
  x <- test_matrix()
  cpu <- cuda_knn(x, k = 2, device = "cpu", batch_size = 2)
  gpu <- cuda_knn(x, k = 2, device = "cuda", batch_size = 2)

  expect_identical(gpu$index, cpu$index)
  expect_equal(gpu$distance, cpu$distance, tolerance = 1e-8)
  expect_identical(gpu$device, "cuda")
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
  expect_error(cuda_pca(x, center = NA, device = "cpu"), "TRUE or FALSE")
  expect_error(cuda_pca(x, scale. = 1, device = "cpu"), "TRUE or FALSE")
})
