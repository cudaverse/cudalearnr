.provenance_matrix <- function() {
  matrix(
    c(
      1, 2, 3,
      2, 3, 4,
      4, 2, 1,
      5, 3, 2,
      8, 7, 9,
      9, 8, 8
    ),
    ncol = 3,
    byrow = TRUE
  )
}

test_that("all numerical result types expose one provenance schema", {
  x <- .provenance_matrix()

  svd_fit <- cuda_svd(x, device = "cpu")
  expect_identical(cuda_provenance(svd_fit)$stage, "decomposition")
  expect_identical(svd_fit$compute_device, "cpu")
  expect_identical(svd_fit$backend, "base")
  expect_identical(svd_fit$requested_device, "cpu")

  pca_fit <- cuda_pca(x, n_components = 2, device = "cpu")
  expect_identical(
    cuda_provenance(pca_fit)$stage,
    c("preprocessing", "decomposition")
  )
  expect_identical(pca_fit$compute_device, "cpu")
  expect_identical(pca_fit$backend, "stats")
  expect_identical(
    pca_fit$parameters,
    list(n_components = 2L, center = TRUE, scale = FALSE)
  )

  distance <- cuda_distance(x, device = "cpu")
  distance_provenance <- cuda_provenance(distance)
  expect_identical(distance_provenance$stage, "distance")
  expect_identical(distance_provenance$output_device, "cpu")
  expect_identical(attr(distance, "compute_device"), "cpu")
  expect_identical(attr(distance, "requested_device"), "cpu")

  knn <- cuda_knn(x, k = 2, batch_size = 3, device = "cpu")
  expect_identical(
    cuda_provenance(knn)$stage,
    c("distance", "neighbor_selection")
  )
  expect_identical(knn$compute_device, "cpu")
  expect_identical(knn$parameters$batch_size, 3L)

  kmeans <- cuda_kmeans(
    x,
    centers = x[c(1L, 4L), , drop = FALSE],
    device = "cpu"
  )
  expect_identical(
    cuda_provenance(kmeans)$stage,
    c("initialization", "distance", "assignment", "center_update")
  )
  expect_identical(kmeans$compute_device, "cpu")
  expect_identical(kmeans$backend, "base")
})

test_that("automatic fallback is visible and explicit CUDA remains strict", {
  unavailable <- structure(
    list(
      torch_installed = FALSE,
      torch_version = NA_character_,
      cuda_available = FALSE,
      cuda_device_count = 0L,
      reason = "torch_not_installed",
      detection_error = NULL
    ),
    class = "cuda_diagnostics"
  )
  testthat::local_mocked_bindings(
    cuda_diagnostics = function() unavailable,
    .package = "cudatensr"
  )

  fit <- cuda_knn(.provenance_matrix(), k = 2, device = "auto")
  provenance <- cuda_provenance(fit)
  expect_identical(fit$requested_device, "auto")
  expect_identical(provenance$requested_device[[1L]], "auto")
  expect_identical(provenance$selection_reason[[1L]], "torch_not_installed")
  expect_true(provenance$fallback[[1L]])
  expect_identical(provenance$device, c("cpu", "cpu"))
  expect_s3_class(
    tryCatch(
      cuda_pca(.provenance_matrix(), device = "cuda"),
      error = identity
    ),
    "cudaverse_cuda_unavailable"
  )
})

test_that("print methods disclose hybrid-aware compute metadata", {
  svd_fit <- cuda_svd(.provenance_matrix(), device = "cpu")
  pca_fit <- cuda_pca(.provenance_matrix(), device = "cpu")
  knn_fit <- cuda_knn(.provenance_matrix(), k = 2, device = "cpu")
  kmeans_fit <- cuda_kmeans(.provenance_matrix(), centers = 2, seed = 1,
                            device = "cpu")

  expect_output(print(svd_fit), "compute=cpu")
  expect_output(print(pca_fit), "compute=cpu")
  expect_output(print(knn_fit), "distance_device=cpu compute=cpu")
  expect_output(print(kmeans_fit), "distance_device=cpu compute=cpu")
})
