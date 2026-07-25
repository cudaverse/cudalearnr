.learn_matrix <- function(x, argument = "x", min_rows = 2L,
                          min_cols = 1L) {
  if (inherits(x, "cudatensor")) {
    x <- cudatensr::to_cpu(x)
  }
  if (!is.matrix(x) || !is.numeric(x) || nrow(x) < min_rows ||
      ncol(x) < min_cols || anyNA(x) || any(!is.finite(x))) {
    stop(
      sprintf(
        "`%s` must be a finite numeric matrix with at least %s rows and %s columns.",
        argument, min_rows, min_cols
      ),
      call. = FALSE
    )
  }
  x
}

.learn_device <- function(device) {
  device <- match.arg(device, c("auto", "cuda", "cpu"))
  if (device == "auto") {
    device <- if (cudatensr::cuda_available()) "cuda" else "cpu"
  }
  if (device == "cuda" && !cudatensr::cuda_available()) {
    stop("CUDA is unavailable; use `device = \"cpu\"`.", call. = FALSE)
  }
  device
}

.with_preserved_seed <- function(seed, code) {
  if (is.null(seed)) {
    return(force(code))
  }
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) ||
      !is.finite(seed)) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }
  integer_seed <- suppressWarnings(as.integer(seed))
  if (is.na(integer_seed) || seed != integer_seed) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(integer_seed)
  force(code)
}

.torch_matrix <- function(x) {
  torch::torch_tensor(
    x,
    dtype = torch::torch_float64(),
    device = "cuda"
  )
}

.torch_array <- function(x) {
  as.array(x$to(device = "cpu"))
}

#' GPU-aware singular value decomposition
#'
#' @param x A finite numeric matrix or `cudatensor`.
#' @param nu,nv Number of left and right singular vectors to return.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A list with `d`, `u`, `v`, and the actual `device`.
#' @export
#' @examples
#' cuda_svd(matrix(rnorm(30), 10, 3), device = "cpu")
cuda_svd <- function(x, nu = min(nrow(x), ncol(x)),
                     nv = min(nrow(x), ncol(x)),
                     device = c("auto", "cuda", "cpu")) {
  x <- .learn_matrix(x)
  device <- .learn_device(device)
  rank <- min(dim(x))
  for (value in list(nu = nu, nv = nv)) {
    if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
        value < 0 || value > rank || value != as.integer(value)) {
      stop("`nu` and `nv` must be whole numbers between zero and matrix rank.",
           call. = FALSE)
    }
  }
  nu <- as.integer(nu)
  nv <- as.integer(nv)

  if (device == "cpu") {
    result <- base::svd(x, nu = nu, nv = nv)
    return(structure(
      list(d = result$d, u = result$u, v = result$v, device = "cpu"),
      class = "cuda_svd"
    ))
  }

  result <- torch::torch_svd(.torch_matrix(x), some = TRUE)
  u <- if (nu == 0L) matrix(numeric(), nrow(x), 0L) else
    .torch_array(result[[1]][, seq_len(nu), drop = FALSE])
  v <- if (nv == 0L) matrix(numeric(), ncol(x), 0L) else
    .torch_array(result[[3]][, seq_len(nv), drop = FALSE])
  structure(
    list(
      d = as.vector(.torch_array(result[[2]])),
      u = u,
      v = v,
      device = "cuda"
    ),
    class = "cuda_svd"
  )
}

#' GPU-aware principal component analysis
#'
#' @param x A matrix with observations in rows and features in columns.
#' @param n_components Number of components to return.
#' @param center Whether to centre features.
#' @param scale. Whether to scale features to unit variance.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A `cuda_pca` object with scores in `x`, loadings in `rotation`,
#'   standard deviations, centring/scaling values, and actual device.
#' @export
#' @examples
#' fit <- cuda_pca(iris[, 1:4], n_components = 2, device = "cpu")
#' fit
cuda_pca <- function(x, n_components = 2L, center = TRUE, scale. = FALSE,
                     device = c("auto", "cuda", "cpu")) {
  x <- .learn_matrix(as.matrix(x), min_cols = 2L)
  device <- .learn_device(device)
  max_components <- min(nrow(x) - 1L, ncol(x))
  if (!is.numeric(n_components) || length(n_components) != 1L ||
      is.na(n_components) || n_components < 1 ||
      n_components > max_components ||
      n_components != as.integer(n_components)) {
    stop(
      sprintf("`n_components` must be between 1 and %s.", max_components),
      call. = FALSE
    )
  }
  if (isTRUE(scale.) && any(apply(x, 2L, stats::sd) == 0)) {
    stop("Cannot scale constant features.", call. = FALSE)
  }
  n_components <- as.integer(n_components)

  if (device == "cpu") {
    fit <- stats::prcomp(x, center = center, scale. = scale.,
                         rank. = n_components)
    return(structure(
      list(
        sdev = fit$sdev[seq_len(n_components)],
        rotation = fit$rotation[, seq_len(n_components), drop = FALSE],
        x = fit$x[, seq_len(n_components), drop = FALSE],
        center = fit$center,
        scale = fit$scale,
        device = "cpu"
      ),
      class = "cuda_pca"
    ))
  }

  tensor <- .torch_matrix(x)
  centre_values <- if (isTRUE(center)) {
    tensor$mean(dim = 1L, keepdim = TRUE)
  } else {
    torch::torch_zeros(
      c(1L, ncol(x)),
      dtype = torch::torch_float64(),
      device = "cuda"
    )
  }
  transformed <- tensor - centre_values
  scale_values <- if (isTRUE(scale.)) {
    transformed$std(dim = 1L, unbiased = TRUE, keepdim = TRUE)
  } else {
    torch::torch_ones(
      c(1L, ncol(x)),
      dtype = torch::torch_float64(),
      device = "cuda"
    )
  }
  transformed <- transformed / scale_values
  decomposition <- torch::torch_svd(transformed, some = TRUE)
  components <- seq_len(n_components)
  scores <- decomposition[[1]][, components, drop = FALSE] *
    decomposition[[2]][components]

  structure(
    list(
      sdev = as.vector(.torch_array(
        decomposition[[2]][components] / sqrt(nrow(x) - 1)
      )),
      rotation = .torch_array(
        decomposition[[3]][, components, drop = FALSE]
      ),
      x = .torch_array(scores),
      center = if (isTRUE(center)) as.vector(.torch_array(centre_values)) else FALSE,
      scale = if (isTRUE(scale.)) as.vector(.torch_array(scale_values)) else FALSE,
      device = "cuda"
    ),
    class = "cuda_pca"
  )
}

.cosine_unit_rows <- function(x, argument) {
  row_scale <- apply(abs(x), 1L, max)
  if (any(row_scale == 0)) {
    stop(
      sprintf(
        "Cosine distance is undefined for zero-length rows in `%s`.",
        argument
      ),
      call. = FALSE
    )
  }

  scaled <- x / row_scale
  scaled / sqrt(rowSums(scaled^2))
}

#' Pairwise distances with an optional CUDA backend
#'
#' @param x,y Numeric matrices with observations in rows. When `y` is `NULL`,
#'   computes all pairwise distances within `x`.
#' @param metric `"euclidean"` or `"cosine"`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A dense numeric distance matrix with a `device` attribute.
#' @export
#' @examples
#' cuda_distance(matrix(1:12, 4, 3), device = "cpu")
cuda_distance <- function(x, y = NULL,
                          metric = c("euclidean", "cosine"),
                          device = c("auto", "cuda", "cpu")) {
  x <- .learn_matrix(x)
  self <- is.null(y)
  if (self) {
    y <- x
  } else {
    y <- .learn_matrix(y, "y")
  }
  if (ncol(x) != ncol(y)) {
    stop("`x` and `y` must have the same number of columns.", call. = FALSE)
  }
  metric <- match.arg(metric)
  if (metric == "cosine") {
    x_unit <- .cosine_unit_rows(x, "x")
    y_unit <- if (self) x_unit else .cosine_unit_rows(y, "y")
  }
  device <- .learn_device(device)

  if (device == "cuda") {
    x_gpu <- .torch_matrix(if (metric == "cosine") x_unit else x)
    y_gpu <- if (self) {
      x_gpu
    } else {
      .torch_matrix(if (metric == "cosine") y_unit else y)
    }
    result <- if (metric == "euclidean") {
      torch::torch_cdist(x_gpu, y_gpu, p = 2)
    } else {
      1 - x_gpu$matmul(y_gpu$t())
    }
    distance <- .torch_array(result)
  } else if (metric == "euclidean") {
    squared <- outer(rowSums(x^2), rowSums(y^2), "+") -
      2 * tcrossprod(x, y)
    distance <- sqrt(pmax(squared, 0))
  } else {
    distance <- 1 - tcrossprod(x_unit, y_unit)
  }
  if (metric == "cosine") {
    distance <- pmin(pmax(distance, 0), 2)
  }
  attr(distance, "device") <- device
  distance
}

.knn_batch_size <- function(batch_size, n) {
  integer_batch_size <- suppressWarnings(as.integer(batch_size))
  if (!is.numeric(batch_size) || length(batch_size) != 1L ||
      is.na(batch_size) || !is.finite(batch_size) ||
      is.na(integer_batch_size) || integer_batch_size < 1L ||
      batch_size != integer_batch_size) {
    stop("`batch_size` must be one positive whole number.", call. = FALSE)
  }
  min(integer_batch_size, n)
}

.knn_distance_state <- function(x, metric, device, cosine_values = NULL) {
  values <- if (metric == "cosine") {
    if (is.null(cosine_values)) {
      .cosine_unit_rows(x, "x")
    } else {
      cosine_values
    }
  } else {
    x
  }
  storage <- if (device == "cuda") .torch_matrix(values) else NULL
  squared_norm <- if (device == "cpu" && metric == "euclidean") {
    rowSums(values^2)
  } else {
    NULL
  }
  list(
    values = values,
    storage = storage,
    squared_norm = squared_norm,
    metric = metric,
    device = device
  )
}

.knn_distance_block <- function(state, rows) {
  if (state$device == "cuda") {
    query <- state$storage[rows, , drop = FALSE]
    result <- if (state$metric == "euclidean") {
      torch::torch_cdist(query, state$storage, p = 2)
    } else {
      1 - query$matmul(state$storage$t())
    }
    distance <- .torch_array(result)
  } else if (state$metric == "euclidean") {
    squared <- outer(
      state$squared_norm[rows],
      state$squared_norm,
      "+"
    ) - 2 * tcrossprod(
      state$values[rows, , drop = FALSE],
      state$values
    )
    distance <- sqrt(pmax(squared, 0))
  } else {
    distance <- 1 - tcrossprod(
      state$values[rows, , drop = FALSE],
      state$values
    )
  }

  distance <- matrix(
    distance,
    nrow = length(rows),
    ncol = nrow(state$values)
  )
  if (state$metric == "cosine") {
    distance <- pmin(pmax(distance, 0), 2)
  }
  distance
}

#' k-nearest neighbours
#'
#' @param x Numeric matrix with observations in rows.
#' @param k Number of neighbours.
#' @param metric Exact distance metric, `"euclidean"` or `"cosine"`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @param batch_size Maximum number of query rows in each dense distance block.
#'   Larger batches may be faster but use more memory.
#' @return A `cuda_knn` list with `index` and `distance` matrices of size
#'   `nrow(x)` by `k`, followed by the selected `metric` and actual `device`.
#'   Neighbours in every row are ordered by distance and then row index.
#'
#' @details
#' Neighbours are exact: every row is compared with every other row. The
#' observation itself is always excluded. Equal distances are resolved
#' deterministically in favour of the smaller row index.
#'
#' The implementation constructs at most a
#' `min(batch_size, nrow(x))`-by-`nrow(x)` dense distance block instead of a
#' complete pairwise distance matrix. On CUDA, distance blocks are computed
#' with torch and transferred to the CPU for deterministic neighbour ordering.
#' @export
#' @examples
#' cuda_knn(
#'   matrix(rnorm(30), 10, 3),
#'   k = 3,
#'   batch_size = 4,
#'   device = "cpu"
#' )
cuda_knn <- function(x, k = 15L, metric = c("euclidean", "cosine"),
                     device = c("auto", "cuda", "cpu"),
                     batch_size = 256L) {
  x <- .learn_matrix(x)
  integer_k <- suppressWarnings(as.integer(k))
  if (!is.numeric(k) || length(k) != 1L || is.na(k) ||
      !is.finite(k) || is.na(integer_k) ||
      integer_k < 1L || integer_k >= nrow(x) || k != integer_k) {
    stop("`k` must be a whole number between 1 and nrow(x) - 1.",
         call. = FALSE)
  }
  metric <- match.arg(metric)
  cosine_values <- if (metric == "cosine") {
    .cosine_unit_rows(x, "x")
  } else {
    NULL
  }
  device <- .learn_device(device)
  batch_size <- .knn_batch_size(batch_size, nrow(x))
  state <- .knn_distance_state(x, metric, device, cosine_values)
  reference_index <- seq_len(nrow(x))
  index <- matrix(NA_integer_, nrow(x), integer_k)
  neighbour_distance <- matrix(NA_real_, nrow(x), integer_k)

  starts <- seq.int(1L, nrow(x), by = batch_size)
  for (start in starts) {
    rows <- seq.int(
      start,
      length.out = min(batch_size, nrow(x) - start + 1L)
    )
    distances <- .knn_distance_block(state, rows)
    selected <- vapply(
      seq_along(rows),
      function(i) {
        candidates <- reference_index[-rows[[i]]]
        ordering <- order(
          distances[i, candidates],
          candidates,
          method = "radix"
        )
        candidates[ordering[seq_len(integer_k)]]
      },
      integer(integer_k)
    )
    selected <- t(matrix(
      selected,
      nrow = integer_k,
      ncol = length(rows)
    ))
    selected_distance <- distances[cbind(
      rep(seq_along(rows), each = integer_k),
      as.vector(t(selected))
    )]

    index[rows, ] <- selected
    neighbour_distance[rows, ] <- matrix(
      selected_distance,
      nrow = length(rows),
      ncol = integer_k,
      byrow = TRUE
    )
  }

  structure(
    list(
      index = index,
      distance = neighbour_distance,
      metric = metric,
      device = device
    ),
    class = "cuda_knn"
  )
}

#' GPU-aware k-means clustering
#'
#' @param x Numeric matrix with observations in rows.
#' @param centers Number of clusters or a matrix of initial centres.
#' @param iter.max Maximum Lloyd iterations.
#' @param tolerance Convergence tolerance for centre movement.
#' @param seed Optional random seed used for initial centres.
#' @param device Device used for the distance step.
#' @return A `cuda_kmeans` list containing integer `cluster` assignments,
#'   final `centers`, per-cluster `withinss`, `tot.withinss`, the number of
#'   `iter`ations, a logical `converged` flag, and the actual distance `device`.
#' @export
#' @examples
#' set.seed(1)
#' x <- rbind(matrix(rnorm(40), 20, 2), matrix(rnorm(40, 4), 20, 2))
#' cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
cuda_kmeans <- function(x, centers, iter.max = 100L, tolerance = 1e-6,
                        seed = NULL,
                        device = c("auto", "cuda", "cpu")) {
  x <- .learn_matrix(x)
  device <- .learn_device(device)
  if (!is.numeric(iter.max) || length(iter.max) != 1L ||
      is.na(iter.max) || iter.max < 1 || iter.max != as.integer(iter.max)) {
    stop("`iter.max` must be a positive whole number.", call. = FALSE)
  }
  if (!is.numeric(tolerance) || length(tolerance) != 1L ||
      is.na(tolerance) || !is.finite(tolerance) || tolerance <= 0) {
    stop("`tolerance` must be a positive finite number.", call. = FALSE)
  }
  if (length(centers) == 1L && is.numeric(centers)) {
    k <- as.integer(centers)
    if (is.na(k) || k < 1L || k >= nrow(x) || centers != k) {
      stop("Numeric `centers` must be between 1 and nrow(x) - 1.",
           call. = FALSE)
    }
    centre_matrix <- .with_preserved_seed(
      seed,
      x[sample.int(nrow(x), k), , drop = FALSE]
    )
  } else {
    centre_matrix <- .learn_matrix(as.matrix(centers), "centers",
                                   min_rows = 1L)
    if (ncol(centre_matrix) != ncol(x)) {
      stop("Initial centres must have the same number of columns as `x`.",
           call. = FALSE)
    }
    k <- nrow(centre_matrix)
  }

  converged <- FALSE
  final_distance <- cuda_distance(x, centre_matrix, device = device)
  cluster <- max.col(-final_distance, ties.method = "first")
  for (iteration in seq_len(as.integer(iter.max))) {
    new_centres <- centre_matrix
    for (group in seq_len(k)) {
      members <- x[cluster == group, , drop = FALSE]
      if (nrow(members) > 0L) {
        new_centres[group, ] <- colMeans(members)
      }
    }
    movement <- max(abs(new_centres - centre_matrix))
    centre_matrix <- new_centres
    final_distance <- cuda_distance(x, centre_matrix, device = device)
    cluster <- max.col(-final_distance, ties.method = "first")
    if (movement <= tolerance) {
      converged <- TRUE
      break
    }
  }
  withinss <- vapply(
    seq_len(k),
    function(group) {
      members <- which(cluster == group)
      indices <- cbind(members, rep.int(group, length(members)))
      sum(final_distance[indices]^2)
    },
    numeric(1)
  )

  structure(
    list(
      cluster = cluster,
      centers = centre_matrix,
      withinss = withinss,
      tot.withinss = sum(withinss),
      iter = iteration,
      converged = converged,
      device = device
    ),
    class = "cuda_kmeans"
  )
}

#' @export
print.cuda_pca <- function(x, ...) {
  cat(sprintf(
    "<cuda_pca components=%s device=%s>\n",
    ncol(x$rotation), x$device
  ))
  print(x$rotation, ...)
  invisible(x)
}

#' @export
print.cuda_knn <- function(x, ...) {
  cat(sprintf(
    "<cuda_knn observations=%s k=%s metric=%s device=%s>\n",
    nrow(x$index), ncol(x$index), x$metric, x$device
  ))
  invisible(x)
}

#' @export
print.cuda_kmeans <- function(x, ...) {
  cat(sprintf(
    "<cuda_kmeans clusters=%s iterations=%s converged=%s device=%s>\n",
    nrow(x$centers), x$iter, x$converged, x$device
  ))
  print(x$centers, ...)
  invisible(x)
}
