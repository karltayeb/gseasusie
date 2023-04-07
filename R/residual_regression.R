#' compute expected effect for each covariate
#' @param fit a susie-type fit
susie_coef = function(fit){
  return(colSums((fit$alpha * fit$mu)))
}

#' compute prediction
#' @param fit a susie-type fit
susie_pred = function(fit, X){
  # TODO: take X as an argument
  return((fit$intercept + X %*% susie_coef(fit))[, 1])
}

#' compute logistic regression with offset
#' @param y a vector of binary responses
#' @param x a vector for a single covariate
#' @param o a vector of offsets (predictions from SuSiE)
logistic_regression = function(y, x, o){
  X <- matrix(c(rep(1, length(x)), x), ncol = 2)
  summary(fastglm::fastglm(x=X, y=y, offset=o, family='binomial'))$coef %>% tail(1)
}

#' fit marginal residual regression, using susie predictions as offset
#' report residual enrichments for gene sets `X`
#' after accounting for logistic SuSiE predictions
#' @param fit a logistic susie fit
#' @param y the binary response/gene list
#' @param X a n x p gene set matrix, n (genes) must be same as susie fit
#' @export
fit_residual_regression = function(X, y, fit){
  pred <- susie_pred(fit, X)
  p <- dim(X)[2]

  message('performing logistic regression on residuals')
  tictoc::tic()
  residual_res <- furrr::future_map(1:p, ~logistic_regression(y, X[, .x], pred))
  tictoc::toc()
  residual_res <- do.call(rbind, residual_res)
  residual_res <- tibble::as_tibble(residual_res) %>%  dplyr::mutate(geneSet = colnames(X))
  colnames(residual_res) <- c('residual_beta', 'residual_se', 'residual_z', 'residual_p', 'geneSet')
  return(residual_res)
}

#' fit marginal regression to each geneset
#' @param X an n x p gene set matrix (genes x gene sets)
#' @param y a binary response vector (length n)
#' @export
fit_marginal_regression = function(X, y){
  p <- dim(X)[2]
  message('fitting marginal logistic regression')
  tictoc::tic()
  res <- furrr::future_map(1:p, ~logistic_regression(y, X[, .x], NULL))
  tictoc::toc()
  res <- do.call(rbind, res)
  res <- tibble::as_tibble(res) %>%  dplyr::mutate(geneSet = colnames(X))
  colnames(res) <- c('marginal_beta', 'marginal_se', 'marginal_z', 'marginal_p', 'geneSet')
  return(res)
}

.fit_univariate_regression_jax = function(X, y, offset=0, proc){
  message('\tfitting univariate regression...')
  basilisk::basiliskRun(
    proc, function(X, y, offset) {
      np <- reticulate::import("numpy")
      reticulate::source_python(system.file("python", "logistic_regression.py", package = "gseasusie"))
      mpy <- logistic_regression_jax(X, np$array(y), np$array(offset))
      mpy <- mpy %>%
        tibble::as_tibble() %>%
        dplyr::mutate(geneSet = colnames(X)) %>%
        dplyr::mutate(pval = dplyr::if_else(is.finite(effect), pval, 1.)) %>%
        dplyr::select(geneSet, effect, effect_se, intercept, intercept_se, loglik, lrts, pval, eps, total_iteration)
      mpy
    }, X=X, y=y, offset=offset
  )
}

#' @export
fit_marginal_regression_jax = function(X, y, offset=0, stride=1000){
  proc <- basilisk::basiliskStart(jax_env)
  on.exit(basilisk::basiliskStop(proc))

  p = dim(X)[2]
  start_idx = seq(1, p, stride)
  end_idx = pmin(start_idx + stride - 1, p)

  message('fitting marginal logistic regressions...')
  tictoc::tic()
  mpy <- purrr::map_dfr(
    1:length(start_idx), ~.fit_univariate_regression_jax(
      X[, start_idx[.x]:end_idx[.x]], y, offset=offset, proc)
  )
  tictoc::toc()
  return(mpy)
}

#' @export
fit_residual_regression_jax = function(X, y, fit, stride=1000){
  proc <- basilisk::basiliskStart(jax_env)
  on.exit(basilisk::basiliskStop(proc))

  p = dim(X)[2]
  start_idx = seq(1, p, stride)
  end_idx = pmin(start_idx + stride - 1, p)
  offset <- susie_pred(fit, X)

  message('fitting residual logistic regressions...')
  tictoc::tic()
  mpy <- purrr::map_dfr(
    1:length(start_idx), ~.fit_univariate_regression_jax(
      X[, start_idx[.x]:end_idx[.x]], y, offset, proc)
  )
  tictoc::toc()
  return(mpy)
}


.fit_univariate_vb_jax = function(X, y, tau0, offset, proc){
  mpy <- basilisk::basiliskRun(
    proc, function(X, y, tau0, offset) {
      np <- reticulate::import("numpy")
      jnp <- reticulate::import("numpy")
      reticulate::source_python(system.file("python", "logistic_regression_vb.py", package = "gseasusie"))
      mpy <- marginal_vb_jax(jnp$array(X), jnp$array(y), tau0, jnp$array(offset))
      mpy
    }, X=X, y=y, tau0=tau0, offset=offset
  )
  return(mpy)
}

#' Fit Univariate VB Regression
#'
#' Fits variational approximation for univariate regression with a normal prior
#' Implimented in JAX, fits a regression model for each column of X
#'   y ~ Bernoulli(sigmoid(x * b))
#'   b ~ N(0, 1/tau0)
#' @param X an n x p matrix
#' @param y a n vector binary response
#' @param tau0 the prior precision of the effect
#' @param offset offset
#' @export
fit_univariate_vb_regression_jax = function(X, y, tau0, offset=NULL){
  proc <- basilisk::basiliskStart(gseasusie:::jax_env)
  on.exit(basilisk::basiliskStop(proc))

  message('Compute univariate logistic regression VB...')
  tictoc::tic()
  if(is.null(offset)){
    offset = rep(0, length(y))
  }
  res <- .fit_univariate_vb_jax(X, y, tau0, offset, proc)
  tictoc::toc()

  return(res)
}

# 2d quadrature of 1 dimensional logistic-normal
.compute_evidence_quadrature_2d = function(X, y, offset, params, n, proc){
  mpy <- basilisk::basiliskRun(
    proc, function(X, y, offset, params, n) {
      # imports
      np <- reticulate::import("numpy")
      jnp <- reticulate::import("jax.numpy")
      reticulate::source_python(system.file("python", "logistic_regression_quadrature.py", package = "gseasusie"))

      # number of covariates
      p <- dim(X)[2]

      # get quadrature points
      q <- statmod::gauss.quad.prob(n, dist='normal')
      nodes <- jnp$array(q$nodes)
      weights <- jnp$array(q$weights)

      mpy <- with(params, purrr::map(1:p, ~logreg_quad_jax(jnp$array(X[, .x]), jnp$array(y), jnp$array(offset), b_mu, b_sigma, b0_mu, b0_sigma, nodes, weights)))
      mpy <- np$array(unlist(mpy))
      mpy
  }, X=X, y=y, offset=offset, params=params, n=n)
  return(mpy)
}
compute_evidence_quadrature_2d = function(X, y, offset, params, n=128){
  proc <- basilisk::basiliskStart(jax_env)
  on.exit(basilisk::basiliskStop(proc))

  message('computing evidence via quadrature...')
  tictoc::tic()
  res <- .compute_evidence_quadrature_2d(X, y, offset, params, n, proc)
  tictoc::toc()

  return(res)
}

# 2d quadrature of 1 dimensional logistic-normal
.compute_evidence_quadrature_best_b0= function(X, y, offset, params, n, proc){
  mpy <- basilisk::basiliskRun(
    proc, function(X, y, offset, params, n) {
      # imports
      np <- reticulate::import("numpy")
      jnp <- reticulate::import("jax.numpy")
      reticulate::source_python(system.file("python", "logistic_regression_quadrature.py", package = "gseasusie"))

      # number of covariates
      p <- dim(X)[2]

      # get quadrature points
      q <- statmod::gauss.quad.prob(n, dist='normal')
      nodes <- jnp$array(q$nodes)
      weights <- jnp$array(q$weights)

      mpy <- with(params, purrr::map(1:p, ~logreg_quad_best_intercept_jax(jnp$array(X[, .x]), jnp$array(y), jnp$array(offset), b_mu, b_sigma, nodes, weights)))
      mpy <- np$array(unlist(mpy))
      mpy
  }, X=X, y=y, offset=offset, params=params, n=n)
  return(mpy)
}

compute_evidence_quadrature_best_b0 = function(X, y, offset, params, n=128){
  proc <- basilisk::basiliskStart(jax_env)
  on.exit(basilisk::basiliskStop(proc))

  message('computing approximate evidence via quadrature (best b0)...')
  tictoc::tic()
  res <- .compute_evidence_quadrature_best_b0(X, y, offset, params, n, proc)
  tictoc::toc()

  return(res)
}


# quadrature with fixed intercept
.compute_evidence_quadrature_fixed_b0= function(X, y, offset, b0, params, n, proc){
  mpy <- basilisk::basiliskRun(
    proc, function(X, y, offset, b0, params, n) {
      # imports
      np <- reticulate::import("numpy")
      jnp <- reticulate::import("jax.numpy")
      reticulate::source_python(system.file("python", "logistic_regression_quadrature.py", package = "gseasusie"))

      # number of covariates
      p <- dim(X)[2]

      # get quadrature points
      q <- statmod::gauss.quad.prob(n, dist='normal')
      nodes <- jnp$array(q$nodes)
      weights <- jnp$array(q$weights)

      mpy <- with(params, purrr::map(1:p, ~logreg_quad_fixed_intercept_jax(jnp$array(X[, .x]), jnp$array(y), jnp$array(offset), b_mu, b_sigma, b0[.x], nodes, weights)))
      mpy <- np$array(unlist(mpy))
      mpy
  }, X=X, y=y, offset=offset, b0=b0, params=params, n=n)
  return(mpy)
}

compute_evidence_quadrature_fixed_b0 = function(X, y, offset, b0, params, n=128){
  proc <- basilisk::basiliskStart(jax_env)
  on.exit(basilisk::basiliskStop(proc))

  if(length(b0) == 1){
    b0 <- rep(1, dim(X)[2]) * b0
  }

  message('computing approximate evidence via quadrature (fixed  b0)...')
  tictoc::tic()
  res <- .compute_evidence_quadrature_fixed_b0(X, y, offset, b0, params, n, proc)
  tictoc::toc()

  return(res)
}



.compute_evidence_quadrature_fixed_b0_batched= function(X, y, offset, b0, params, n, proc){
  mpy <- basilisk::basiliskRun(
    proc, function(X, y, offset, b0, params, n) {
      # imports
      np <- reticulate::import("numpy")
      jnp <- reticulate::import("jax.numpy")
      reticulate::source_python(system.file("python",
                                            "logistic_regression_quadrature.py",
                                            package = "gseasusie"))

      # get quadrature points
      q <- statmod::gauss.quad.prob(n, dist='normal')
      nodes <- jnp$array(q$nodes)
      weights <- jnp$array(q$weights)

      mpy <- with(params, logreg_quad_fixed_intercept_X_jax(jnp$array(X),
                                                            jnp$array(y),
                                                            jnp$array(offset),
                                                            jnp$array(b_mu),
                                                            jnp$array(b_sigma),
                                                            jnp$array(b0),
                                                            nodes,
                                                            weights))

      mpy <- np$array(unlist(mpy))
      mpy
    }, X=X, y=y, offset=offset, b0=b0, params=params, n=n)
  return(mpy)
}

compute_evidence_quadrature_fixed_b0_batched = function(X, y, offset, b0, params, n=128){
  proc <- basilisk::basiliskStart(jax_env)
  on.exit(basilisk::basiliskStop(proc))

  if(length(b0) == 1){
    b0 <- rep(1, dim(X)[2]) * b0
  }

  if(length(params$b_mu) == 1){
    params$b_mu <- rep(1, dim(X)[2]) * params$b_mu
  }

  if(length(params$b_sigma) == 1){
    params$b_sigma <- rep(1, dim(X)[2]) * params$b_sigma
  }

  message('computing approximate evidence via quadrature (fixed  b0)...')
  tictoc::tic()
  res <- .compute_evidence_quadrature_fixed_b0_batched(X, y, offset, b0, params, n, proc)
  tictoc::toc()

  return(res)
}

