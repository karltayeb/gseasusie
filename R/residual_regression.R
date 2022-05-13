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