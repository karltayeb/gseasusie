#' @export
logistic_susie_sparse <- function(X_sp, y, L = 10L, prior_variance = 1., maxiter = 100, tol = 1e-5) {
  np <- reticulate::import("numpy")
  gibss <- reticulate::import("gibss")
  gibss$logistic_sparse$fit_logistic_susie2(X_sp, np$array(y), L = as.integer(L), prior_variance = prior_variance, maxiter = as.integer(maxiter), tol = tol)
}
