
### VB LOWER BOUND SUSIE BELOW
### NOTE: This is basic code, and I did not attempt to mirror the level of numerical sophistication in the susie functions
### If this idea is worth pursuing further, then this code can be improved

# library(BiocGenerics)  # TODO: don't pollute global namespace with this
# library(matrixStats)

colScale = function(x, scale = TRUE) {
    x = BiocGenerics::t((BiocGenerics::t(x) * scale))
    return(x)
}

centeredmatmul = function(dat, A){
  n <- dim(dat$X.raw)[1]
  res <- dat$X.raw %*% A
  if(dat$center){
    if(is.null(dim(A))){
      res <- res - sum(dat$X.shift * A)
    } else{
      res <- BiocGenerics::t(BiocGenerics::t(res) - (dat$X.shift %*% A)[1,])
    }
  }
  return(res)
}

centeredmatmul2 = function(dat, A){
  n <- dim(dat$X.raw)[1]
  res <- dat$X2.raw %*% A
  if(dat$center){
    if(is.null(dim(A))){
      res <- res -
        (2 * dat$X.raw %*% (diag(dat$X.shift) %*% A)) +
        sum(dat$X.shift^2 * A)
    } else{
      XDA <- dat$X.raw %*% (diag(dat$X.shift) %*% A)
      res <- res - 2*XDA
      res <- t(t(res) + (dat$X.shift^2 %*% A)[1,])
    }
  }
  return(res)
}

test_cmm = function(dat){
  p <- dim(dat$X)[2]
  A <- matrix(rnorm(10*p), nrow = p)
  a <- dat$X %*% A
  b <- centeredmatmul(dat, A)
  all.equal(a, b)
}

calc_Q = function(dat, Sigma2, Mu, Alpha, Z, delta) {
  # X is data matrix
  # Sigma2[j, l] is the posterior variance for b_l when entry j is selected, p x L
  # Mu[j, l] is the posterior mean for b_l when entry j is selected, p x L
  # Alpha[j, l] is the posterior probability selecting entry j from b_l, p x L
  # Z is matrix of covariates (e.g. column for intercept, top 10 PCs, etc)
  # delta is current estimate for effects of Z variables
  ASU2 = Alpha * (Sigma2 + Mu^2) # [j, l] = alpha[j, l] * (Sigma2[j, l] + Mu[j, l]^2)
  # Q = BiocGenerics::rowSums(dat$X2 %*% ASU2) # start w/ sum_l E_(ql)[(x_i' b_l)^2]
  Q = centeredmatmul2(dat, rowSums(ASU2))[,1]

  # now add 2 sum_l sum_{k>l} (x_i' b_l_post)(x_i' b_k_post)
  b_post_mat = Mu * Alpha # each column is posterior mean for b_l, p x L
  X_b_post = centeredmatmul(dat, b_post_mat) # [i, l] = x_i' b_l_post
  Q = Q + BiocGenerics::rowSums(X_b_post)^2 - BiocGenerics::rowSums(X_b_post^2)

  # now, add other terms with Z and delta
  Q = Q + as.numeric(2 * centeredmatmul(dat, rowSums(b_post_mat)) * (Z %*% delta)) + as.numeric((Z %*% delta)^2)
  return(Q)
}

g = function(x) { # ilogit function
  1 / (1 + exp(-x))
}


update_xi = function(dat, Sigma2, Mu, Alpha, Z, delta) {
  # X is data matrix
  # Sigma2[j, l] is the posterior variance for b_l when entry j is selected, p x L
  # Mu[j, l] is the posterior mean for b_l when entry j is selected, p x L
  # Alpha[j, l] is the posterior probability selecting entry j from b_l, p x L
  # Z is matrix of covariates (e.g. column for intercept, top 10 PCs, etc)
  # delta is current estimate for effects of Z variables

  Q = calc_Q(dat, Sigma2, Mu, Alpha, Z, delta)

  xi = sqrt(Q)

  return(xi)

}


compute_g_xi = function(dat, xi){
  g_xi = g(xi) # vector of g(xi_i), pre-compute once
  g_xi_5_xi = ((g_xi - .5) / xi) # vector of (g(xi_i) - .5) / xi_i, pre-compute once
  g_xi_5_xi[xi == 0] = .25 # case of 0/0 (e.g. x_i is all 0), use L'Hopital
  common_denoms = g_xi_5_xi %*% dat$X2 # appears many times, compute once, posterior precisions
  return(list(g_xi, g_xi_5_xi = g_xi_5_xi, common_denoms = common_denoms))
}

update_b_l = function(dat, xi, prior_weights, V, Sigma2, Mu, Alpha, l, Z, delta, g_xi_precomp) {
  # X is data matrix
  # y is binary response
  # xi is lower-bound approximation parameters
  # prior_weights is prior probabilities for selecting j (p-vector)
  # V is prior variance of current effect (scalar)
  # Sigma2[j, l] is the posterior variance for b_l when entry j is selected, p x L
  # Mu[j, l] is the posterior mean for b_l when entry j is selected, p x L
  # Alpha[j, l] is the posterior probability selecting entry j from b_l, p x L
  # l is index to update
  # Z is matrix of covariates (e.g. column for intercept, top 10 PCs, etc)
  # delta is current estimate for effects of Z variables
  if(is.null(g_xi_precomp)){
    g_xi_precomp <- compute_g_xi(dat, xi, V)
  }
  g_xi_5_xi <- g_xi_precomp$g_xi_5_xi
  common_denoms <- g_xi_precomp$common_denoms + (1 / V)

  b_post_mat = Mu * Alpha # each column is posterior mean for b_l, p x L
  b_post_not_l = rowSums(as.matrix(b_post_mat[, -l], nrow = nrow(Mu))) # posterior, sum_(k != l) b_k_post


  # update Alpha[, l]
  tmp <- centeredmatmul(dat, b_post_not_l)[,1]
  nums = (dat$y - .5 -  (g_xi_5_xi * (tmp + (Z %*% delta)[,1]))) %*% dat$X # numerator in exp
  Alpha[, l] = as.numeric(log(prior_weights) + (nums^2 / (2*common_denoms)) - (1/2)*log(common_denoms))
  Alpha[, l] = exp(Alpha[, l] - max(Alpha[, l])) # remove max for stability, everything still proportional
  Alpha[, l] = Alpha[, l] / sum(Alpha[, l]) # normalize, sum to 1

  # update Mu[, l]
  Mu[, l] = as.numeric(nums / common_denoms)

  # update Sigma[, l]
  Sigma2[, l] = as.numeric(1 / common_denoms)
  return(list(Sigma2 = Sigma2, Mu = Mu, Alpha = Alpha))
}


update_delta = function(dat, xi, Mu, Alpha, Z) {
  # X is data matrix
  # y is binary response
  # xi is lower-bound approximation parameters
  # Mu[j, l] is the posterior mean for b_l when entry j is selected, p x L
  # Alpha[j, l] is the posterior probability selecting entry j from b_l, p x L
  # Z is matrix of covariates (e.g. column for intercept, top 10 PCs, etc)

  g_xi = g(xi) # vector of g(xi_i), pre-compute once
  g_xi_5_xi = ((g_xi - .5) / xi) # vector of (g(xi_i) - .5) / xi_i, pre-compute once
  g_xi_5_xi[xi == 0] = .25

  #D = Matrix::Diagonal(x = g_xi_5_xi / 2)
  D = Matrix::Diagonal(x = g_xi_5_xi)
  ZtDZ = (t(Z) * g_xi_5_xi) %*% Z  # D %*% Z
  DXB = colScale(centeredmatmul(dat, rowSums(Mu * Alpha)), g_xi_5_xi)#  %*% rowSums(Mu * Alpha)
  RHS = t(Z) %*% (dat$y - .5 - DXB) # RHS of system LL'delta = RHS

  # NOTE: the below system can/should be solved w/ Cholesky and forward/backward substitution (but if Z has small # of columns, shouldn't matter)
  #Lt = chol(ZtDZ)
  #u = forwardsolve(t(Lt), RHS)
  #delta = backsolve(Lt, u)
  ## COULD USE chol2inv!!!!
  #delta = solve(ZtDZ, RHS)
  delta = solve(as.matrix(ZtDZ), as.matrix(RHS))

  return(as.numeric(delta))

}

update_all = function(dat,
                      xi,
                      prior_weights,
                      V,
                      Sigma2,
                      Mu,
                      Alpha,
                      Z,
                      delta,
                      estimate_prior_variance,
                      share_prior_variance,
                      intercept) {
  #  X is data matrix
  # y is binary response
  # xi is lower-bound approximation parameters
  # prior_weights is prior probabilities for selecting j (p-vector)
  # V is prior variance (scalar if the same for all L effects, or vector of length L if different)
  # Sigma2[j, l] is the posterior variance for b_l when entry j is selected, p x L
  # Mu[j, l] is the posterior mean for b_l when entry j is selected, p x L
  # Alpha[j, l] is the posterior probability selecting entry j from b_l, p x L
  # Z is matrix of covariates (e.g. column for intercept, top 10 PCs, etc)
  # delta is current estimate for effects of Z variables
  # estimate_prior_variance is logical for if prior variance, V, should be estimated

  L = ncol(Mu)

  # update delta
  if (intercept) { # if covariates and/or intercept
    delta = update_delta(dat, xi, Mu, Alpha, Z)
  }

  if (length(V) == L) {
    prior_vars = V
  } else if (length(V) == 1) {
    prior_vars = rep(V, L)
    V = prior_vars
  } else {
    stop("Argument 'V' must be of length either 1 or L")
  }

  # now, iterate over l = 1:L
  g_xi_precomp <- compute_g_xi(dat, xi)
  for (l in 1:L) {
    res_l = update_b_l(dat, xi, prior_weights, prior_vars[l], Sigma2, Mu, Alpha, l, Z, delta, g_xi_precomp)
    Sigma2 = res_l$Sigma2
    Mu = res_l$Mu
    Alpha = res_l$Alpha
  }

  # now, update xi
  xi = update_xi(dat, Sigma2, Mu, Alpha, Z, delta)

  if (estimate_prior_variance == TRUE) { # if estimating prior variance
    ASU2 = Alpha * (Sigma2 + Mu^2) # [j, l] = alpha[j, l] * (Sigma2[j, l] + Mu[j, l]^2)
    if (share_prior_variance) { # if common prior variance for all effects
      V = sum(ASU2) / ncol(Mu) # ncol(Mu) = L = sum_l sum_j alpha_jl
    } else { # if different for all effects (already know length is L, checked a few lines before)
      V = colSums(ASU2)
    }
  }


  return(list(Sigma2 = Sigma2, Mu = Mu, Alpha = Alpha, delta = delta, xi = xi, V = V))

}

init_logistic_susie = function(
  X, y, L = 10, V = 1, prior_weights = NULL,
  init.intercept = NULL, intercept = TRUE, Z = NULL,
  estimate_prior_variance = TRUE, share_prior_variance = FALSE,
  standardize=FALSE, center=TRUE) {

  p = ncol(X)
  n = nrow(X)

  se <- sqrt(sparseMatrixStats::colVars(X))
  X.raw <- X
  X.shift <- BiocGenerics::colMeans(X.raw)

  if(standardize){
    X <- scale(X)
  }else if(center){
    X <- scale(X, scale=F)
  }

  if (is.null(Z)) {
    Z = matrix(1, nrow = n, ncol = 1)
  } else {
    col_variances = apply(Z, MARGIN = 2, var)
    if (any(col_variances == 0)) { # is constant column in Z matrix
      stop("Matrix 'Z' cannot have a constant column")
    }
    Z = cbind(matrix(1, nrow = n, ncol = 1), Z) # add intercept column
  }
  if (is.null(prior_weights)) {
    prior_weights = rep(1 / p, p)
  }

  #' save all the things that don't change here
  dat <- list(
    X=X, X2=X^2,
    X.raw = X.raw, X2.raw = X.raw^2, X.shift=X.shift, X.scale=se,
    y=y, Z=Z, prior_weights=prior_weights,
    intercept=intercept,
    estimate_prior_variance=estimate_prior_variance,
    share_prior_variance=share_prior_variance,
    standardize = standardize, center = center)

  # place to store posterior info for each l = 1, ..., L
  # initialize: could think of something better
  #delta = glm(y ~ Z - 1, family = "binomial")$coef # initialize to regular GLM solution
  # TODO: what if we have fixed intercept but also covariates?
  if (!is.null(init.intercept)) { # if init.intercept is specified
    delta = init.intercept
  } else { # initialize to regular GLM solution w/ just Z (no X)
    delta = glm(as.numeric(y) ~ Z - 1, family = "binomial")$coef
  }
  Alpha = matrix(prior_weights, nrow = p, ncol = L)
  Mu = matrix(0, nrow = p, ncol = L)
  Sigma2 = matrix(V, nrow = p, ncol = L, byrow = T)
  xi = update_xi(dat, Sigma2, Mu, Alpha, Z, delta)

  post_info = list(Sigma2 = Sigma2, Mu = Mu, Alpha = Alpha, delta = delta, xi = xi, V = V)

  elbo <- calc_ELBO(dat, post_info)
  res = list(dat=dat, post_info=post_info, elbo=elbo)
  return(res)
}

iter_logistic_susie = function(res, compute_elbo=T){
  dat <- res$dat
  post_info <- res$post_info
  post_info  <- update_all(
    dat, post_info$xi, dat$prior_weights,
    post_info$V, post_info$Sigma2, post_info$Mu,
    post_info$Alpha, dat$Z, post_info$delta,
    dat$estimate_prior_variance, dat$share_prior_variance, dat$intercept)
  res$post_info <- post_info
  if(compute_elbo){
    elbo <- calc_ELBO(dat, post_info)
    res$elbo <- c(res$elbo, elbo)
  }
  return(res)
}

wrapup_logistic_susie = function(res){
  # TODO put back into original X scale
  X <- res$dat$X
  post_info <- res$post_info
  res = list(
    alpha = t(post_info$Alpha),
    mu = t(post_info$Mu),
    mu2 = t(post_info$Sigma2 + post_info$Mu^2),
    elbo = res$elbo,
    dat = res$dat,
    post_info = res$post_info
  )
  class(res) <- 'susie'
  colnames(res$alpha) <- colnames(X)
  colnames(res$mu) <- colnames(X)
  res$pip <- susieR::susie_get_pip(res)
  names(res$pip) <- colnames(X)
  res$sets <- susieR::susie_get_cs(res, X=X)
  res$intercept <- res$post_info$delta[1]
  return(res)
}

# calculate the variational lower bound
# CAREFUL: Not sure what to do when Alpha[j, l] = 0 (we get 0*ln(0)). I will set this to 0
# Note: This expression is only valid when xi has been updated to be sqrt(Q), where Q is the
# expectation of the square of the linear predictor under the variational posterior (what we update xi to nomrally)
calc_ELBO = function(dat, post_info){
  # unpack
  Alpha <- post_info$Alpha
  Mu <- post_info$Mu
  Sigma2 <- post_info$Sigma2
  V = post_info$V
  prior_weights = dat$prior_weights
  Z  <-  dat$Z
  delta  <- post_info$delta
  xi  <-  post_info$xi

  p = nrow(Mu)
  L = ncol(Mu)
  P = matrix(prior_weights, nrow = p, ncol = L)
  b_post = rowSums(Alpha * Mu)

  V = matrix(V, nrow = p, ncol = L, byrow = T) # for proper arithmetic, either if V is a scalar or a vector of length L

  expected_log_lik = sum(log(g(xi)) + (dat$y - .5) * as.numeric(centeredmatmul(dat, b_post) + (Z %*% delta)) - (xi / 2))
  KL_div_vb_prior = Alpha * (log(Alpha) - log(P) + (log(V) / 2) - (log(Sigma2) / 2) - .5 + ((Sigma2 + Mu^2) / (2 * V)))
  KL_div_vb_prior[Alpha == 0] = 0
  KL_div_vb_prior = sum(KL_div_vb_prior)

  if (KL_div_vb_prior < 0) { # to diagnose any issues
    warning("KL Divergence calculated to be < 0")
  }

  # TERMS BELOW WRONG B/C FORGOT MULTIPLICIATIVE FACTOR OF alpha_jl
  #KL_div_vb_prior = sum(log(Alpha)) - L*sum(log(prior_weights)) + (L*p*log(V) / 2) - sum(log(Sigma2) / 2) - (L*p / 2) + (sum(Sigma2 + Mu^2) / (2*V))
  #KL_div_vb_prior = sum(log(Alpha)) - sum(log(Sigma2) / 2) + (sum(Sigma2 + Mu^2) / (2*V)) # up to a constant

  ELBO = expected_log_lik - KL_div_vb_prior
  return(ELBO)
}

#' Fit logistic SuSiE with variational inference
#'
#' This is a driver function for logistic SuSiE (Sum of Single Effects Regression)
#'
#' @param X a sparse matrix of covariates (n x p)
#' @param y a vector response (n)
#' @param L the number of non-zero effects
#' @param V the prior variance of effects, must either be length 1 or L
#' @param prior_weights the prior probability of selecting covariate (defaults to 1/p)
#' @param init.initercept initialize the intercept parameter
#' @param intercept boolean to include intercept in the model
#' @param Z matrix of covariates to always include in the model
#' @param estimate_prior_variance: boolean to estimate `V`
#' @param share_prior_variance: boolean to estimate a common `V` for all effects,
#'  if False, estimate `V` for each effect seperately
#' @param standardize boolean to standardize `X` before fitting
#' @param center_X boolean to share `X` before fitting
#' @param tol tolerance to declare convergence
#' @param maxit maximum number of iterations
#' @param verbose print status/progress bar while fitting
#' @export
fit_logistic_susie = function(
  X, y, L = 10, V = 1, prior_weights = NULL,
  init.intercept = NULL, intercept = TRUE, Z = NULL,
  estimate_prior_variance = TRUE, share_prior_variance = FALSE,
  standardize=FALSE, center_X=TRUE,
  tol = 1e-3, maxit = 1000, verbose=0) {

  res <- init_logistic_susie(
    X, y, L, V, prior_weights, init.intercept, intercept, Z,
    estimate_prior_variance, share_prior_variance, standardize, center_X)
  if(verbose > 0){
    pb <- progress::progress_bar$new(
      format = "[:bar] :current/:total (:percent)", total=maxit
    )
    pb$tick(0)
  }
  converged <- FALSE
  for (i in 1:maxit){
    if(verbose>0){pb$tick()}
    res <- iter_logistic_susie(res)
    if(diff(tail(res$elbo, 2)) < tol){
      converged <- TRUE
      if(verbose){message('\nconverged')}
      break
    }
  }
  if(!converged){
    warning(paste0('did not converge after ', maxit, ' iterations'))
  }
  res <- wrapup_logistic_susie(res)
  res$converged <- TRUE
  return(res)
}
