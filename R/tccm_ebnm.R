# Two component covariate moderated EBNM
g = function(x) {1/(1+exp(-x))}

initialize_tccm_ebnm = function(beta, se, X, f0, f1, h, update.f0, update.f1, update.h){
  res = list(
    f0 = f0, f1 = f1, h = h,
    data = list(beta=beta, se=se, X=X, update.f0=update.f0, update.f1=update.f1, update.h=update.h),
    converged = FALSE,
    elbo = -Inf
  )
  return(res)
}

iterate_tccm_ebnm = function(res) {
  res$gamma <- compute_responsibilities(res)
  res$elbo <- c(res$elbo, compute_elbo_tccm_ebnm(res))
  print(paste0('E: ', diff(tail(res$elbo, 2))))
  if(res$data$update.f0){
    res$f0$params <- res$f0$update(res$data, res$gamma, res$f0$params)
  }
  if(res$data$update.f1){
    res$f1$params <- res$f1$update(res$data, res$gamma, res$f1$params)
  }
  if(res$data$update.h){
    # NOTE: we don't pass data, enforce decoupaling of logistic regression
    # from the rest of the model conditioned on responsibilities
    res$h$params <- res$h$update(res$gamma, res$h$params)
  }
  res$elbo <- c(res$elbo, compute_elbo_tccm_ebnm(res))
  print(paste0('M: ', diff(tail(res$elbo, 2))))
  return(res)
}

compute_responsibilities = function(res) {
  logit_pi <- res$h$get_prior_logits(res$data, res$h$params)
  f0_loglik <- res$f0$loglik(res$dat, res$f0$params) # + log(1/(1+exp(logit_pi)))
  f1_loglik <- res$f1$loglik(res$dat, res$f1$params) #+ log(1/(1+exp(-logit_pi)))
  logits <- f1_loglik - f0_loglik + logit_pi
  responsibilities = 1/(1 + exp(-logits))
  return(responsibilities)
}

compute_elbo_tccm_ebnm = function(res) {
  gamma <- pmin(pmax(res$gamma, 1e-10), 1-1e-10)
  f0_loglik <- res$f0$loglik(res$dat, res$f0$params)
  f1_loglik <- res$f1$loglik(res$dat, res$f1$params)
  logit_pi <- res$h$get_prior_logits(res$data, res$h$params)

  elbo <- sum((1- gamma) * f0_loglik + gamma * f1_loglik)
  elbo <- elbo - sum(gamma * log(gamma))
  elbo <- elbo + res$h$elbo(res$data, res$h$params, gamma)
  return(elbo)
}

#' @param beta a vector of observations (length n)
#' @param se a vector of standard errors (length n)
#' @param X an n x p matrix of covariates
#' @param f0 distribution/fit functions for null component
#' @param f1 distribution/fit function for non-null component
#' @param h fit functions for covariate model
fit_tccm_ebnm = function(beta,
                         se,
                         X,
                         f0,
                         f1,
                         h,
                         update.f0=FALSE,
                         update.f1=FALSE,
                         update.h=TRUE,
                         tol=1e-5,
                         maxit=1000,
                         verbose=TRUE) {
  res <- initialize_tccm_ebnm(beta, se, X, f0, f1, h, update.f0, update.f1, update.h)

  # set up progress bar
  if(verbose > 0){
    pb <- progress::progress_bar$new(
      format = "[:bar] :current/:total (:percent)",
      total=maxit)
    pb$tick(0)
  }

  # iterate over updates
  for (i in 1:maxit){
    if(verbose>0){pb$tick()}
    res <- iterate_tccm_ebnm(res)
    if(abs(diff(tail(res$elbo, 2))) < tol){
      res$converged <- TRUE
      if(verbose){message('\nconverged')}
      break
    }
  }
  return(res)
}


#' density families
#' loglik is a function that takes data and a list of params
#' update is a function that takes data, responsibilities
#'  and return params

generate_f_pointmass = function(){
  loglik = function(data, params){
    ll <- dnorm(data$beta, sd=data$se, log=TRUE)
    return(ll)
  }
  update = function(data, responsibilities, params){
    return(list())
  }
  params.init = list()
  f = list(
    loglik = loglik,
    update = update,
    params = params.init
  )
  return(f)
}

generate_f_normal = function(){
  loglik = function(data, params){
    sigma0 <- params$sigma0
    sigma <- sqrt(data$se^2 + sigma0^2)
    ll = dnorm(data$beta, mean=0, sd=sigma, log=TRUE)
    return(ll)
  }
  update = function(data, responsibilities, params=NULL){
    ll = function(x){ -sum(responsibilities * loglik(data, list(sigma0=x)))}
    sigma0 = optimize(ll, c(1e-8, 1000))$minimum
    params <- list(sigma0=sigma0)
    return(params)
  }
  params.init <- list(sigma0=1000)
  f = list(
    loglik = loglik,
    update = update,
    params = params.init
  )
}


#' params is a logistic.susie results object
generate_h_susie = function(data, L=10) {
  update = function(responsibilities, params){
    params$dat$y <- responsibilities
    for(i in 1:1){
      params <- iter_logistic_susie(params)
    }
    return(params)
  }
  get_prior_logits = function(data, params) {
    expected_beta <- rowSums(params$post_info$Alpha * params$post_info$Mu)
    logits <- params$post_info$delta + (params$dat$X %*% expected_beta)[, 1]
    return(logits)
  }
  elbo = function(data, params, responsibilities){
    # TODO: there is some data dependent term we need to add
    # since the data we give logistic susie changes per iteration
    elbo  <- tail(params$elbo, 1)
    # correcte elbo to account for uncertainty in responsibilities?
    #logits  <- get_prior_logits(data, params)
    #elbo <- elbo + sum(responsibilities * g(-logits) + (1-responsibilities) * g(logits))
    return(elbo)
  }

  # how to initialize?
  # TODO: initialize with point-normal responsibilities
  y.init <- as.integer(abs(data$beta / data$se) > 2)

  # TODO: expose more initialization options
  params.init = fit_logistic_susie(data$X, y.init, L=L, maxit=2)
  h = list(
    update = update,
    get_prior_logits = get_prior_logits,
    elbo = elbo,
    params = params.init
  )
  return(h)
}

generate_h_constant = function(logit) {
  update = function(responsibilities, params){
    pi = mean(responsibilities) + 1e-10
    return(list(logits = log(pi) - log(1-pi)))
  }
  get_prior_logits = function(data, params){
    logits <- rep(params$logit, length(data$beta))
    return(logits)
  }
  elbo = function(data, params, responsibilities){
    logits <- rep(params$logit, length(data$beta))
    elbo <- sum(responsibilities * g(-logits) + (1-responsibilities) * g(logits))
    return(elbo)
  }
  params.init = list(logit=logit)
  h = list(
    update = update,
    get_prior_logits = get_prior_logits,
    elbo = elbo,
    params = params.init
  )
  return(h)
}

#' Fit Point Normal SuSiE
#' @export
fit_tccm_point_normal_susie = function(beta, se, X, L=10, update.f1=TRUE, update.h=TRUE, maxit=100, verbose=TRUE) {
  data <- list(beta = beta, se=se, X=X)
  f0 <- generate_f_pointmass()
  f1 <- generate_f_normal()
  h <- generate_h_susie(data, L=L)
  res <- fit_tccm_ebnm(beta, se, X, f0, f1, h, update.f1=update.f1, update.h=update.h, maxit = maxit, verbose = verbose)
  res$h$params <- wrapup_logistic_susie(res$h$params)

  # for compatability with susie functions...
  res$mu <- res$h$params$mu
  res$mu2 <- res$h$params$mu2
  res$alpha <- res$h$params$alpha
  res$sets <- res$h$params$sets
  res$pip <- res$h$params$pip
  res$intercept <- res$h$params$intercept
  return(res)
}

#' Fit Point-Normal
#' @export
fit_tccm_point_normal = function(beta, se, X, logit=-1, update.f1=FALSE, update.h=FALSE, maxit=100, verbose=TRUE) {
  data <- list(beta = beta, se=se, X=X)
  f0 <- generate_f_pointmass()
  f1 <- generate_f_normal()
  h <- generate_h_constant(logit)
  res <- fit_tccm_ebnm(beta, se, X, f0, f1, h, update.f1 = update.f1, update.h=update.h, maxit = maxit, verbose = verbose)

  # for compatability with susie functions...
  res$mu <- res$h$params$mu
  res$mu2 <- res$h$params$mu2
  res$alpha <- res$h$params$alpha
  res$sets <- res$h$params$sets
  res$pip <- res$h$params$pip
  res$intercept <- res$h$params$intercept
  class('res') <- 'susie'
  return(res)
}
