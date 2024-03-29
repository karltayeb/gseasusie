#' wrapper for VEB.boost to fit logistic susie
#' and tidy up output to be consistent with other susie-type outputs
#' @export
fit_logistic_susie_veb_boost = function(X, y, L=10, ...){
    message('fitting logistic susie (via VEB.Boost)')
    tictoc::tic()
    veb.fit = VEB.Boost::veb_boost_stumps(
        X, y, family = 'binomial',
        include_stumps=FALSE,
        max_log_prior_var=35,
        scale_X = 'NA',
        growMode = 'NA',
        changeToConstant=F,
        k=L, ...
    )

    alpha <- t(do.call(cbind, lapply(veb.fit$leaves, function(x) x$learner$currentFit$alpha)))
    mu <- t(do.call(cbind, lapply(veb.fit$leaves, function(x) x$learner$currentFit$mu)))
    mu2 <- t(do.call(cbind, lapply(veb.fit$leaves, function(x) x$learner$currentFit$sigma2_post)))
    mu2 <- mu2 + mu^2
    elbo <- veb.fit$ELBO_progress[[2]]

    coef <- colSums(alpha * mu)
    intercept <- veb.fit$mu1[1] - sum(X[1,] * coef)


    res <- list(alpha=alpha, intercept=intercept, mu=mu, mu2=mu2, elbo=elbo, veb.fit=veb.fit)
    class(res) <- 'susie'
    colnames(res$alpha) <- colnames(X)
    colnames(res$mu) <- colnames(X)
    res$pip <- susieR::susie_get_pip(res)
    names(res$pip) <- colnames(X)
    res$sets <- susieR::susie_get_cs(res, X=X)
    tictoc::toc()
    return(res)
}
