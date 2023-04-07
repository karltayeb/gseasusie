test_that( "True effect has smallest ELBO", {
  n <- 1000
  p <- 10
  X <- matrix(rnorm(n*p), nrow = n)
  y <- rbinom(n, 1, 1/(1+exp(-X[,1])))

  tictoc::tic()
  res <- fit_univariate_vb_regression_jax(X, y, 1)
  tictoc::toc()
  expect_equal(which.max(res$elbo), 1)
})


test_that( "True effect has smallest ELBO", {
  n <- 1000
  p <- 10
  X <- matrix(rnorm(n*p), nrow = n)
  y <- rbinom(n, 1, 1/(1+exp(-X[,1])))

  res <- compute_evidence_quadrature_jax(X, y, 0, 1, 0, 1, n=128)
  expect_equal(which.max(res$elbo), 1)
})
