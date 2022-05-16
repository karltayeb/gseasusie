import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

import pandas as pd
from scipy import stats
from scipy import sparse
import numpy as np

"""
likelihood
"""

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def predict(beta, x, offset=0):
    return(beta[0] + beta[1]*x + offset)

def loglik(beta, x, y, offset=0, penalty=0):
    preds = predict(beta, x, offset)
    loss = jnp.sum(y * preds + jnp.log(sigmoid(-preds)))
    loss = loss - penalty * beta[1]**2  # add penalty to beta1 (for fitting null model)
    return(loss)


"""
functions for computing (approximate) standard erros
"""
def W(beta, x, offset=0):
    pred = predict(beta, x, offset)
    lnW = pred - 2 * jnp.log(1 + jnp.exp(pred))
    W = jnp.exp(lnW)
    return W

def XWX(W, x):
    a11 = jnp.sum(W)
    a12 = jnp.sum(W * x)
    a22 = jnp.sum(W * x * x)
    return jnp.array([[a11, a12], [a12, a22]])
    
def sandwich_betahat_cov(beta, x, y, offset=0):
    """
    get sandwich estimate of betahat covariance (not working)
    """
    What = W(beta, x, offset)
    Wtilde = y - sigmoid(predict(beta, x, offset))
    
    XWXhat = XWX(What, x)
    XWXtilde = XWX(Wtilde, x)
    return jnp.linalg.solve(XWXhat, jnp.linalg.solve(XWXhat, XWXtilde).T)

def sandwich_betahat_se(beta, x, y, offset=0):
    """
    estimate se of beta0, beta1 using sandwiched covariance estimate
    """
    return jnp.sqrt(jnp.diag(sandwich_betahat_cov(beta, x, y, offset)))

def betahat_se(beta, x, y, offset=0):
    """
    estimate se of beta0, beta1 from inverse Fisher information
    """
    What = W(beta, x, offset)
    cov = jnp.linalg.inv(XWX(What, x))
    return jnp.sqrt(jnp.diag(cov))

"""
# fit univariate logistic regression via Newton method
# which is equivalent to interatively reweighted least squares (I think)
# we initialize with the null model... hopefully that's okay!
"""

J = jax.jacfwd(loglik, 0)
H = jax.hessian(loglik, 0)

@jit
def newtonStep(beta, x, y, offset=0, penalty=0):
    beta = beta[0]
    beta_next = beta - jnp.linalg.solve(H(beta, x, y, offset, penalty), J(beta, x, y, offset, penalty))
    diff = jnp.sum((beta_next - beta)**2)
    return beta_next, diff

@jit
def mle(beta_init, x, y, offset=0, penalty=0, tol=1e-6): 
    beta_init = (beta_init, tol+1) 
    step = lambda b: newtonStep(b, x, y, offset, penalty)
    beta, diff = jax.lax.while_loop(lambda b: b[1] > tol, step, beta_init)
    fit_loglik = loglik(beta, x, y, offset) + penalty * beta[1]**2  # get rid of penalty (only for optimization)
    se = betahat_se(beta, x, y, offset=0)
    return {
        'intercept': beta[0],
        'effect': beta[1],
        'intercept_se': se[0],
        'effect_se': se[1],
        'loglik': fit_loglik,
        'eps': diff}


def logistic_regression_jax(X, y, offset=0):
    """
    driver function takes a matrix of covariates and fits
    a univariate regression to each
    @param X an n x p matrix
    @param y a binary vector of length n
    @param offset (lenght 1 or n) of offsets (log-odds scale)
    """
    # convert X to dense if sparse
    if sparse.issparse(X):
        X = X.toarray()

    # compute null model (same for all x!)
    ybar = np.mean(y)
    beta0 = np.log(ybar) - jnp.log(1 - ybar)
    beta0 = np.array([beta0, 0.0])
    
    # fit null model to account for offet
    null_fit = mle(np.array([0.0, 0.0]), X[:, 0], y, penalty=1e20, offset=offset)
    beta0[0] = null_fit['intercept']

    # compute mle for each x, initialized with null model
    mle1 = lambda x, y: mle(beta0, x, y, offset=offset)
    vmap_mle = vmap(mle1, (0, None), 0)
    betahat = vmap_mle(X.T, y)
    
    # add statistics
    null_loglik = loglik(beta0, X[:, 0], y, offset=offset)
    betahat['lrts'] = -2 * (null_loglik - betahat['loglik'])
    betahat['pval'] = 1 - stats.chi2.cdf(betahat['lrts'], df=1)
    
    # report as datafram
    df = pd.DataFrame(betahat)
    return df