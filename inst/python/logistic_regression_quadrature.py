# To impliment quadrature all we need is a function that computes the 
# log likelihood at the set of sample points then
# E[p(y | x, b, b0)] \approx \sum p(y | x, ,b_i, b0_i) w_i

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

def conditional_log_likelihood(x, y, b, b0):
    """
    computes \sum logp(y_i | sigmoid(b*x_i + b0))
    x, y can be arbitrary shape but must have same shape
    b, b0 are scalar parameters
    """
    logits = b0 + x * b
    ll = logits * y - jnp.log(1 + jnp.exp(logits))
    return jnp.sum(ll)


# integrate over b
conditional_log_likelihood_vec_b= vmap(
    conditional_log_likelihood,
    (None, None, 0, None), 0)

def logreg_quad_fixed_intercept(x, y, b, b_weights, b0):
    conditional = conditional_log_likelihood_vec_b(x, y, b, b0)
    marginal = logsumexp(conditional + jnp.log(b_weights))
    return marginal

# integrate over b0
logreg_quad_fixed_intercept_vec_b0 = vmap(
    logreg_quad_fixed_intercept, 
    (None, None, None, None, 0), 0
)
def _logreg_quad(x, y, b, b_weights, b0, b0_weights):
    conditional = logreg_quad_fixed_intercept_vec_b0(x, y, b, b_weights, b0)
    marginal = logsumexp(conditional + jnp.log(b0_weights))
    return marginal


def scale_nodes(nodes, mu, sigma):
    """
    assuming nodes are for N(0, 1)
    rescale quadrature points for N(mu, sigma^2)
    """
    return sigma * nodes + mu

def logreg_quad(x, y, b_mu, b_sigma, b0_mu, b0_sigma, nodes, weights):
    """
    x, y 1d data
    compute E[p(y | b, b0, x)]
    b ~ N(b_mu, b_sigma^2)
    b0 ~ N(b0_mu, b0_sigma^2)

    nodes and weights for Gauss-Hermite quadrature
    E_{N(0, 1)}[f(x)] \approx \sum_i weights[i] * f(nodes[i])
    """
    b_nodes = scale_nodes(nodes, b_mu, b_sigma)
    b0_nodes = scale_nodes(nodes, b0_mu, b0_sigma)
    logp = _logreg_quad(x, y, b_nodes, weights, b0_nodes, weights)
    return logp


logreg_quad_X = vmap(
    logreg_quad,
    (1, None, None, None, None, None, None, None),
    0
)


import numpy as np
X = np.random.normal(size=1000000).reshape(2000, -1)
x = X[:, 1]
y = np.random.binomial(1, 1/(1 + np.exp(-x)))

conditional_log_likelihood(x, y, 1, 0)
conditional_log_likelihood_vec_b(x, y, np.array([-1, 0, 1]), 0)

nodes, weights = np.polynomial.hermite_e.hermegauss(64)

logreg_quad_fixed_intercept(x, y, nodes, weights, 0)
_logreg_quad(x, y, nodes, weights, nodes, weights)
logreg_quad(x, y, 0, 0.000001, 0, 1000, nodes, weights)

res = logreg_quad_X(X, y, 0, 1, 0, 1, nodes, weights)
