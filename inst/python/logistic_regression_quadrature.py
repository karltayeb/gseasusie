# To impliment quadrature all we need is a function that computes the 
# log likelihood at the set of sample points then
# E[p(y | x, b, b0)] \approx \sum p(y | x, ,b_i, b0_i) w_i

import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.special import logsumexp, logit
from jax.lax import while_loop
from jax.nn import sigmoid
from jax.scipy.optimize import minimize

from jax.scipy import optimize
def conditional_log_likelihood_jax(x, y, offset, b, b0):
    """
    computes \sum logp(y_i | sigmoid(b*x_i + b0))
    x, y can be arbitrary shape but must have same shape
    b, b0 are scalar parameters
    """
    logits = b0 + x * b + offset
    ll = logits * y - jnp.log(1 + jnp.exp(logits))
    return jnp.sum(ll)

def scale_nodes(nodes, mu, sigma):
    """
    assuming nodes are for N(0, 1)
    rescale quadrature points for N(mu, sigma^2)
    """
    return sigma * nodes + mu

####
# Option 1: fixed b0
####

# integrate over b
conditional_log_likelihood_vec_b_jax = vmap(
    conditional_log_likelihood_jax,
    (None, None, None, 0, None), 0)

def _logreg_quad_fixed_intercept_jax(x, y, offset, b, b_weights, b0):
    conditional = conditional_log_likelihood_vec_b_jax(x, y, offset, b, b0)
    marginal = logsumexp(conditional + jnp.log(b_weights))
    return marginal

@jit
def logreg_quad_fixed_intercept_jax(x, y, offset, b_mu, b_sigma, b0, nodes, weights):
    b_nodes = scale_nodes(nodes, b_mu, b_sigma)
    return _logreg_quad_fixed_intercept_jax(x, y, offset, b_nodes, weights, b0)

#####
# Option 2: \int max_{b0} p(y, b | b0)
#####

def b0_fixed_point_iteration(val):
    b0, x, sumy, b, offset, delta, iter = val
    psi = b*x + b0 + offset
    p = sigmoid(psi)
    g = sumy - jnp.sum(p)
    h = - jnp.sum(p * (1 - p))
    b0_new = b0 - g/h
    delta = jnp.abs((b0_new - b0)/b0)
    return (b0, x, sumy, b, offset, delta, iter + 1)

def b0_converged(val):
    return val[-2] > 1e-4

@jit
def best_conditional_log_likelihood_jax(x, y, offset, b):
    # step 1: optimize b0 by simple fixed point method
    b0_init = logit(jnp.mean(y))
    val_init = (b0_init, x, jnp.sum(y), b, offset, 1, 0)
    val_out = while_loop(b0_converged, b0_fixed_point_iteration, val_init)
    b0 = val_out[0]

    # step 2: return coniditional log likelihood for optimal b0
    res = conditional_log_likelihood_jax(x, y, b, b0, offset) 
    return dict(b0=b0, loglik=res)

# alternatively, use jax.scipy.optimize.minimize....
def b0_loss(b0_arr, x, y, offset, b):
    return -1 * conditional_log_likelihood_jax(x, y, offset, b, b0_arr[0])

@jit
def best_conditional_log_likelihood2_jax(x, y, offset, b):
    b0_init = jnp.array([logit(jnp.mean(y))])
    opt = minimize(b0_loss, b0_init, args=(x, y, offset, b), method='BFGS')
    return dict(b0=opt.x, loglik=-opt.fun)

best_conditional_log_likelihood_jax_b_vec = vmap(
    best_conditional_log_likelihood2_jax,
    (None, None, None, 0), {'b0': 0, 'loglik': 0}
)

def _logreg_quad_best_intercept_jax(x, y, offset, b, b_weights):
    conditional = best_conditional_log_likelihood_jax_b_vec(x, y, offset, b)['loglik']
    marginal = logsumexp(conditional + jnp.log(b_weights))
    return marginal

@jit
def logreg_quad_best_intercept_jax(x, y, offset, b_mu, b_sigma, nodes, weights):
    """
    x, y 1d data
    compute E[p(y | b, b0, x)]
    b ~ N(b_mu, b_sigma^2)

    nodes and weights for Gauss-Hermite quadrature
    E_{N(0, 1)}[f(x)] \approx \sum_i weights[i] * f(nodes[i])
    """
    b_nodes = scale_nodes(nodes, b_mu, b_sigma)
    logp = _logreg_quad_best_intercept_jax(x, y, offset, b_nodes, weights)
    return logp


####
# Option 3: 2d quadrature
####

# integrate over b0
logreg_quad_fixed_intercept_vec_b0_jax = vmap(
    _logreg_quad_fixed_intercept_jax, 
    (None, None, None, None, None, 0), 0
)

def _logreg_quad_jax(x, y, offset, b, b_weights, b0, b0_weights):
    conditional = logreg_quad_fixed_intercept_vec_b0_jax(x, y, offset, b, b_weights, b0)
    marginal = logsumexp(conditional + jnp.log(b0_weights))
    return marginal

@jit
def logreg_quad_jax(x, y, offset, b_mu, b_sigma, b0_mu, b0_sigma, nodes, weights):
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
    logp = _logreg_quad_jax(x, y, offset, b_nodes, weights, b0_nodes, weights)
    return logp


# # Examples/Tests
import numpy as np
X = np.random.normal(size=10000).reshape(2000, -1)
x = X[:, 1]
y = np.random.binomial(1, 1/(1 + np.exp(-x)))

best_conditional_log_likelihood_jax(x, y, 0, 1)
best_conditional_log_likelihood2_jax(x, y, 0, 1)
# res = logreg_quad_X_jax(X, y, 0, 1, 0, 1, nodes, weights)

# conditional_log_likelihood_jax(x, y, 1, 0)
# conditional_log_likelihood_vec_b_jax(x, y, np.array([-1, 0, 1]), 0)

# nodes, weights = np.polynomial.hermite_e.hermegauss(64)

# logreg_quad_fixed_intercept_jax(x, y, nodes, weights, 0)
# logreg_quad_jax(x, y, 0, 0.000001, 0, 1000, nodes, weights)



# sigmas = (np.arange(10) + 1) / 100

# %timeit logreg_quad_sigma_jax(x, y, 0, sigmas, 0, 1, nodes, weights)
# %timeit logreg_quad_sigma_jax_jit(x, y, 0, sigmas, 0, 1, nodes, weights)

# res = logreg_quad_sigma_jax(x, y, 0, np.array([0.1, 1., 10, 100]), 0, 1, nodes, weights)