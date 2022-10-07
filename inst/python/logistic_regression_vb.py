import jax.numpy as jnp
import jax
from jax import vmap, jit
from jax.lax import while_loop
import numpy as np

def polya_gamma_mean(b, c):
    """
    Mean of PG(b, c)
    """
    # TODO: PG(b, 0) = b/4, deal with divide by 0
    # return b/c * (sigmoid(c) - 0.5)  # equivalent 
    return 0.5 * b/c * jnp.tanh(c/2)


def normal_kl(mu, var, mu0=0, var0=1):
    return 0.5 * (jnp.log(var0) - jnp.log(var) + var/var0 + (mu - mu0)**2/var0 - 1)


def update_intercept(x, y, mu, tau, xi, delta, tau0, offset):
    kappa = y - 0.5
    xb = x * mu + offset
    omega = polya_gamma_mean(1, xi)
    delta = jnp.sum(kappa - xb * omega) / jnp.sum(omega)
    return delta


def update_b(x, y, mu, tau, xi, delta, tau0, offset):
    omega = polya_gamma_mean(1, xi)
    kappa = y - 0.5
    tau = jnp.sum(omega * x**2) + tau0
    nu = jnp.sum((kappa - omega * (delta + offset)) * x)
    return dict(mu = nu/tau, tau=tau)


def update_xi(x, y, mu, tau, xi, delta, tau0, offset):
    xb2 = x**2 * (mu**2 + 1/tau) + 2*x*mu*(delta + offset) + (delta + offset)**2
    return jnp.sqrt(xb2)


def compute_elbo(x, y, mu, tau, xi, delta, tau0, offset):
    kappa = y - 0.5
    xb = (x * mu) + delta + offset
    bound = jnp.log(jax.nn.sigmoid(xi)) + (kappa * xb) - (0.5 * xi)
    kl = normal_kl(mu, 1/tau, 0, 1/tau0)
    return jnp.sum(bound) - kl


def vb_iter(val):
    # unpack
    x, y, mu, tau, xi, delta, tau0, offset, elbo, diff, iter = val

    # do updates
    delta = update_intercept(x, y, mu, tau, xi, delta, tau0, offset)

    b = update_b(x, y, mu, tau, xi, delta, tau0, offset)
    mu = b['mu']
    tau = b['tau']

    xi = update_xi(x, y, mu, tau, xi, delta, tau0, offset)
    elbo_new = compute_elbo(x, y, mu, tau, xi, delta, tau0, offset)

    diff = elbo_new - elbo

    # pack up the results
    iter = iter + 1
    val = (x, y, mu, tau, xi, delta, tau0, offset, elbo_new, diff, iter)
    print(elbo)
    return val


def fit_univariate_vb(x, y, mu, tau, xi, delta, tau0, offset):    
    elbo = compute_elbo(x, y, mu, tau, xi, delta, tau0, offset)
    val_init = (
        x, y,
        jnp.array(mu), jnp.array(tau), jnp.array(xi), jnp.array(delta),
        jnp.array(tau0), jnp.array(offset), elbo,
        jnp.array(1e10), 0)

    def vb_cond(val):
        return val[-2] > 1e-6

    res = while_loop(vb_cond, vb_iter, val_init)

    return dict(
        mu=res[2],
        tau=res[3],
        xi=res[4],
        delta=res[5],
        tau0=res[6],
        elbo=res[8],
        diff=res[9],
        iter=res[10]
    )


def fit_univariate_vb_debug(x, y, mu, tau, xi, delta, tau0, offset):    
    elbo = compute_elbo(x, y, mu, tau, xi, delta, tau0, offset)
    val_init = (
        x, y,
        jnp.array(mu), jnp.array(tau), jnp.array(xi), jnp.array(delta),
        jnp.array(tau0), jnp.array(offset), elbo,
        jnp.array(1e10), 0)

    res = val_init
    for i in range(10):
        res = vb_iter(res)

    return dict(
        mu=res[2],
        tau=res[3],
        xi=res[4],
        delta=res[5],
        tau0=res[6],
        elbo=res[8],
        diff=res[9],
        iter=res[10]
    )

univariate_vb_vec_jax = jit(vmap( 
    fit_univariate_vb, 
    in_axes=(1, None, 0, 0, 1, 0, 0, None),  # y and offset are not mapped over
    out_axes={'mu': 0, 'tau': 0, 'xi': 0, 'delta': 0, 'tau0': 0, 'elbo': 0, 'diff': 0, 'iter': 0}
))


def marginal_vb_jax(X, y, tau0, offset=None):
    n, p = X.shape
    mu_init = np.zeros(p)
    tau_init = np.ones(p)
    xi_init = np.ones((n, p)) * 1e-3
    delta_init = np.zeros(p)
    tau0 = np.ones(p)
    if offset is None:
        offset = np.zeros(n)
    res = univariate_vb_vec_jax(X, y, mu_init, tau_init, xi_init, delta_init, tau0, offset)
    res = {k: np.array(v) for k, v in res.items()}  # convert to numpy for use in R
    return res

# import numpy as np
# x = np.random.normal(size= 1000)
# X = np.random.normal(size = (1000 ,50))
# X[:, 1] = x
# y = np.random.binomial(1, 1/(1+np.exp(1-x)))
# offset = np.zeros(1000)
# mu = 0
# tau = 1
# xi = np.ones(1000)
# delta = 1
# tau0 = 1

# res_X = marginal_vb_jax(X, y, 1., offset=offset + 1)
# res_x = fit_univariate_vb(x, y, 0, 1, xi, delta, 1, offset + 1)

# res3 = fit_univariate_vb_debug(x, y, 0, 1, xi, delta, 1, x, offset)
# elbo = compute_elbo(x, y, mu, tau, xi, delta, tau0)

# val_init = (x, y,
#     jnp.array(mu), jnp.array(tau), jnp.array(xi), jnp.array(delta),
#     jnp.array(tau0), elbo, jnp.array(1e10), 0
# )
# res = while_loop(vb_cond, vb_iter, val_init)

# val = vb_iter(val_init)

# res = fit_univariate_vb(x, y, 0., 1., xi, 0., 1.)
# res2 = marginal_vb_jax(X, y, 1.) 
# res['elbo'], res['diff'], res['iter']