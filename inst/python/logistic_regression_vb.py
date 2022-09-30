import jax.numpy as jnp
import jax
from jax import vmap

def polya_gamma_mean(b, c):
    """
    Mean of PG(b, c)
    """
    # TODO: PG(b, 0) = b/4, deal with divide by 0
    # return b/c * (sigmoid(c) - 0.5)  # equivalent 
    return 0.5 * b/c * jnp.tanh(c/2)

def normal_kl(mu, var, mu0=0, var0=1):
    return 0.5 * (jnp.log(var0) - jnp.log(var) + var/var0 + (mu - mu0)**2/var0 - 1)

def update_intercept(x, y, mu, tau, xi, delta, tau0):
    kappa = y - 0.5
    xb = x * mu
    omega = polya_gamma_mean(1, xi)
    delta = jnp.sum(kappa - xb * omega) / jnp.sum(omega)
    return delta

def update_b(x, y, mu, tau, xi, delta, tau0):
    omega = polya_gamma_mean(1, xi)
    kappa = y - 0.5
    tau = jnp.sum(omega * x**2) + tau0
    nu = jnp.sum((kappa - omega*delta) * x)
    return dict(mu = nu/tau, tau=tau)

def update_xi(x, y, mu, tau, xi, delta, tau0):
    xb2 = x**2 * (mu**2 + 1/tau) + 2*x*mu*delta + delta**2
    return jnp.sqrt(xb2)

def compute_elbo(x, y, mu, tau, xi, delta, tau0):
    kappa = y - 0.5
    xb = (x * mu) + delta
    bound = jnp.log(jax.nn.sigmoid(xi)) + (kappa * xb) - (0.5 * xi)
    kl = normal_kl(mu, 1/tau, 0, 1/tau0)
    return jnp.sum(bound) - kl


def fit_univariate_vb(x, y, mu, tau, xi, delta, tau0):    
    elbos = jnp.array([compute_elbo(x, y, mu, tau, xi, delta, tau0)])

    for i in range(100):
        delta = update_intercept(x, y, mu, tau, xi, delta, tau0)
        
        b = update_b(x, y, mu, tau, xi, delta, tau0)
        mu = b['mu']
        tau = b['tau']
        
        xi = update_xi(x, y, mu, tau, xi, delta, tau0)
        elbo = compute_elbo(x, y, mu, tau, xi, delta, tau0)

    return dict(mu=mu, tau=tau, xi=xi, delta=delta, tau0=tau0, elbo=elbo)


univariate_vb_vec_jax = vmap( 
    fit_univariate_vb, 
    in_axes=(1, None, 0, 0, 1, 0, 0),
    out_axes={'mu': 0, 'tau': 0, 'xi': 0, 'delta': 0, 'tau0': 0, 'elbo': 0}
)


def marginal_vb_jax(X, y, tau0):
    n, p = X.shape
    mu_init = np.zeros(p)
    tau_init = np.ones(p)
    xi_init = np.ones((n, p)) * 1e-3
    delta_init = np.zeros(p)
    tau0 = np.ones(p)
    return univariate_vb_vec_jax(X, y, mu_init, tau_init, xi_init, delta_init, tau0) 
