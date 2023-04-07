from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
import numpy as np

def univariate_bayesian_lr(x, y, scale=1.):
    beta = numpyro.sample('mu', dist.Normal(0., scale))
    delta = numpyro.param('delta', 0.)

    logits = beta * x + delta
    numpyro.sample('y', dist.BernoulliLogits(logits), obs=y)


n = 1000
delta = -1
beta = 1
x = np.random.normal(size=n)

logits = beta*x + delta
y = np.random.binomial(1, 1/(1+np.exp(-logits)))

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

kernel = NUTS(univariate_bayesian_lr)
num_samples = 2000
mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
mcmc.run(
    rng_key_, x=x, y=y
)