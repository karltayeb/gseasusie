import jax.numpy as jnp
import numpy as np

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def tilted_bound(data: dict, params: dict):
    tmp = params['mu'] + 0.5 * (1 - 2 * params['xi']) * params['var']
    bound = data['y'] * params['mu'] \
        - 0.5 * params['xi']**2 * params['var'] \
        + jnp.log(sigmoid(tmp))
    return bound

def xi_fixed_point(data: dict, params: dict):
    tmp = params['mu'] + 0.5 * (1 - 2 * params['xi']) * params['var']
    xi = sigmoid(tmp)
    return xi