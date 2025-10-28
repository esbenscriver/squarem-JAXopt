import jax
import jax.numpy as jnp
from jax import random

from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

maxiter = 1_000

N = 10

a = random.uniform(random.PRNGKey(111), (N, ))


def fun(xy: jax.Array) -> jax.Array:
    x_old, y_old = xy[0, :], xy[1, :]
    x_new = (1 - a) * x_old + a * jnp.cos(y_old)
    y_new = (1 - a) * y_old + a * jnp.sin(x_old)
    return jnp.stack([x_new, y_new])  # Keep x unchanged


initial_guess = random.uniform(random.PRNGKey(112), (2, N))

fxp_none = FixedPointIteration(fixed_point_fun=fun, maxiter=maxiter, verbose=False)
result_none = fxp_none.run(initial_guess)
print("FixedPointIteration converged in", result_none.state.iter_num, "iterations.")

fxp_anderson = AndersonAcceleration(fixed_point_fun=fun, maxiter=maxiter, verbose=True)
result_anderson = fxp_anderson.run(initial_guess)

fxp_squarem = SquaremAcceleration(fixed_point_fun=fun, maxiter=maxiter, verbose=True)
result_squarem = fxp_squarem.run(initial_guess)
