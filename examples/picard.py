import jax
import jax.numpy as jnp
from jax import random

from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

N = 100_000

a = random.uniform(random.PRNGKey(111), (N, 1))


def fun(x: jax.Array) -> jax.Array:
    y = (1 - a) + a * jnp.cos(x)
    return y


initial_guess = jnp.zeros_like(a)

fxp_none = FixedPointIteration(fixed_point_fun=fun, verbose=True)
result_none = fxp_none.run(initial_guess)

fxp_anderson = AndersonAcceleration(fixed_point_fun=fun, verbose=True)
result_anderson = fxp_anderson.run(initial_guess)

fxp_squarem = SquaremAcceleration(fixed_point_fun=fun, verbose=True)
result_squarem = fxp_squarem.run(initial_guess)
