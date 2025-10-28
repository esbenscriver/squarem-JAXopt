import jax
import jax.numpy as jnp

from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)


def fun(xy: jax.Array) -> jax.Array:
    x_old, y_old = xy[0], xy[1]
    return jnp.stack([jnp.sin(y_old), jnp.cos(x_old)])


initial_guess = jnp.zeros((2,))

fxp_none = FixedPointIteration(fixed_point_fun=fun, verbose=True)
result_none = fxp_none.run(initial_guess)

fxp_anderson = AndersonAcceleration(fixed_point_fun=fun, verbose=True)
result_anderson = fxp_anderson.run(initial_guess)

fxp_squarem = SquaremAcceleration(fixed_point_fun=fun, verbose=True)
result_squarem = fxp_squarem.run(initial_guess)
