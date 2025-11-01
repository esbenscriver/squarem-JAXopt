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

fxp_none = FixedPointIteration(fixed_point_fun=fun, verbose=False)
result_none = fxp_none.run(initial_guess)

fxp_anderson = AndersonAcceleration(fixed_point_fun=fun, verbose=False)
result_anderson = fxp_anderson.run(initial_guess)

fxp_squarem = SquaremAcceleration(fixed_point_fun=fun, verbose=False)
result_squarem = fxp_squarem.run(initial_guess)

# Create comparison table
print("\n" + "="*60)
print("ALGORITHM COMPARISON TABLE")
print("="*60)
print(f"{'Algorithm':<25} {'Iterations':<12} {'Func Evals':<12} {'Error':<12}")
print("-"*60)
print(f"{'FixedPointIteration':<25} {result_none.state.iter_num:<12} {result_none.state.num_fun_eval:<12} {result_none.state.error:<12.2e}")
print(f"{'AndersonAcceleration':<25} {result_anderson.state.iter_num:<12} {result_anderson.state.num_fun_eval:<12} {result_anderson.state.error:<12.2e}")
print(f"{'SquaremAcceleration':<25} {result_squarem.state.iter_num:<12} {result_squarem.state.num_fun_eval:<12} {result_squarem.state.error:<12.2e}")
print("="*60)
