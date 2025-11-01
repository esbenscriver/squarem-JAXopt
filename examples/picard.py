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

fxp_none = FixedPointIteration(fixed_point_fun=fun, verbose=False)
result_none = fxp_none.run(initial_guess)

fxp_anderson = AndersonAcceleration(fixed_point_fun=fun, verbose=False)
result_anderson = fxp_anderson.run(initial_guess)

fxp_squarem = SquaremAcceleration(fixed_point_fun=fun, verbose=False)
result_squarem = fxp_squarem.run(initial_guess)

print("\n" + "=" * 60)
print("ALGORITHM COMPARISON TABLE")
print("=" * 60)
print(f"{'Algorithm':<25} {'Iterations':<12} {'Func Evals':<12} {'Error':<12}")
print("-" * 60)
print(
    f"{'FixedPointIteration':<25} {result_none.state.iter_num:<12} {result_none.state.num_fun_eval:<12} {result_none.state.error:<12.2e}"
)
print(
    f"{'AndersonAcceleration':<25} {result_anderson.state.iter_num:<12} {result_anderson.state.num_fun_eval:<12} {result_anderson.state.error:<12.2e}"
)
print(
    f"{'SquaremAcceleration':<25} {result_squarem.state.iter_num:<12} {result_squarem.state.num_fun_eval:<12} {result_squarem.state.error:<12.2e}"
)
print("=" * 60)
