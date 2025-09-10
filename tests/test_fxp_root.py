import jax
import jax.numpy as jnp
from jax import random

from jaxopt import AndersonAcceleration, FixedPointIteration
from squarem import SquaremAcceleration

import pytest

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

def choose_solver(solver, fun):
    if solver == "None":
        return FixedPointIteration(fixed_point_fun=fun)
    elif solver == "Anderson":
        return AndersonAcceleration(fixed_point_fun=fun)
    elif solver == "SQUAREM":
        return SquaremAcceleration(fixed_point_fun=fun)
    else:
        raise ValueError(f"Unknown solver: {solver}")


@pytest.mark.parametrize(
    "N, accelerator",
    [
        (1000, "None"),
        (1000, "Anderson"),
        (1000, "SQUAREM"),
    ],
)
def test_solver(N: int, accelerator: str):
    a = random.uniform(random.PRNGKey(111), (N, 1))
    b = random.uniform(random.PRNGKey(112), (1, 1))

    def fun(x: jnp.ndarray) -> jnp.ndarray:
        y = a + x @ b
        return y

    fxp = choose_solver(accelerator, fun)
    result = fxp.run(jnp.zeros_like(a))

    assert jnp.allclose(result.params, fun(result.params)), (
        f"Error: {jnp.linalg.norm(fun(result.params) - result.params)}"
    )
