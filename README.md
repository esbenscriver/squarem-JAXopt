
[![PyPI version](https://img.shields.io/pypi/v/squarem-JAXopt.svg)](https://pypi.org/project/squarem-JAXopt/)
[![CI](https://github.com/esbenscriver/squarem-JAXopt/actions/workflows/ci.yml/badge.svg)](https://github.com/esbenscriver/squarem-JAXopt/actions/workflows/ci.yml)
[![CD](https://github.com/esbenscriver/squarem-JAXopt/actions/workflows/cd.yml/badge.svg)](https://github.com/esbenscriver/squarem-JAXopt/actions/workflows/cd.yml)
# squarem-JAXopt
[JAX](https://github.com/jax-ml/jax) implementation of the SQUAREM accelerator for solving fixed-point equations, originally proposed by [Du and Varadhan (2020)](https://doi.org/10.18637/jss.v092.i07). The SQUAREM accelerator is implemented using [JAXopt](https://github.com/google/jaxopt), enabling efficient automatic differentiation of the fixed-point equations via the implicit function theorem (see [Blondel et al., 2022](https://arxiv.org/abs/2105.15183) for details).

## Installation

```bash
pip install squarem-jaxopt
```

## Usage

```python

import jax
import jax.numpy as jnp
from jax import random

from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

N = 4

a = random.uniform(random.PRNGKey(111), (N, 1))
b = random.uniform(random.PRNGKey(112), (1, 1))


def fun(x: jax.Array) -> jax.Array:
    y = a + x @ b
    return y


fxp_none = FixedPointIteration(fixed_point_fun=fun, verbose=True)
result_none = fxp_none.run(jnp.zeros_like(a))

fxp_anderson = AndersonAcceleration(fixed_point_fun=fun, verbose=True)
result_anderson = fxp_anderson.run(jnp.zeros_like(a))

fxp_squarem = SquaremAcceleration(fixed_point_fun=fun, verbose=True)
result_squarem = fxp_squarem.run(jnp.zeros_like(a))
```