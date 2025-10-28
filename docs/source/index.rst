squarem-jaxopt
==============

**squarem-jaxopt** is a JAX implementation of the SQUAREM accelerator for solving fixed-point equations,
originally proposed by `Du and Varadhan (2020) <https://doi.org/10.18637/jss.v092.i07>`_.

The SQUAREM accelerator is implemented using `JAXopt <https://github.com/google/jaxopt>`_, enabling
efficient automatic differentiation of the fixed-point equations via the implicit function theorem.

Installation
------------

.. code-block:: bash

   pip install squarem-jaxopt

Quick Start
-----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax import random
   from squarem_jaxopt import SquaremAcceleration

   # Enable 64-bit precision
   jax.config.update("jax_enable_x64", True)

   # Define a fixed-point function
   def fun(x: jax.Array) -> jax.Array:
       a = random.uniform(random.PRNGKey(111), (4, 1))
       b = random.uniform(random.PRNGKey(112), (1, 1))
       return a + x @ b

   # Create and run the solver
   fxp_squarem = SquaremAcceleration(fixed_point_fun=fun, verbose=True)
   result = fxp_squarem.run(jnp.zeros((4, 1)))

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

