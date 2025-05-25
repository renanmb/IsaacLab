# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities to configure an environment for SKRL library.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # or ml_framework="jax"

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env  # for PyTorch, or...
    from skrl.envs.jax.wrappers import wrap_env    # for JAX

    env = wrap_env(env, wrapper="isaaclab")

"""

from .exporter import export_policy_as_jit, export_policy_as_onnx
from .vecenv_wrapper import SkrlVecEnvWrapper
