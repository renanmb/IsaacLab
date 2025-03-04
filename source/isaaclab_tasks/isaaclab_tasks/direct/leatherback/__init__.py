# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .leatherback_env import *

from .leatherback_go_through_poses import LeatherbackGoThroughPosesEnv, LeatherbackGoThroughPosesEnvCfg
from .leatherback_go_through_positions import LeatherbackGoThroughPositionsEnv, LeatherbackGoThroughPositionsEnvCfg
from .leatherback_go_to_pose import LeatherbackGoToPoseEnv, LeatherbackGoToPoseEnvCfg
from .leatherback_go_to_position import LeatherbackGoToPositionEnv, LeatherbackGoToPositionEnvCfg
from .leatherback_push_block import LeatherbackPushBlockEnv, LeatherbackPushBlockEnvCfg
from .leatherback_race_waypoints import LeatherbackRaceWaypointsEnv, LeatherbackRaceWaypointsEnvCfg
from .leatherback_race_wayposes import LeatherbackRaceWayposesEnv, LeatherbackRaceWayposesEnvCfg
from .leatherback_track_velocities import LeatherbackTrackVelocitiesEnv, LeatherbackTrackVelocitiesEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Leatherback-Direct-v0",
    entry_point=f"{__name__}.leatherback_env:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_env:LeatherbackEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
