# Copyright (c) 2026, Isaac Lab Diff Flight Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Four-motor asset adapted from isaaclab_mbrl_flight/assets/quadrotor.py.

Dimensions and dynamics are explicit reconstruction values.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg

from . import ASSETS_DIR

ARM = 0.2 / math.sqrt(2.0)
MASS = 0.75
INERTIA = 0.012
MAX_THRUST = 3.0 * MASS * 9.81 / 4.0

QUADROTOR_CFG = MultirotorCfg(
    class_type="isaaclab_ipc.assets.newton_multirotor:NewtonMultirotor",
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS_DIR / "quadrotor.usda")),
    init_state=MultirotorCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0), rps={".*": 150.0}),
    actuators={
        "motors": ThrusterCfg(
            thruster_names_expr=["rotor_0", "rotor_1", "rotor_2", "rotor_3"],
            dt=1.0 / 300.0,
            thrust_range=(0.0, MAX_THRUST),
            thrust_const_range=(8.175e-5, 8.175e-5),
            tau_inc_range=(0.01, 0.01),
            tau_dec_range=(0.01, 0.01),
            torque_to_thrust_ratio=0.01,
            integration_scheme="euler",
            use_discrete_approximation=True,
        )
    },
    allocation_matrix=(
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (ARM, ARM, -ARM, -ARM),
        (-ARM, ARM, ARM, -ARM),
        (0.01, -0.01, 0.01, -0.01),
    ),
    rotor_directions=(1, -1, 1, -1),
)
