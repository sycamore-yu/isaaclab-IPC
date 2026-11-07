"""Standard Isaac Lab task discovery."""

import gymnasium as gym

gym.register(
    id="Isaac-IPC-Avoid-Direct",
    entry_point="isaaclab_ipc.tasks.env:IPCEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_ipc.tasks.env:make_cfg"},
)

gym.register(
    id="Isaac-IPC-Forest-Direct",
    entry_point="isaaclab_ipc.tasks.env:IPCEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_ipc.tasks.env:make_forest_cfg"},
)

for scenario in ("Avoid", "Forest"):
    gym.register(
        id=f"Isaac-IPC-{scenario}-ACMPC-Direct",
        entry_point="isaaclab_ipc.tasks.mpc_env:IPCMPCEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_ipc.tasks.mpc_env:make_{'forest_' if scenario == 'Forest' else ''}mpc_cfg",
            "rsl_rl_cfg_entry_point": "isaaclab_ipc.rl.config:MPCTrainCfg",
        },
    )

    gym.register(
        id=f"Isaac-IPC-{scenario}-ACMPC-CTBR-Direct",
        entry_point="isaaclab_ipc.tasks.ctbr_env:CTBRMPCEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_ipc.tasks.ctbr_env:make_{'forest_' if scenario == 'Forest' else 'avoid_'}ctbr_cfg",
            "rsl_rl_cfg_entry_point": "isaaclab_ipc.rl.config:CTBRTrainCfg",
        },
    )

for name, factory in (("CTBR", "make_ctbr_cfg"), ("PathCover-Avoid", "make_avoid_ctbr_cfg"), ("PathCover-Forest", "make_forest_ctbr_cfg")):
    gym.register(
        id=f"Isaac-ACMPC-{name}-Direct",
        entry_point="isaaclab_ipc.tasks.ctbr_env:CTBRMPCEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_ipc.tasks.ctbr_env:{factory}",
            "rsl_rl_cfg_entry_point": "isaaclab_ipc.rl.config:CTBRTrainCfg",
        },
    )

for name, factory in (("HighSpeed", "high_speed"), ("Dense", "dense")):
    gym.register(
        id=f"Isaac-IPC-SUPER-{name}-ACMPC-Direct",
        entry_point="isaaclab_ipc.tasks.super_env:SuperMPCEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_ipc.tasks.super_env:make_{factory}_cfg",
            "rsl_rl_cfg_entry_point": f"isaaclab_ipc.tasks.super_env:make_{factory}_agent_cfg",
        },
    )
