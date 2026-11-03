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
        id=f"Isaac-IPC-{scenario}-TurboMPC-Direct",
        entry_point="isaaclab_ipc.tasks.mpc_env:IPCMPCEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_ipc.tasks.mpc_env:make_{'forest_' if scenario == 'Forest' else ''}mpc_cfg",
            "rsl_rl_cfg_entry_point": "isaaclab_ipc.rl.config:TurboMPCTrainCfg",
        },
    )

    gym.register(
        id=f"Isaac-IPC-{scenario}-TurboMPC-CTBR-Direct",
        entry_point="isaaclab_ipc.tasks.ctbr_env:CTBRMPCEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_ipc.tasks.ctbr_env:make_{'forest_' if scenario == 'Forest' else ''}ctbr_cfg",
            "rsl_rl_cfg_entry_point": "isaaclab_ipc.rl.config:CTBRTrainCfg",
        },
    )
