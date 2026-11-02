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
