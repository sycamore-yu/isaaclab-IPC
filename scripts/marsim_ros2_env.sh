#!/usr/bin/env bash
# Source this file before launching the isolated MARSIM evaluation overlay.
export IPC_MARSIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export IPC_MARSIM_RUNTIME="${IPC_MARSIM_RUNTIME:-$IPC_MARSIM_ROOT/build/marsim_ros2}"
export TURBOMPC_NATIVE_BUILD_DIR="${TURBOMPC_NATIVE_BUILD_DIR:-$IPC_MARSIM_ROOT/build/turbompc/native/cu130}"
export IPC_MARSIM_PYTHON="$IPC_MARSIM_ROOT/.venv/bin/python"
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
# Dependencies belong to the shared base repository, not to an isolated overlay.
_ipc_git_common_dir="$(git -C "$IPC_MARSIM_ROOT" rev-parse --path-format=absolute --git-common-dir)"
_ipc_shared_root="$(dirname "$_ipc_git_common_dir")"
IPC_MARSIM_DEPS="${IPC_MARSIM_DEPS:-$_ipc_shared_root/build/marsim_ros2/deps/root}"
unset _ipc_git_common_dir
export AMENT_PREFIX_PATH="$IPC_MARSIM_DEPS/opt/ros/jazzy${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export LD_LIBRARY_PATH="$IPC_MARSIM_DEPS/usr/lib/x86_64-linux-gnu:$IPC_MARSIM_DEPS/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Native CUDA AC-MPC links cuDSS dynamically. The wheel is in the shared
# project's cu130 environment, not in each worktree's isolated ROS overlay.
_ipc_cudss_lib="$_ipc_shared_root/build/turbompc/envs/cu130/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$_ipc_cudss_lib" ]]; then
    export LD_LIBRARY_PATH="$_ipc_cudss_lib:$LD_LIBRARY_PATH"
fi
unset _ipc_shared_root _ipc_cudss_lib
export PYTHONPATH="$IPC_MARSIM_ROOT/src:$IPC_MARSIM_DEPS/opt/ros/jazzy/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SYSTEM_DEFAULT}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$IPC_MARSIM_ROOT/ros2/isaaclab_marsim/config/fastdds_local.xml}"
source "$IPC_MARSIM_RUNTIME/install/setup.bash"
