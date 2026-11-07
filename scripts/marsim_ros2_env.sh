#!/usr/bin/env bash
# Source this file before launching the isolated MARSIM evaluation overlay.
export IPC_MARSIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export IPC_MARSIM_RUNTIME="${IPC_MARSIM_RUNTIME:-$IPC_MARSIM_ROOT/build/marsim_ros2}"
export TURBOMPC_NATIVE_BUILD_DIR="${TURBOMPC_NATIVE_BUILD_DIR:-$IPC_MARSIM_ROOT/build/turbompc/native/cu130}"
export IPC_MARSIM_PYTHON="$IPC_MARSIM_ROOT/.venv/bin/python"
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
IPC_MARSIM_DEPS="$IPC_MARSIM_RUNTIME/deps/root"
export AMENT_PREFIX_PATH="$IPC_MARSIM_DEPS/opt/ros/jazzy${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export LD_LIBRARY_PATH="$IPC_MARSIM_DEPS/usr/lib/x86_64-linux-gnu:$IPC_MARSIM_DEPS/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$IPC_MARSIM_ROOT/src:$IPC_MARSIM_DEPS/opt/ros/jazzy/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SYSTEM_DEFAULT}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$IPC_MARSIM_ROOT/ros2/isaaclab_marsim/config/fastdds_local.xml}"
source "$IPC_MARSIM_RUNTIME/install/setup.bash"
