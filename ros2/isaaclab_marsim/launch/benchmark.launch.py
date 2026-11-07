"""Select one MARSIM executor and one navigation policy for the official SUPER task."""

import os
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    execution, policy, benchmark = value("execution"), value("policy"), value("benchmark")
    simulated = value("time_mode") == "sim"
    if simulated and execution != "dynamic":
        raise ValueError("Lockstep sim mode currently requires Flightmare dynamic execution")
    root = Path(__file__).resolve().parents[3]
    python = value("python")
    output = str(Path(value("output")).expanduser().absolute())
    simulation = Path(get_package_share_directory("perfect_drone_sim"))
    config = str(simulation / "config" / f"{benchmark}.yaml")
    render_prefix = value("render_prefix")
    actions = []
    lidar = None

    if execution == "ideal":
        executor = Node(package="perfect_drone_sim", executable="perfect_drone_node", output="log",
                        parameters=[{"config_name": f"{benchmark}.yaml"}], prefix=render_prefix)
        actions.append(executor)
    else:
        executor_parameters = [value("executor_parameters")] if value("executor_parameters") else []
        executor = Node(package="isaaclab_marsim", executable="flightmare_node", output="screen",
                        parameters=executor_parameters + [{"config_path": config,
                                     "simulated_time": simulated, "use_sim_time": simulated,
                                     "command_source": "position_command" if policy == "super" else "ctbr"}])
        lidar = Node(package="isaaclab_marsim", executable="marsim_lidar_node", output="log",
                     parameters=[{"config_path": config, "odom_topic": "/marsim/odometry",
                                  "simulated_time": simulated, "use_sim_time": simulated}], prefix=render_prefix)
        actions += [executor, lidar]

    if value("mission") == "true":
        actions.append(Node(package="mission_planner", executable="waypoint_mission", output="log",
                            parameters=[{"config_name": "waypoint.yaml", "data_name": "benchmark.txt",
                                         "simulated_time": simulated, "use_sim_time": simulated}]))
    if policy == "super":
        controller = Node(package="super_planner", executable="fsm_node", output="log",
                          parameters=[{"config_name": f"static_{benchmark}.yaml",
                                       "simulated_time": simulated, "use_sim_time": simulated}])
    else:
        if policy == "acmpc" and not value("checkpoint"):
            raise ValueError("policy:=acmpc requires checkpoint:=/absolute/path/model.pt")
        native_ctbr = policy in ("fixed-ctbr", "acmpc")
        policy_command = [
            python, "-m", "isaaclab_ipc.ros_ctbr_policy" if native_ctbr else "isaaclab_ipc.ros_policy", "--ros-args",
            "-p", f"policy:={policy if native_ctbr else 'fixed'}",
            "-p", f"execution:={execution}", "-p", f"benchmark:={benchmark}",
            "-p", f"output_directory:={output}/policy",
            "-p", f"simulated_time:={'true' if simulated else 'false'}",
            "-p", f"use_sim_time:={'true' if simulated else 'false'}",
        ]
        if native_ctbr:
            policy_command += ["-p", f"frontend:={value('frontend')}",
                               "-p", f"reference_speed:={float(value('reference_speed'))}"]
        if value("checkpoint"):
            policy_command += ["-p", f"checkpoint:={value('checkpoint')}"]
        controller = ExecuteProcess(cmd=policy_command, output="screen")
    actions.append(controller)
    if simulated:
        driver = ExecuteProcess(
            cmd=[python, "-m", "isaaclab_ipc.ros_sim_clock"],
            output="screen",
        )
        actions.append(driver)
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=driver, on_exit=[EmitEvent(event=Shutdown(reason="Lockstep driver exited"))])))

    if value("record") == "true":
        monitor = ExecuteProcess(cmd=[
            python, str(root / "scripts/evaluate_marsim_ros2.py"),
            "--mode", execution, "--benchmark", benchmark, "--policy", policy,
            "--output", output, "--timeout", value("timeout"),
            "--time-mode", value("time_mode"),
        ], output="screen")
        actions += [monitor, RegisterEventHandler(OnProcessExit(
            target_action=monitor, on_exit=[EmitEvent(event=Shutdown(reason="Evaluation completed"))]))]
    for process in [executor, controller] + ([lidar] if lidar else []):
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=process, on_exit=[EmitEvent(event=Shutdown(reason="Executor, LiDAR or policy exited"))])))
    if value("rviz") == "true":
        actions.append(Node(package="rviz2", executable="rviz2", output="log", prefix=render_prefix,
                            arguments=["-d", str(simulation / "rviz2/fpv.rviz")]))
    return actions


def generate_launch_description():
    root = Path(__file__).resolve().parents[3]
    return LaunchDescription([
        DeclareLaunchArgument("execution", default_value="ideal", choices=["ideal", "dynamic"]),
        DeclareLaunchArgument("time_mode", default_value="wall", choices=["wall", "sim"]),
        DeclareLaunchArgument("policy", default_value="fixed-ctbr", choices=["super", "ipc", "fixed-ctbr", "acmpc"]),
        DeclareLaunchArgument("frontend", default_value="pathcover", choices=["pathcover", "none"]),
        DeclareLaunchArgument("reference_speed", default_value="3.0",
                              description="Native PathCover nominal reference speed in m/s"),
        DeclareLaunchArgument("benchmark", default_value="high_speed", choices=["high_speed", "dense"]),
        DeclareLaunchArgument("checkpoint", default_value=""),
        DeclareLaunchArgument("executor_parameters", default_value="", description="Optional Flightmare ROS parameter YAML"),
        DeclareLaunchArgument("python", default_value=os.environ.get("IPC_MARSIM_PYTHON", str(root / ".venv/bin/python"))),
        DeclareLaunchArgument("output", default_value=str(root / "results" / ("marsim-" + datetime.now().strftime("%Y%m%d-%H%M%S")))),
        DeclareLaunchArgument("timeout", default_value="120.0"),
        DeclareLaunchArgument("mission", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("record", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("rviz", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("render_prefix", default_value="", description="For headless NVIDIA rendering: vglrun -d egl0"),
        OpaqueFunction(function=start),
    ])
