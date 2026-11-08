"""Build an isolated ROS 2 SUPER/MARSIM overlay without editing reference repositories."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPER_REVISION = "2ad3419c127a617c6d7df6925e81a14175a9c096"
PACKAGES = ("mission_planner", "rog_map", "super_planner", "mars_uav_sim/mars_quadrotor_msgs",
            "mars_uav_sim/marsim_render", "mars_uav_sim/perfect_drone_sim")


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def prepare_super(source: Path, target: Path):
    """Use upstream ROS 2 build templates in a private, pinned source snapshot."""
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SUPER_REVISION:
        raise ValueError(f"Expected SUPER {SUPER_REVISION}; found {revision}")
    if not target.exists():
        target.mkdir(parents=True)
        archive = subprocess.Popen(["git", "-C", str(source), "archive", revision], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(target)], stdin=archive.stdout, check=True)
        if archive.wait():
            raise RuntimeError("SUPER git archive failed")
    # Empty upstream log directories are absent from git archive.
    for name in ("cmd_logs", "replan_logs"):
        (target / "super_planner/log" / name).mkdir(parents=True, exist_ok=True)
    for package in PACKAGES:
        directory = target / package
        cmake = (directory / "ros/ros2.CMakeLists.txt").read_text()
        # The upstream CLion hint overwrites the dependency search path on current ROS 2.
        cmake = "\n".join(line for line in cmake.splitlines() if not line.startswith("set(CMAKE_PREFIX_PATH ")) + "\n"
        if package.endswith("marsim_render"):
            cmake = cmake.replace("find_package(rclcpp REQUIRED)", "find_package(rclcpp REQUIRED)\nfind_package(std_msgs REQUIRED)")
            cmake = cmake.replace("find_package(glfw3 REQUIRED)", "find_package(glfw3 REQUIRED)\nget_target_property(glfw3_INCLUDE_DIRS glfw INTERFACE_INCLUDE_DIRECTORIES)")
        if package.endswith("perfect_drone_sim"):
            cmake = cmake.replace("find_package(OpenCV REQUIRED)", "find_package(OpenCV REQUIRED)\nfind_package(pcl_conversions REQUIRED)")
            cmake = cmake.replace("set(dependencies\n", "set(dependencies\n        pcl_conversions\n")
        write_changed(directory / "CMakeLists.txt", cmake)
        manifest = (directory / "ros/ros2.package.xml").read_text()
        dependencies = {
            "super_planner": ("mars_quadrotor_msgs", "pcl_conversions", "tf2_ros", "visualization_msgs"),
            "mars_uav_sim/perfect_drone_sim": ("pcl_conversions", "tf2_ros", "visualization_msgs"),
            "mars_uav_sim/marsim_render": ("rclcpp", "std_msgs"),
        }
        for dependency in dependencies.get(package, ()):
            if f"<depend>{dependency}</depend>" not in manifest:
                manifest = manifest.replace("    <export>", f"    <depend>{dependency}</depend>\n    <export>")
        write_changed(directory / "package.xml", manifest)
    shader = target / "mars_uav_sim/marsim_render/config/pattern/360camera.vs"
    # w=0 and an unwritten point size can cover the depth buffer with near-zero
    # returns. Clip range-out points explicitly; preserve all valid-point math.
    shader_text = shader.read_text().replace(
        "\t\tgl_Position = vec4(0.0f,0.0f,0.0f,0.0f);\n\t\tourColor = aColor;",
        "\t\tgl_Position = vec4(2.0f,2.0f,2.0f,1.0f);\n"
        "\t\tgl_PointSize = 1.0f;\n\t\tourColor = aColor;", 1)
    write_changed(shader, shader_text)
    for name in ("src/super_core/fsm.cpp", "src/super_core/super_planner.cpp",
                 "include/ros_interface/ros2/ros2_adapter.hpp"):
        cpp = target / "super_planner" / name
        write_changed(cpp, re.sub(r"(?<![\w:])isnan\(", "std::isnan(", cpp.read_text()))
    planner = target / "super_planner/src/super_core/super_planner.cpp"
    # Upstream repeats the backup solve for debugging and discards both outputs.
    # Keep the first solve and its logged initial conditions.
    duplicate_backup = """            Trajectory traj;
            double out_ts;
            back_traj_opt_->optimize(ref_exp_traj.posTraj(),
                                     t0,
                                     te,
                                     init_ts,
                                     sfc0,
                                     init_times,
                                     init_ps,
                                     traj,
                                     out_ts
            );
"""
    planner_text = planner.read_text().replace(
        "        auto sfc0 = back_traj_info.getSFC();\n", "").replace(duplicate_backup, "")
    if "constexpr double rest_speed_limit = 0.05;" not in planner_text:
        planner_text = planner_text.replace(
            "        latest_replan.reset();\n        latest_replan.setGoal(goal_p, goal_yaw, robot_state_);\n"
            "        if (robot_state_.rcv == false) {",
            "        robot_state_ = map_ptr_->getRobotState();\n"
            "        latest_replan.reset();\n        latest_replan.setGoal(goal_p, goal_yaw, robot_state_);\n"
            "        if (!robot_state_.rcv || ros_ptr_->getSimTime() - robot_state_.rcv_time > 0.1) {")
        planner_text = planner_text.replace(
            "        gi_.goal_p = goal_p;",
            """        // At 100 Hz, 0.05 m/s moves 0.5 mm per command (5 mm in 100 ms).
        constexpr double rest_speed_limit = 0.05;
        if (!robot_state_.v.allFinite() || !robot_state_.p.allFinite() ||
            robot_state_.v.norm() > rest_speed_limit) {
            ros_ptr_->warn(" -- [SUPER] PlanFromRest waiting for measured speed <= 0.05 m/s.");
            return FAILED;
        }
        const Vec3f rest_position = robot_state_.p;
        gi_.goal_p = goal_p;""", 1)
        planner_text = planner_text.replace(
            "        if (replan_total_t > cfg_.replan_forward_dt) {",
            "        if (!last_exp_traj_info.empty() && replan_total_t > cfg_.replan_forward_dt) {")
        planner_text = planner_text.replace(
            "        back_traj_info.setEmpty();\n"
            "        RET_CODE back_ret_code = generateBackupTrajectory(exp_traj_info, back_traj_info);;",
            """        // The vehicle has held its rest position while Exp and yaw were solved.
        exp_traj_info.setTrajectory(ros_ptr_->getSimTime(),
                                    exp_traj_info.posTraj(), exp_traj_info.yawTraj());
        back_traj_info.setEmpty();
        RET_CODE back_ret_code = generateBackupTrajectory(exp_traj_info, back_traj_info);

        if (back_ret_code == SUCCESS || back_ret_code == FINISH || back_ret_code == NO_NEED) {
            const auto commit_state = map_ptr_->getRobotState();
            const double start_WT = ros_ptr_->getSimTime();
            // Allow at most one tenth of a map cell of motion during the solve.
            if (!commit_state.rcv || start_WT - commit_state.rcv_time > 0.1 ||
                !commit_state.v.allFinite() || !commit_state.p.allFinite() ||
                commit_state.v.norm() > rest_speed_limit ||
                (commit_state.p - rest_position).norm() > 0.1 * cfg_.resolution) {
                ros_ptr_->warn(" -- [SUPER] PlanFromRest moved during planning; wait for rest and retry.");
                return FAILED;
            }
            robot_state_ = commit_state;
            // Shift both absolute origins; preserve every polynomial and relative join time.
            exp_traj_info.setTrajectory(start_WT,
                                        exp_traj_info.posTraj(), exp_traj_info.yawTraj());
            latest_replan.setExpTraj(exp_traj_info.posTraj());
            latest_replan.setExpYawTraj(exp_traj_info.yawTraj());
            if (back_ret_code == SUCCESS) {
                const double start_TT = back_traj_info.getStartTT();
                back_traj_info.setTrajectory(start_WT + start_TT, start_TT,
                                             back_traj_info.posTraj(), back_traj_info.yawTraj());
                latest_replan.setBackupTraj(back_traj_info.posTraj());
                latest_replan.setBackupYawTraj(back_traj_info.yawTraj());
            }
        }""")
    write_changed(planner, planner_text)
    rog = target / "rog_map/include/rog_map_ros/rog_map_ros2.hpp"
    rog_text = rog.read_text()
    if "mutable std::mutex robot_state_mutex_;" not in rog_text:
        rog_text = rog_text.replace("#include <rclcpp/rclcpp.hpp>", "#include <mutex>\n#include <rclcpp/rclcpp.hpp>")
        rog_text = rog_text.replace(
            "        rclcpp::Node::SharedPtr nh_;",
            "        rclcpp::Node::SharedPtr nh_;\n        mutable std::mutex robot_state_mutex_;")
        rog_text = rog_text.replace(
            "            updateRobotState(std::make_pair(Vec3f(odom_msg->pose.pose.position.x,",
            """            {
                std::lock_guard<std::mutex> guard(robot_state_mutex_);
                // SUPER's /lidar_slam/odom contract supplies world-frame velocity.
                robot_state_.v = Vec3f(odom_msg->twist.twist.linear.x,
                                      odom_msg->twist.twist.linear.y,
                                      odom_msg->twist.twist.linear.z);
                updateRobotState(std::make_pair(Vec3f(odom_msg->pose.pose.position.x,""")
        rog_text = rog_text.replace(
            "                                                  odom_msg->pose.pose.orientation.z)));",
            "                                                  odom_msg->pose.pose.orientation.z)));\n            }")
        rog_text = rog_text.replace(
            "    public:\n",
            """    public:
        RobotState getRobotState() const {
            std::lock_guard<std::mutex> guard(robot_state_mutex_);
            return ROGMap::getRobotState();
        }
""", 1)
    write_changed(rog, rog_text)
    quickhull = target / "super_planner/include/utils/geometry/quickhull.h"
    if "#include <cstdint>" not in quickhull.read_text():
        write_changed(quickhull, quickhull.read_text().replace("#include <cmath>", "#include <cmath>\n#include <cstdint>"))
    mission = target / "mission_planner/include/waypoint_mission/ros2_waypoint_planner.hpp"
    pending_goal = "            if (new_goal || cur_t - last_pub_time > cfg_.publish_dt) {\n                new_goal = false;"
    ready_goal = ("            if (new_goal || cur_t - last_pub_time > cfg_.publish_dt) {\n"
                  "                // Keep the pending goal until a consumer has been discovered.\n"
                  "                if (goal_pub_->get_subscription_count() == 0) return;\n"
                  "                new_goal = false;")
    mission_text = mission.read_text().replace(pending_goal, ready_goal)
    if 'nh_->get_parameter("simulated_time").as_bool()' not in mission_text:
        mission_text = mission_text.replace(
            'nh_->declare_parameter<std::string>("data_name", dft_data_name);',
            'nh_->declare_parameter<std::string>("data_name", dft_data_name);\n'
            '            nh_->declare_parameter<bool>("simulated_time", false);')
        mission_text = mission_text.replace(
            'goal_pub_timer_ = nh_->create_wall_timer(',
            'goal_pub_timer_ = nh_->get_parameter("simulated_time").as_bool()\n'
            '                ? nh_->create_timer(std::chrono::milliseconds(10),\n'
            '                    std::bind(&WaypointPlanner::GoalPubTimerCallback, this), goal_pub_timer_cb_group_)\n'
            '                : nh_->create_wall_timer(')
    mission_text = mission_text.replace(
        'goal_pub_timer_cbk_group_', 'goal_pub_timer_cb_group_')
    mission_text = mission_text.replace(
        "goal_pub_ = nh_->create_publisher<geometry_msgs::msg::PoseStamped>(cfg_.goal_pub_topic, qos);",
        "goal_pub_ = nh_->create_publisher<geometry_msgs::msg::PoseStamped>(\n"
        "                cfg_.goal_pub_topic, rclcpp::QoS(1).reliable().transient_local());")
    write_changed(mission, mission_text)
    fsm = target / "super_planner/include/ros_interface/ros2/fsm_ros2.hpp"
    fsm_text = fsm.read_text().replace(
        "cfg_.click_goal_topic,\n                        qos,",
        'cfg_.click_goal_topic,\n                        cfg_.click_goal_topic == "/planning/click_goal"\n'
        "                            ? rclcpp::QoS(1).reliable().transient_local() : qos,")
    if '#include <std_msgs/msg/u_int64.hpp>' not in fsm_text:
        fsm_text = fsm_text.replace(
            '#include <rclcpp/rclcpp.hpp>',
            '#include <rclcpp/rclcpp.hpp>\n#include <std_msgs/msg/u_int64.hpp>')
    if '"/marsim/sim_policy_done"' not in fsm_text:
        fsm_text = fsm_text.replace(
            'rclcpp::TimerBase::SharedPtr execution_timer_, replan_timer_, cmd_timer_;',
            'rclcpp::TimerBase::SharedPtr execution_timer_, replan_timer_, cmd_timer_;\n'
            '        rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr sim_tick_sub_;\n'
            '        rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr sim_done_pub_;\n'
            '        double next_sim_replan_time_{0.0};')
        fsm_text = fsm_text.replace(
            '            if (cfg_.timer_en) {',
            '''            if (nh_->get_parameter("simulated_time").as_bool()) {
                sim_done_pub_ = nh_->create_publisher<std_msgs::msg::UInt64>(
                    "/marsim/sim_policy_done", 10);
                sim_tick_sub_ = nh_->create_subscription<std_msgs::msg::UInt64>(
                    "/marsim/sim_tick", rclcpp::QoS(10).reliable(),
                    [this](std_msgs::msg::UInt64::ConstSharedPtr tick) {
                        const auto expected_ns = static_cast<int64_t>(tick->data) * 10000000LL;
                        if (nh_->now().nanoseconds() != expected_ns) {
                            RCLCPP_ERROR(nh_->get_logger(),
                                "SUPER simulation clock mismatch at step %lu", tick->data);
                            return;
                        }
                        // Run the whole planning/control update before acknowledging
                        // a simulation step. No wall-clock time elapses for physics.
                        callMainFsmOnce();
                        const double t = nh_->now().seconds();
                        if (machine_state_ == FOLLOW_TRAJ) {
                            if (next_sim_replan_time_ == 0.0) {
                                next_sim_replan_time_ = t + 1.0 / cfg_.replan_rate;
                            } else if (t + 1e-9 >= next_sim_replan_time_) {
                                next_sim_replan_time_ += 1.0 / cfg_.replan_rate;
                                callReplanOnce();
                            }
                        } else {
                            next_sim_replan_time_ = 0.0;
                        }
                        const bool command = machine_state_ == FOLLOW_TRAJ ||
                                             machine_state_ == EMER_STOP;
                        pubCmdTimerCallback();
                        std_msgs::msg::UInt64 done;
                        done.data = (tick->data << 1) | static_cast<uint64_t>(command);
                        sim_done_pub_->publish(done);
                    });
            } else if (cfg_.timer_en) {''', 1)
    write_changed(fsm, fsm_text)
    entry = target / "super_planner/Apps/fsm_node_ros2.cpp"
    entry_text = entry.read_text().replace(
        "rclcpp::executors::MultiThreadedExecutor executor;",
        'const int executor_threads = node->declare_parameter<int>("executor_threads", 4);\n'
        '    RCLCPP_INFO(node->get_logger(), "Executor threads: %d", executor_threads);\n'
        "    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), executor_threads);")
    if 'if (!node->declare_parameter<bool>("simulated_time", false))' not in entry_text:
        entry_text = entry_text.replace(
            '    // 检查是否使用仿真时间\n    while (rclcpp::ok()) {',
            '    // Keep upstream wall-clock mode; preserve /clock in lockstep mode.\n'
            '    if (!node->declare_parameter<bool>("simulated_time", false)) {\n'
            '    while (rclcpp::ok()) {')
        entry_text = entry_text.replace(
            '    // 初始化FSM', '    }\n\n    // 初始化FSM')
    write_changed(entry, entry_text)
    perfect_entry = target / "mars_uav_sim/perfect_drone_sim/src/ros2_perfect_drone_node.cpp"
    perfect_text = perfect_entry.read_text()
    if "catch (const std::runtime_error& error)" not in perfect_text:
        perfect_text = perfect_text.replace(
            '#include "rclcpp/rclcpp.hpp"', '#include "rclcpp/rclcpp.hpp"\n#include <stdexcept>\n#include <string>')
        perfect_text = perfect_text.replace(
            "    rclcpp::spin(node);",
            """    try {
        rclcpp::spin(node);
    } catch (const std::runtime_error& error) {
        // SIGINT can invalidate the context during the global-map subscriber query.
        if (rclcpp::ok(node->get_node_base_interface()->get_context()) ||
            std::string(error.what()).rfind(
                "could not count subscribers: rcl node's context is invalid,", 0) != 0) {
            throw;
        }
    }""")
    write_changed(perfect_entry, perfect_text)
    # The README spells this correctly, while the upstream filename contains a typo.
    launch = target / "mission_planner/launch"
    shutil.copy2(launch / "banchmark_high_speed.launch.py", launch / "benchmark_high_speed.launch.py")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--super-root", type=Path, required=True)
    parser.add_argument("--flightmare-root", type=Path)
    parser.add_argument("--deps-prefix", type=Path, action="append", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--upstream-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--packages", nargs="+", help="Build only these packages after their dependencies exist")
    parser.add_argument("--workspace", type=Path, default=ROOT / "build/marsim_ros2",
                        help="Independent ROS source/build/install directory")
    args = parser.parse_args()
    if not args.upstream_only and args.flightmare_root is None:
        parser.error("--flightmare-root is required unless --upstream-only is selected")
    workspace = args.workspace.resolve()
    source = workspace / "src/SUPER"
    revision = prepare_super(args.super_root.resolve(), source)
    provenance = {"super_revision": revision, "reference": str(args.super_root.resolve()), "source": str(source)}
    flightmare = workspace / "flightmare"
    if args.flightmare_root and not args.upstream_only:
        subprocess.run([sys.executable, str(ROOT / "ros2/isaaclab_marsim/prepare_flightmare.py"),
                        "--source", str(args.flightmare_root.resolve()), "--target", str(flightmare)], check=True)
    flightmare_source = flightmare / "isaaclab_marsim_source.json"
    if flightmare_source.exists():
        provenance["flightmare"] = json.loads(flightmare_source.read_text())
    (workspace / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.prepare_only:
        print(source)
        return
    environment = os.environ.copy()
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    environment["MAKEFLAGS"] = f"-j{args.jobs}"
    if args.deps_prefix:
        prefixes = [str(path.resolve()) for path in args.deps_prefix]
        environment["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefixes + [environment.get("CMAKE_PREFIX_PATH", "")])
        ros_prefixes = [prefix for prefix in prefixes if (Path(prefix) / "share/ament_index").exists()]
        environment["AMENT_PREFIX_PATH"] = os.pathsep.join(ros_prefixes + [environment.get("AMENT_PREFIX_PATH", "")])
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(path.resolve() / "lib") for path in args.deps_prefix]
            + [str(path.resolve() / "lib/x86_64-linux-gnu") for path in args.deps_prefix]
            + [environment.get("LD_LIBRARY_PATH", "")])
    command = ["colcon", "--log-base", str(workspace / "log"), "build", "--executor", "sequential",
               "--base-paths", str(source)]
    if not args.upstream_only:
        command.append(str(ROOT / "ros2/isaaclab_marsim"))
    if args.packages:
        command += ["--packages-select", *args.packages]
    command += ["--build-base", str(workspace / "build"), "--install-base", str(workspace / "install"),
                "--symlink-install", "--cmake-args", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
                "-DPython3_EXECUTABLE=/usr/bin/python3", f"-DSUPER_ROOT={source}"]
    if args.flightmare_root:
        command.append(f"-DFLIGHTMARE_ROOT={flightmare}")
    for prefix in args.deps_prefix:
        glfw = prefix.resolve() / "lib/x86_64-linux-gnu/cmake/glfw3"
        if glfw.exists():
            command.append(f"-Dglfw3_DIR={glfw}")
    subprocess.run(command, cwd=workspace, env=environment, check=True)
    print(f"source {workspace / 'install/setup.bash'}")


if __name__ == "__main__":
    main()
