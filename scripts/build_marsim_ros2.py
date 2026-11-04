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
    for name in ("src/super_core/fsm.cpp", "src/super_core/super_planner.cpp",
                 "include/ros_interface/ros2/ros2_adapter.hpp"):
        cpp = target / "super_planner" / name
        write_changed(cpp, re.sub(r"(?<![\w:])isnan\(", "std::isnan(", cpp.read_text()))
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
    mission_text = mission_text.replace(
        "goal_pub_ = nh_->create_publisher<geometry_msgs::msg::PoseStamped>(cfg_.goal_pub_topic, qos);",
        "goal_pub_ = nh_->create_publisher<geometry_msgs::msg::PoseStamped>(\n"
        "                cfg_.goal_pub_topic, rclcpp::QoS(1).reliable().transient_local());")
    write_changed(mission, mission_text)
    fsm = target / "super_planner/include/ros_interface/ros2/fsm_ros2.hpp"
    write_changed(fsm, fsm.read_text().replace(
        "cfg_.click_goal_topic,\n                        qos,",
        'cfg_.click_goal_topic,\n                        cfg_.click_goal_topic == "/planning/click_goal"\n'
        "                            ? rclcpp::QoS(1).reliable().transient_local() : qos,"))
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
    args = parser.parse_args()
    if not args.upstream_only and args.flightmare_root is None:
        parser.error("--flightmare-root is required unless --upstream-only is selected")
    workspace = ROOT / "build/marsim_ros2"
    source = workspace / "src/SUPER"
    revision = prepare_super(args.super_root.resolve(), source)
    provenance = {"super_revision": revision, "reference": str(args.super_root.resolve()), "source": str(source)}
    flightmare = workspace / "flightmare"
    if args.flightmare_root:
        subprocess.run([sys.executable, str(ROOT / "ros2/isaaclab_marsim/prepare_flightmare.py"),
                        "--source", str(args.flightmare_root.resolve()), "--target", str(flightmare)], check=True)
        provenance["flightmare"] = json.loads((flightmare / "isaaclab_marsim_source.json").read_text())
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
