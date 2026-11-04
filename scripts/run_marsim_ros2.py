"""Run and record one isolated ROS 2 evaluation, then stop only its own processes."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stop(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ideal", "dynamic"), required=True)
    parser.add_argument("--time-mode", choices=("wall", "sim"), default="wall")
    parser.add_argument("--benchmark", choices=("high_speed", "dense"), required=True)
    parser.add_argument("--policy", choices=("super", "ipc", "acmpc"), default="super")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--executor-parameters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--domain", type=int, default=124)
    parser.add_argument("--render-prefix", default="")
    parser.add_argument("--official", action="store_true", help="Run the original mission_planner ideal launch")
    args = parser.parse_args()
    if args.official and (args.mode != "ideal" or args.policy != "super"):
        parser.error("--official uses the upstream ideal SUPER executor; use the unified launch for dynamic")
    if args.time_mode == "sim" and (args.mode != "dynamic" or args.official):
        parser.error("--time-mode sim requires the unified dynamic Flightmare executor")
    if args.policy == "acmpc" and args.checkpoint is None:
        parser.error("--policy acmpc requires --checkpoint")
    if args.executor_parameters and args.mode != "dynamic":
        parser.error("--executor-parameters applies to the dynamic executor")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOG_DIR=str(output / "ros"))
    environment.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "SYSTEM_DEFAULT")
    if args.official:
        command = shlex.split(args.render_prefix) + [
            "ros2", "launch", "mission_planner", f"benchmark_{args.benchmark}.launch.py"]
    else:
        command = ["ros2", "launch", "isaaclab_marsim", "benchmark.launch.py",
                   f"execution:={args.mode}", f"benchmark:={args.benchmark}", f"policy:={args.policy}",
                   f"time_mode:={args.time_mode}",
                   f"output:={output}", "record:=false", f"render_prefix:={args.render_prefix}"]
        if args.checkpoint:
            command.append(f"checkpoint:={args.checkpoint.absolute()}")
        if args.executor_parameters:
            command.append(f"executor_parameters:={args.executor_parameters.absolute()}")
    monitor_command = [sys.executable, str(ROOT / "scripts/evaluate_marsim_ros2.py"),
                       "--mode", args.mode, "--benchmark", args.benchmark, "--policy", args.policy,
                       "--output", str(output), "--timeout", str(args.timeout)]
    monitor_command += ["--time-mode", args.time_mode]
    binaries = [ROOT / "build/marsim_ros2/install" / package / "lib" / package / executable
                for package, executable in (("isaaclab_marsim", "flightmare_node"),
                                            ("isaaclab_marsim", "marsim_lidar_node"),
                                            ("perfect_drone_sim", "perfect_drone_node"),
                                            ("super_planner", "fsm_node"))]
    (output / "run.json").write_text(json.dumps({
        "launch": command, "monitor": monitor_command, "ros_domain_id": args.domain,
        "time_mode": args.time_mode,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "executable_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in binaries if path.exists()},
        "executor_parameters_sha256": hashlib.sha256(args.executor_parameters.read_bytes()).hexdigest()
        if args.executor_parameters else None,
        "middleware_environment": {name: environment.get(name) for name in (
            "RMW_IMPLEMENTATION", "ROS_AUTOMATIC_DISCOVERY_RANGE", "FASTDDS_BUILTIN_TRANSPORTS",
            "FASTRTPS_DEFAULT_PROFILES_FILE")},
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("ros2/isaaclab_marsim", "src/isaaclab_ipc", "scripts")
            for path in sorted((ROOT / directory).rglob("*"))
            if path.is_file() and path.suffix in (".py", ".sh", ".cpp", ".hpp", ".xml", ".txt")
        },
    }, indent=2) + "\n")
    launcher = monitor = None
    started_wall = time.time()
    try:
        with (output / "launch.log").open("w") as launch_log, (output / "monitor.log").open("w") as monitor_log:
            monitor = subprocess.Popen(monitor_command, stdout=monitor_log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            launcher = subprocess.Popen(command, stdout=launch_log, stderr=subprocess.STDOUT,
                                        env=environment, start_new_session=True)
            while monitor.poll() is None and launcher.poll() is None:
                time.sleep(0.1)
            if monitor.poll() is None:
                stop(monitor)
                return 4
            return monitor.returncode
    finally:
        stop(monitor)
        stop(launcher)
        super_timing = ROOT / "build/marsim_ros2/src/SUPER/super_planner/log/time_consuming.csv"
        if args.policy == "super" and super_timing.exists() and super_timing.stat().st_mtime >= started_wall:
            shutil.copy2(super_timing, output / "super_time_consuming.csv")
        if (output / "metrics.json").exists():
            print((output / "metrics.json").read_text(), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
