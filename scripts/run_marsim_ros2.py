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


def renderer_library_sha256(executable, environment):
    """Record the renderer selected by the launch environment's dynamic loader."""
    dependencies = subprocess.check_output(["ldd", str(executable)], env=environment, text=True)
    for line in dependencies.splitlines():
        fields = line.split()
        if fields[:2] == ["libmarsim_render.so", "=>"] and fields[2] != "not":
            library = Path(fields[2]).resolve()
            return {str(library): hashlib.sha256(library.read_bytes()).hexdigest()}
    raise RuntimeError(f"Cannot resolve libmarsim_render.so for {executable}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ideal", "dynamic"), required=True)
    parser.add_argument("--time-mode", choices=("wall", "sim"), default="wall")
    parser.add_argument("--benchmark", choices=("high_speed", "dense"), required=True)
    parser.add_argument("--policy", choices=("super", "ipc", "fixed-ctbr", "acmpc"), default="fixed-ctbr")
    parser.add_argument("--frontend", choices=("pathcover", "none"), default="pathcover")
    parser.add_argument("--reference-speed", type=float, default=3.0,
                        help="Native PathCover nominal speed in m/s; physical scene/CTBR speed limits remain separate")
    parser.add_argument("--sfc-constraint-mode", choices=("polytope", "section", "progress"), default="polytope",
                        help="Full SFC, fixed-clock slabs, or optimized arc-length progress")
    parser.add_argument("--sfc-section-half-length", type=float, default=0.5,
                        help="Requested longitudinal half-length in metres; clipped to the original SFC")
    parser.add_argument("--capture-reference-plans", action="store_true",
                        help="Retain returned CPU plans for offline continuous reference checks; write once at shutdown")
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
    if args.policy in ("fixed-ctbr", "acmpc") and args.mode != "dynamic":
        parser.error("The native CTBR route requires --mode dynamic")
    if args.capture_reference_plans and args.policy not in ("fixed-ctbr", "acmpc"):
        parser.error("--capture-reference-plans requires a native CTBR policy")
    if not 0 < args.reference_speed < float("inf"):
        parser.error("--reference-speed must be finite and positive")
    if not 0 < args.sfc_section_half_length < float("inf"):
        parser.error("--sfc-section-half-length must be finite and positive")
    if args.sfc_constraint_mode in ("section", "progress") and (args.frontend != "pathcover"
                                                 or args.policy not in ("fixed-ctbr", "acmpc")):
        parser.error("Section constraints require native CTBR with the PathCover frontend")
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
        command = ["ros2", "launch", str(ROOT / "ros2/isaaclab_marsim/launch/benchmark.launch.py"),
                   f"execution:={args.mode}", f"benchmark:={args.benchmark}", f"policy:={args.policy}",
                   f"frontend:={args.frontend}",
                   f"reference_speed:={args.reference_speed}",
                   f"sfc_constraint_mode:={args.sfc_constraint_mode}",
                   f"sfc_section_half_length_m:={args.sfc_section_half_length}",
                   f"capture_reference_plans:={'true' if args.capture_reference_plans else 'false'}",
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
    runtime = Path(environment.get("IPC_MARSIM_RUNTIME", ROOT / "build/marsim_ros2")).resolve()
    binaries = [runtime / "install" / package / "lib" / package / executable
                for package, executable in (("isaaclab_marsim", "flightmare_node"),
                                            ("isaaclab_marsim", "marsim_lidar_node"),
                                            ("perfect_drone_sim", "perfect_drone_node"),
                                            ("super_planner", "fsm_node"))]
    native_binaries = list((ROOT / "src/isaaclab_ipc").glob("_pathcover_native*.so"))
    native_binaries.extend(Path(environment.get("TURBOMPC_NATIVE_BUILD_DIR",
        ROOT / "build/turbompc/native/cu130")).glob("_turbompc_native*.so"))
    renderer_executable = binaries[1] if args.mode == "dynamic" else binaries[2]
    runtime_libraries = renderer_library_sha256(renderer_executable, environment)
    (output / "run.json").write_text(json.dumps({
        "launch": command, "monitor": monitor_command, "ros_domain_id": args.domain,
        "time_mode": args.time_mode, "frontend": args.frontend,
        "reference_speed_m_s": args.reference_speed,
        "sfc_constraint_mode": args.sfc_constraint_mode,
        "sfc_section_half_length_m": args.sfc_section_half_length,
        "capture_reference_plans": args.capture_reference_plans,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "marsim_runtime": str(runtime),
        "executable_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in binaries if path.exists()},
        "runtime_library_sha256": runtime_libraries,
        "native_library_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in native_binaries},
        "executor_parameters_sha256": hashlib.sha256(args.executor_parameters.read_bytes()).hexdigest()
        if args.executor_parameters else None,
        "middleware_environment": {name: environment.get(name) for name in (
            "RMW_IMPLEMENTATION", "ROS_AUTOMATIC_DISCOVERY_RANGE", "FASTDDS_BUILTIN_TRANSPORTS",
            "FASTRTPS_DEFAULT_PROFILES_FILE")},
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("ros2/isaaclab_marsim", "src/isaaclab_ipc", "scripts")
            for path in sorted((ROOT / directory).rglob("*"))
            if path.is_file() and path.suffix in (".py", ".sh", ".cpp", ".hpp", ".xml", ".txt", ".yaml", ".yml", ".json")
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
        super_timing = runtime / "src/SUPER/super_planner/log/time_consuming.csv"
        if args.policy == "super" and super_timing.exists() and super_timing.stat().st_mtime >= started_wall:
            shutil.copy2(super_timing, output / "super_time_consuming.csv")
        if (output / "metrics.json").exists():
            print((output / "metrics.json").read_text(), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
