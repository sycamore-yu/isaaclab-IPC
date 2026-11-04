"""Stage pinned Flightmare dynamics and record the required source-copy fixes."""

import argparse
import difflib
import json
import subprocess
from pathlib import Path

REVISION = "d4218aedac18cbe9364a0a0df10ab992c4b65e4f"


def prepare(source: Path, target: Path) -> Path:
    """Return a private, patched source tree; never write to the reference."""
    source, target = source.resolve(), target.resolve()
    if target == source or source in target.parents:
        raise ValueError("Flightmare target must be outside its reference repository")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"Expected Flightmare {REVISION}, found {actual}")
    if not target.exists():
        target.mkdir(parents=True)
        archive = subprocess.Popen(["git", "-C", str(source), "archive", REVISION], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(target)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError("Flightmare git archive failed")

    patches = []
    base = target / "flightlib"

    def replace(relative, before, after):
        path = base / relative
        original = path.read_text()
        if after in original:
            return
        if original.count(before) != 1:
            raise ValueError(f"Source patch no longer matches: {relative}: {before[:60]}")
        updated = original.replace(before, after, 1)
        patches.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                           f"a/flightlib/{relative}", f"b/flightlib/{relative}"))
        path.write_text(updated)

    # CTBR is acceleration, while clampThrust accepts a single rotor force.
    replace("src/objects/quadrotor.cpp", "dynamics_.clampThrust(cmd_.collective_thrust);",
            "std::clamp(cmd_.collective_thrust, dynamics_.collective_thrust_min()/dynamics_.getMass(), "
            "dynamics_.collective_thrust_max()/dynamics_.getMass());")
    # Upstream normalizes the old quaternion immediately before overwriting it.
    replace("src/objects/quadrotor.cpp", "state_.qx /= state_.qx.norm();",
            "next_state.qx /= next_state.qx.norm();")
    # Independent rotor clipping can spend all authority on yaw and cancel roll/pitch.
    # Preserve feasible allocations exactly; otherwise preserve collective thrust,
    # then roll/pitch direction, and allocate only the remaining authority to yaw.
    replace("src/objects/quadrotor.cpp",
            "  const Vector<4> motor_thrusts_des = B_allocation_inv_ * thrust_and_torque;\n\n"
            "  return dynamics_.clampThrust(motor_thrusts_des);",
            "  const Vector<4> motor_thrusts_des = B_allocation_inv_ * thrust_and_torque;\n"
            "  const Scalar lo = dynamics_.collective_thrust_min() / 4;\n"
            "  const Scalar hi = dynamics_.collective_thrust_max() / 4;\n"
            "  if ((motor_thrusts_des.array() >= lo).all() &&\n"
            "      (motor_thrusts_des.array() <= hi).all()) return motor_thrusts_des;\n\n"
            "  // Saturation priority: collective, roll/pitch, then yaw.\n"
            "  Vector<4> thrust = Vector<4>::Constant(dynamics_.clampThrust(force / 4));\n"
            "  const Vector<4> roll_pitch = B_allocation_inv_ *\n"
            "    Vector<4>(0, body_torque_des.x(), body_torque_des.y(), 0);\n"
            "  const Vector<4> yaw = B_allocation_inv_ *\n"
            "    Vector<4>(0, 0, 0, body_torque_des.z());\n"
            "  const auto add_feasible = [&](const Vector<4>& differential) {\n"
            "    Scalar scale = 1;\n"
            "    for (int i = 0; i < 4; ++i) {\n"
            "      if (differential[i] > 0)\n"
            "        scale = std::min(scale, (hi - thrust[i]) / differential[i]);\n"
            "      else if (differential[i] < 0)\n"
            "        scale = std::min(scale, (lo - thrust[i]) / differential[i]);\n"
            "    }\n"
            "    thrust += std::max(Scalar(0), scale) * differential;\n"
            "  };\n"
            "  add_feasible(roll_pitch);\n"
            "  add_feasible(yaw);\n"
            "  return dynamics_.clampThrust(thrust);")
    replace("include/flightlib/dynamics/quadrotor_dynamics.hpp", "  bool setMass(const Scalar mass);",
            "  bool setInertiaDiagonal(const Vector<3>& diagonal) {\n"
            "    if (!diagonal.allFinite() || (diagonal.array() <= 0).any()) return false;\n"
            "    J_ = diagonal.asDiagonal(); J_inv_ = J_.inverse(); return true;\n"
            "  }\n  bool setMass(const Scalar mass);")
    replace("src/dynamics/quadrotor_dynamics.cpp",
            "    // update relevant variables\n    updateInertiaMarix();",
            "    // Refresh the rotor force bound after loading its speed and thrust map.\n"
            "    thrust_max_ = motorOmegaToThrust(Vector<4>::Constant(motor_omega_max_))[0];\n"
            "    // update relevant variables\n    updateInertiaMarix();")
    replace("include/flightlib/objects/quadrotor.hpp", "  // low-level controller\n",
            "  void setBodyrateGains(const Vector<3>& gains) { Kinv_ang_vel_tau_ = gains.asDiagonal(); }\n"
            "  void resetMotorThrusts(const Vector<4>& force) {\n"
            "    motor_thrusts_ = dynamics_.clampThrust(force);\n"
            "    motor_omega_ = dynamics_.motorThrustToOmega(motor_thrusts_);\n"
            "  }\n  // low-level controller\n")
    replace("include/flightlib/objects/quadrotor.hpp", "  const Matrix<3, 3> Kinv_ang_vel_tau_ =",
            "  Matrix<3, 3> Kinv_ang_vel_tau_ =")
    if patches:
        with (target / "isaaclab_marsim.patch").open("a") as stream:
            stream.writelines(patches)
    (target / "isaaclab_marsim_source.json").write_text(json.dumps({
        "source": str(source), "revision": REVISION, "license": "MIT",
        "patch": "isaaclab_marsim.patch",
    }, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.target))


if __name__ == "__main__":
    main()
