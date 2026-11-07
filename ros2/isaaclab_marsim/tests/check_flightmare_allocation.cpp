// Copyright (c) 2026, Isaac Lab IPC Contributors.
// SPDX-License-Identifier: BSD-3-Clause

// Link against the prepared flightlib_dynamics used by flightmare_node.
// Check the collective contract and real motor dynamics; this is not a mission.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "flightlib/objects/quadrotor.hpp"

using namespace flightlib;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    constexpr Scalar mass = 0.75, max_rotor = 3 * mass * 9.81 / 4;
    // Scalar is float32. Force tolerance is < 4e-7 of maximum collective.
    constexpr Scalar force_tolerance = 8e-6, torque_tolerance = 1e-5;
    const Scalar coefficient = 8.175e-5 / (4 * std::acos(-1.0) * std::acos(-1.0));
    YAML::Node config;
    auto cfg = config["quadrotor_dynamics"];
    cfg["mass"] = mass;
    cfg["arm_l"] = 0.2;
    cfg["motor_omega_min"] = 0.0;
    cfg["motor_omega_max"] = std::sqrt(max_rotor / coefficient);
    cfg["motor_tau"] = 0.01;
    cfg["thrust_map"] = std::vector<double>{coefficient, 0, 0};
    cfg["kappa"] = 0.01;
    cfg["omega_max"] = std::vector<double>{10, 10, 10};
    QuadrotorDynamics dynamics;
    require(dynamics.updateParams(config), "Invalid dynamics parameters");
    require(dynamics.setInertiaDiagonal(Vector<3>::Constant(0.012)), "Invalid inertia");
    Quadrotor quad(dynamics);
    const Vector<3> gains = Vector<3>::Constant(45);
    quad.setBodyrateGains(gains);
    const Matrix<4, 4> allocation = dynamics.getAllocationMatrix();
    const Matrix<4, 4> inverse = allocation.inverse();
    const Scalar low = dynamics.collective_thrust_min() / 4;
    const Scalar high = dynamics.collective_thrust_max() / 4;
    Scalar max_collective_error = 0, max_feasible_error = 0;
    Scalar max_torque_projection_error = 0, max_zero_torque_error = 0;
    int cases = 0, feasible_cases = 0, saturated_cases = 0, boundary_cases = 0;
    const std::vector<Scalar> rates{-10, -3, -0.1, 0, 0.1, 3, 10};
    for (int level = 0; level <= 10; ++level) {
      const Scalar total = 4 * (low + (high - low) * level / 10);
      for (Scalar wx : rates) for (Scalar wy : rates) for (Scalar wz : rates) {
        const Vector<3> omega(wx, wy, wz);
        const Vector<3> torque = dynamics.getJ() * gains.asDiagonal() * omega;
        const Command command(0, total / mass, omega);
        const Vector<4> requested(mass * command.collective_thrust, torque.x(), torque.y(), torque.z());
        const Vector<4> original = inverse * requested;
        const Vector<4> actual = quad.runFlightCtl(0.0025, Vector<3>::Zero(), command);
        require(actual.allFinite(), "Nonfinite allocated rotor force");
        require(actual.minCoeff() >= low && actual.maxCoeff() <= high,
                "Allocated rotor force outside native limits");
        const Vector<4> achieved = allocation * actual;
        max_collective_error = std::max(max_collective_error, std::abs(achieved[0] - requested[0]));
        require(std::abs(achieved[0] - requested[0]) <= force_tolerance,
                "Saturation changed requested collective thrust");

        if ((original.array() >= low).all() && (original.array() <= high).all()) {
          max_feasible_error = std::max(max_feasible_error, (actual - original).cwiseAbs().maxCoeff());
          ++feasible_cases;
        } else {
          // Maximal uniform torque scaling reaches a rotor bound.
          const Scalar distance = std::min(actual.minCoeff() - low, high - actual.maxCoeff());
          require(distance <= force_tolerance, "Saturated allocation left unused torque authority");
          ++saturated_cases;
        }
        if (torque.squaredNorm() == 0) {
          max_zero_torque_error = std::max(max_zero_torque_error,
              (actual - Vector<4>::Constant(requested[0] / 4)).cwiseAbs().maxCoeff());
        } else {
          const Vector<3> achieved_torque = achieved.tail<3>();
          // Distance to the segment [0, requested torque] uses a physical
          // absolute tolerance, including near-zero float32 torques.
          const Scalar scale = std::clamp(achieved_torque.dot(torque) / torque.squaredNorm(),
                                          Scalar(0), Scalar(1));
          max_torque_projection_error = std::max(max_torque_projection_error,
              (achieved_torque - scale * torque).norm());
        }
        if (level == 0 || level == 10) {
          require(achieved.tail<3>().norm() <= torque_tolerance,
                  "Collective at its physical endpoint created attitude torque");
          ++boundary_cases;
        }
        ++cases;
      }
    }
    require(feasible_cases > 0 && saturated_cases > 0, "Missing feasible/saturated coverage");
    require(max_feasible_error < 1e-6, "A feasible native allocation was changed");
    require(max_torque_projection_error <= torque_tolerance, "Saturation changed torque direction or magnitude");
    require(max_zero_torque_error <= force_tolerance, "Zero torque did not allocate equal forces");

    // Dense V15, command held during (20.99,21.09]. Initial omega is explicitly
    // the preceding 10 ms quaternion-increment equivalent, not an unlogged
    // instantaneous measurement. Final omega below is directly measured.
    const Command frozen(0, 6.288733005523682,
        Vector<3>(-2.8279528617858887, -1.2890262603759766, -1.4852842092514038));
    const Vector<3> initial_omega(1.3661607684611647, -1.6451985889525405, -1.5001112303002195);
    const Vector<3> measured_final_omega(-2.8802878856658936, -1.2888009548187256, -1.4847525358200073);
    const Scalar frozen_total = mass * frozen.collective_thrust;
    const Vector<4> frozen_initial = quad.runFlightCtl(0.0025, initial_omega, frozen);
    const Vector<4> expected(0, 2.1647642535302998, 2.472198434676831, 0.07958706593563036);
    require((frozen_initial - expected).cwiseAbs().maxCoeff() <= force_tolerance,
            "Frozen collective-preserving allocation differs from independent double algebra");
    require(std::abs(frozen_initial.sum() - frozen_total) <= force_tolerance,
            "Frozen initial allocation raised collective thrust");
    const Vector<4> frozen_final = quad.runFlightCtl(0.0025, measured_final_omega, frozen);
    const Vector<3> final_torque = dynamics.getJ() * gains.asDiagonal() * (frozen.omega - measured_final_omega);
    const Vector<4> final_request(frozen_total, final_torque.x(), final_torque.y(), final_torque.z());
    require((frozen_final - inverse * final_request).cwiseAbs().maxCoeff() <= force_tolerance,
            "Frozen feasible final allocation was changed");

    // Confirm that the existing first-order motor lag is still active.
    const Vector<4> old_forces(0.5, 1.0, 1.5, 2.0);
    const Vector<4> motor_target(2.0, 1.5, 1.0, 0.5);
    quad.resetMotorThrusts(old_forces);
    quad.runMotors(0.0025, motor_target);
    Vector<4> motor_actual;
    require(quad.getMotorThrusts(motor_actual), "Cannot read physical rotor forces");
    const Scalar c = std::exp(-0.0025 / 0.01);
    const Vector<4> expected_omega =
        c * dynamics.motorThrustToOmega(old_forces) +
        (1 - c) * dynamics.motorThrustToOmega(motor_target);
    const Vector<4> expected_motor = dynamics.motorOmegaToThrust(expected_omega);
    const Scalar motor_lag_error = (motor_actual - expected_motor).cwiseAbs().maxCoeff();
    require(motor_lag_error <= force_tolerance, "Native motor time constant changed");
    require((motor_actual - motor_target).norm() > 0.1, "Motor lag was bypassed");

    // A constructed initial rotor distribution with the recorded initial sum.
    // This checks the source-backed force bound, not an exact flight replay.
    QuadState state;
    state.setZero();
    state.p.z() = 10;
    state.w = initial_omega;
    require(quad.reset(state), "Cannot initialize dynamic frozen-command test");
    const Scalar initial_total = 4.856537118877812;
    quad.resetMotorThrusts(Vector<4>::Constant(initial_total / 4));
    const Scalar rate_error_before = (state.w - frozen.omega).norm();
    Scalar maximum_target_total = 0, maximum_motor_total = initial_total;
    for (int step = 0; step < 40; ++step) {
      require(quad.getState(&state), "Cannot read pre-step state");
      const Vector<4> target = quad.runFlightCtl(0.0025, state.w, frozen);
      maximum_target_total = std::max(maximum_target_total, target.sum());
      require(std::abs(target.sum() - frozen_total) <= force_tolerance,
              "Dynamic rate feedback raised target collective");
      require(quad.run(frozen, 0.0025), "Native motor/RK4 integration failed");
      require(quad.getMotorThrusts(motor_actual), "Cannot read dynamic rotor forces");
      require(motor_actual.minCoeff() >= low && motor_actual.maxCoeff() <= high,
              "Dynamic rotor force violated physical bounds");
      maximum_motor_total = std::max(maximum_motor_total, motor_actual.sum());
      require(motor_actual.sum() <= std::max(initial_total, frozen_total) + force_tolerance,
              "Motor lag exceeded the no-collective-override force bound");
    }
    require(quad.getState(&state), "Final dynamic state became invalid");
    const Scalar rate_error_after = (state.w - frozen.omega).norm();
    require(rate_error_after < rate_error_before, "Dynamic controller did not reduce rate error");

    std::cout << std::setprecision(10)
              << "{\"cases\":" << cases << ",\"feasible_cases\":" << feasible_cases
              << ",\"saturated_cases\":" << saturated_cases << ",\"boundary_cases\":" << boundary_cases
              << ",\"force_tolerance_N\":" << force_tolerance
              << ",\"torque_tolerance_Nm\":" << torque_tolerance
              << ",\"max_collective_force_error_N\":" << max_collective_error
              << ",\"max_feasible_force_error_N\":" << max_feasible_error
              << ",\"max_torque_projection_error_Nm\":" << max_torque_projection_error
              << ",\"max_zero_torque_force_error_N\":" << max_zero_torque_error
              << ",\"frozen_requested_collective_N\":" << frozen_total
              << ",\"frozen_allocated_rotor_forces_N\":[" << frozen_initial[0] << ',' << frozen_initial[1]
              << ',' << frozen_initial[2] << ',' << frozen_initial[3]
              << "],\"frozen_allocated_collective_N\":" << frozen_initial.sum()
              << ",\"motor_lag_force_error_N\":" << motor_lag_error
              << ",\"constructed_dynamic_max_target_collective_N\":" << maximum_target_total
              << ",\"constructed_dynamic_max_actual_collective_N\":" << maximum_motor_total
              << ",\"constructed_dynamic_initial_rate_error_rad_s\":" << rate_error_before
              << ",\"constructed_dynamic_final_rate_error_rad_s\":" << rate_error_after << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
