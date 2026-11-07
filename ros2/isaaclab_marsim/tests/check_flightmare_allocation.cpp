// Copyright (c) 2026, Isaac Lab IPC Contributors.
// SPDX-License-Identifier: BSD-3-Clause

// Link against the prepared flightlib_dynamics used by flightmare_node.
// Check the collective contract and real motor dynamics; this is not a mission.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/LU>

#include "flightlib/objects/quadrotor.hpp"

using namespace flightlib;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct AllocationOracle {
  double scale = -std::numeric_limits<double>::infinity();
  double yaw_lower = 0;
  double yaw_upper = 0;
};

AllocationOracle vertexOracle(const Matrix<4, 4>& allocation,
                              const Vector<4>& requested, double low, double high) {
  // Independent LP vertices in [f0,f1,f2,f3,s]. Three equalities preserve
  // collective and impose proportional RP. Two active variable bounds fix
  // each vertex. This does not use the allocator's affine yaw intersections.
  const auto B = allocation.cast<double>().eval();
  const auto wrench = requested.cast<double>().eval();
  Eigen::Matrix<double, 5, 5> system = Eigen::Matrix<double, 5, 5>::Zero();
  system.topLeftCorner<3, 4>() = B.topRows<3>();
  system(1, 4) = -wrench[1];
  system(2, 4) = -wrench[2];
  Eigen::Matrix<double, 5, 1> rhs = Eigen::Matrix<double, 5, 1>::Zero();
  rhs[0] = wrench[0];
  AllocationOracle answer;
  for (int first = 0; first < 5; ++first) {
    for (int second = first + 1; second < 5; ++second) {
      system.row(3).setZero(); system(3, first) = 1;
      system.row(4).setZero(); system(4, second) = 1;
      const Eigen::FullPivLU<Eigen::Matrix<double, 5, 5>> factor(system);
      if (!factor.isInvertible()) continue;
      for (int first_upper = 0; first_upper < 2; ++first_upper) {
        for (int second_upper = 0; second_upper < 2; ++second_upper) {
          rhs[3] = first == 4 ? first_upper : (first_upper ? high : low);
          rhs[4] = second == 4 ? second_upper : (second_upper ? high : low);
          const auto point = factor.solve(rhs).eval();
          if ((point.head<4>().array() < low - 1e-8).any() ||
              (point.head<4>().array() > high + 1e-8).any() ||
              point[4] < -1e-8 || point[4] > 1 + 1e-8 ||
              (system * point - rhs).cwiseAbs().maxCoeff() > 1e-8) continue;
          const double scale = std::clamp(point[4], 0.0, 1.0);
          const double yaw = B.row(3).dot(point.head<4>());
          if (scale > answer.scale + 1e-9) {
            answer = {scale, yaw, yaw};
          } else if (std::abs(scale - answer.scale) <= 1e-9) {
            answer.yaw_lower = std::min(answer.yaw_lower, yaw);
            answer.yaw_upper = std::max(answer.yaw_upper, yaw);
          }
        }
      }
    }
  }
  require(std::isfinite(answer.scale), "Independent allocation LP has no vertex");
  return answer;
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
    Scalar max_rp_projection_error = 0, max_zero_torque_error = 0;
    double max_rp_optimality_error = 0, max_yaw_optimality_error = 0;
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
          require((actual.array() == original.array()).all(), "Feasible native allocation changed bits");
          ++feasible_cases;
        } else {
          // A lexicographically optimal saturated allocation reaches a rotor bound.
          const Scalar distance = std::min(actual.minCoeff() - low, high - actual.maxCoeff());
          require(distance <= force_tolerance, "Saturated allocation left unused torque authority");
          ++saturated_cases;
        }
        const AllocationOracle oracle = vertexOracle(allocation, requested, low, high);
        const auto rp = torque.head<2>().eval();
        const auto achieved_rp = achieved.segment<2>(1).eval();
        const Scalar scale = rp.squaredNorm() == 0 ? 1 : achieved_rp.dot(rp) / rp.squaredNorm();
        max_rp_projection_error = std::max(max_rp_projection_error, (achieved_rp - scale * rp).norm());
        max_rp_optimality_error = std::max(max_rp_optimality_error, std::abs(double(scale) - oracle.scale));
        const double expected_yaw = std::clamp(double(torque.z()), oracle.yaw_lower, oracle.yaw_upper);
        max_yaw_optimality_error = std::max(max_yaw_optimality_error, std::abs(double(achieved[3]) - expected_yaw));
        if (torque.squaredNorm() == 0) {
          max_zero_torque_error = std::max(max_zero_torque_error,
              (actual - Vector<4>::Constant(requested[0] / 4)).cwiseAbs().maxCoeff());
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
    require(max_rp_projection_error <= torque_tolerance, "Saturation changed roll/pitch direction");
    require(max_rp_optimality_error <= 2e-5, "Roll/pitch authority is not LP-optimal");
    require(max_yaw_optimality_error <= torque_tolerance, "Yaw does not use the closest feasible residual torque");
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
    require(std::abs(frozen_initial.sum() - frozen_total) <= force_tolerance,
            "Frozen initial allocation raised collective thrust");
    const Vector<4> frozen_final = quad.runFlightCtl(0.0025, measured_final_omega, frozen);
    const Vector<3> final_torque = dynamics.getJ() * gains.asDiagonal() * (frozen.omega - measured_final_omega);
    const Vector<4> final_request(frozen_total, final_torque.x(), final_torque.y(), final_torque.z());
    require((frozen_final - inverse * final_request).cwiseAbs().maxCoeff() <= force_tolerance,
            "Frozen feasible final allocation was changed");

    // Dense V19: direct endpoint omega, with the preceding held CTBR command.
    // Static targets, not a replay of the unrecorded rotor-state history.
    const Command dense(0, Scalar(5.861665964126587 / 0.75), Vector<3>(3, 3, 1.4688522815704346));
    const Vector<3> dense_omega(-0.1414310336112976, 0.07296504825353622, 0.22537021338939667);
    const Vector<4> dense_target = quad.runFlightCtl(0.0025, dense_omega, dense);
    const Vector<4> dense_expected(0.2692468774237464, 0, 0.07429408670284054, 5.518125);
    require((dense_target - dense_expected).cwiseAbs().maxCoeff() <= force_tolerance,
            "Dense V19 allocation differs from independent double LP result");

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
              << ",\"max_rp_projection_error_Nm\":" << max_rp_projection_error
              << ",\"max_rp_optimality_error\":" << max_rp_optimality_error
              << ",\"max_yaw_optimality_error_Nm\":" << max_yaw_optimality_error
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
