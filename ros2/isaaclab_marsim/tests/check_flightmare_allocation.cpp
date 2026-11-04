// Copyright (c) 2026, Isaac Lab IPC Contributors.
// SPDX-License-Identifier: BSD-3-Clause

// Link against the same prepared flightlib_dynamics archive as flightmare_node.
// This numerical check does not replace the ROS mission and LiDAR flight tests.
#include <algorithm>
#include <cmath>
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
    const Scalar high = dynamics.collective_thrust_max() / 4;
    Scalar max_collective_error = 0, max_feasible_error = 0;
    Scalar max_rp_error = 0, max_collective_projection_error = 0;
    int cases = 0, feasible_cases = 0, collective_adjusted_cases = 0;
    const std::vector<Scalar> rates{-10, -3, -0.1, 0, 0.1, 3, 10};
    for (int level = 0; level <= 10; ++level) {
      const Scalar collective = dynamics.collective_thrust_max() / mass * level / 10;
      for (Scalar wx : rates) for (Scalar wy : rates) for (Scalar wz : rates) {
        const Vector<3> omega(wx, wy, wz);
        const Vector<3> torque = dynamics.getJ() * gains.asDiagonal() * omega;
        const Vector<4> requested(mass * collective, torque.x(), torque.y(), torque.z());
        const Vector<4> original = inverse * requested;
        const Vector<4> actual = quad.runFlightCtl(0.0025, Vector<3>::Zero(),
                                                 Command(0, collective, omega));
        require(actual.allFinite(), "Nonfinite allocated rotor force");
        require(actual.minCoeff() >= 0 && actual.maxCoeff() <= high,
                "Allocated rotor force outside native limits");
        const Vector<4> achieved = allocation * actual;
        max_collective_error = std::max(max_collective_error, std::abs(achieved[0] - requested[0]));
        collective_adjusted_cases += std::abs(achieved[0] - requested[0]) > 1e-4;
        if ((original.array() >= 0).all() && (original.array() <= high).all()) {
          max_feasible_error = std::max(max_feasible_error, (actual - original).cwiseAbs().maxCoeff());
          ++feasible_cases;
        } else {
          Vector<4> rp = inverse * Vector<4>(0, torque.x(), torque.y(), 0);
          const Scalar spread = rp.maxCoeff() - rp.minCoeff();
          const Scalar scale = spread > high ? high / spread : 1;
          rp *= scale;
          max_rp_error = std::max(max_rp_error,
            (achieved.segment<2>(1) - scale * torque.head<2>()).norm());
          const Scalar closest = 4 * std::max(-rp.minCoeff(),
            std::min(requested[0] / 4, high - rp.maxCoeff()));
          max_collective_projection_error = std::max(max_collective_projection_error,
            std::abs(achieved[0] - closest));
          if ((level == 0 || level == 10) && torque.head<2>().norm() > 1e-4)
            require(achieved.segment<2>(1).norm() > 1e-4,
                    "Zero/max collective removed roll/pitch authority");
        }
        // Roll and pitch retain their requested ratio and sign under saturation.
        require(std::abs(achieved[1] * requested[2] - achieved[2] * requested[1]) < 1e-4,
                "Saturation changed roll/pitch torque direction");
        require(achieved.segment<2>(1).dot(requested.segment<2>(1)) >= -1e-6,
                "Saturation reversed roll/pitch torque");
        ++cases;
      }
    }
    require(feasible_cases > 0 && max_feasible_error < 1e-6,
            "A feasible native allocation was changed");
    require(max_rp_error < 1e-4, "Saturation failed to preserve feasible roll/pitch torque");
    require(max_collective_projection_error < 1e-4,
            "Saturation did not choose the closest feasible collective thrust");
    require(collective_adjusted_cases > 0, "Collective was never adjusted to preserve attitude authority");

    // Reproduce the measured high-speed failure's rate/braking demand.
    QuadState state;
    state.setZero();
    state.p.z() = 10;
    state.w = Vector<3>(-1.65, 2.61, -1.02);
    require(quad.reset(state), "Cannot set measured rate state");
    const Command brake(0, 19.773, Vector<3>(-0.37, 1.58, -0.65));
    const Vector<3> torque = dynamics.getJ() * gains.asDiagonal() * (brake.omega - state.w);
    const Vector<4> request(mass * brake.collective_thrust, torque.x(), torque.y(), torque.z());
    const Vector<4> clipped = allocation * dynamics.clampThrust(inverse * request);
    const Vector<4> restored = allocation * quad.runFlightCtl(0.0025, state.w, brake);
    require(clipped.segment<2>(1).norm() < 1e-6, "Measured baseline no longer saturates diagonally");
    require(restored.segment<2>(1).dot(state.w.head<2>()) < 0,
            "Roll/pitch braking authority was not restored");
    const Scalar initial_rate_error = (state.w - brake.omega).head<2>().norm();
    quad.resetMotorThrusts(Vector<4>::Constant(request[0] / 4));
    for (int step = 0; step < 50; ++step) {
      require(quad.run(brake, 0.01), "Native motor/RK4 integration failed");
    }
    require(quad.getState(&state), "Native state became invalid");
    const Scalar final_rate_error = (state.w - brake.omega).head<2>().norm();
    require(final_rate_error < initial_rate_error, "Native loop did not reduce roll/pitch rate error");
    std::cout << "{\"cases\":" << cases << ",\"feasible_cases\":" << feasible_cases
              << ",\"max_feasible_force_error_N\":" << max_feasible_error
              << ",\"max_collective_force_error_N\":" << max_collective_error
              << ",\"collective_adjusted_cases\":" << collective_adjusted_cases
              << ",\"max_roll_pitch_torque_error_Nm\":" << max_rp_error
              << ",\"max_closest_collective_error_N\":" << max_collective_projection_error
              << ",\"baseline_rp_torque_Nm\":[" << clipped[1] << ',' << clipped[2]
              << "],\"restored_rp_torque_Nm\":[" << restored[1] << ',' << restored[2]
              << "],\"initial_rp_rate_error_radps\":" << initial_rate_error
              << ",\"final_rp_rate_error_radps\":" << final_rate_error << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
