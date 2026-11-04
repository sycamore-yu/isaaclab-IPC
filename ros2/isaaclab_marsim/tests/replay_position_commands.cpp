// Copyright (c) 2026, Isaac Lab IPC Contributors.
// SPDX-License-Identifier: BSD-3-Clause
// Replay captured PositionCommand CSV through the production tracker and native Flightmare.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "flightlib/objects/quadrotor.hpp"
#include "../src/position_command_tracking.hpp"

using namespace flightlib;
using Vec3 = Vector<3>;
struct Reference {
  double t;
  Vec3 p, v, a, j;
  Scalar yaw, yaw_rate;
};

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const Scalar kp = argc > 3 ? std::stof(argv[3]) : 4;
  const Scalar kd = argc > 4 ? std::stof(argv[4]) : 3;
  const Scalar attitude_gain = argc > 5 ? std::stof(argv[5]) : 6;
  std::ifstream input(argv[1]);
  std::vector<Reference> references;
  std::string line;
  while (std::getline(input, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::stringstream values(line);
    Reference r;
    values >> r.t;
    for (Vec3* vector : {&r.p, &r.v, &r.a, &r.j})
      for (int axis = 0; axis < 3; ++axis) values >> (*vector)[axis];
    values >> r.yaw >> r.yaw_rate;
    if (!values) return 2;
    references.push_back(r);
  }
  if (references.empty()) return 2;

  constexpr Scalar mass = 0.75, max_thrust = 29.43, max_rotor = 3 * mass * 9.81 / 4;
  const Scalar coefficient = 8.175e-5 / (4 * std::acos(-1.) * std::acos(-1.));
  YAML::Node config;
  auto cfg = config["quadrotor_dynamics"];
  cfg["mass"] = mass;
  cfg["arm_l"] = 0.2;
  cfg["motor_omega_min"] = 0.;
  cfg["motor_omega_max"] = std::sqrt(max_rotor / coefficient);
  cfg["motor_tau"] = 0.01;
  cfg["thrust_map"] = std::vector<double>{coefficient, 0, 0};
  cfg["kappa"] = 0.01;
  cfg["omega_max"] = std::vector<double>{10, 10, 10};
  QuadrotorDynamics dynamics;
  if (!dynamics.updateParams(config) || !dynamics.setInertiaDiagonal(Vec3::Constant(0.012))) return 3;
  Quadrotor quad(dynamics);
  quad.setBodyrateGains(Vec3::Constant(45));
  Matrix<3, 2> box;
  box.col(0).setConstant(-1e6);
  box.col(1).setConstant(1e6);
  quad.setWorldBox(box);
  QuadState state;
  state.setZero();
  state.p = Vec3(0, -50, 1.5);
  state.q(Quaternion(Eigen::AngleAxis<Scalar>(1.5741, Vec3::UnitZ())));
  quad.reset(state);
  quad.resetMotorThrusts(Vector<4>::Constant(mass * 9.81 / 4));
  flatness::FlatnessMap flatness;
  flatness.reset(mass, 9.81, 0, 0, 0, 1e-4);
  Reference reference{0, state.p, Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), 1.5741, 0};
  std::ofstream output(argv[2]);
  output << std::setprecision(12);
  output << "t,x,y,z,vx,vy,vz,T,wx,wy,wz,target_x,target_y,target_z,tilt_rad,error_m,m0,m1,m2,m3\n";
  size_t index = 0;
  for (int tick = 0; tick <= std::ceil(references.back().t / 0.01); ++tick) {
    const double t = tick * 0.01;
    while (index < references.size() && references[index].t <= t) reference = references[index++];
    const Vec3 force = reference.a + kp * (reference.p - state.p) +
      kd * (reference.v - state.v) + Vec3(0, 0, 9.81);
    const Vec3 derivative = reference.j + kp * (reference.v - state.v) + kd * (reference.a - state.a);
    const Command command = isaaclab_marsim::track_thrust_direction(
      state, force, derivative, reference.yaw, reference.yaw_rate, attitude_gain, 10, max_thrust, flatness);
    if (!command.valid() || command.omega.cwiseAbs().maxCoeff() > 10.00001 ||
        command.collective_thrust < 0 || command.collective_thrust > max_thrust ||
        !quad.run(command, 0.01) || !quad.getState(&state) || !state.x.allFinite()) return 4;
    Vector<4> motors;
    quad.getMotorThrusts(motors);
    output << t;
    for (Scalar value : {state.p.x(), state.p.y(), state.p.z(), state.v.x(), state.v.y(), state.v.z(),
      command.collective_thrust, command.omega.x(), command.omega.y(), command.omega.z(),
      reference.p.x(), reference.p.y(), reference.p.z(),
      std::acos(std::clamp(state.R()(2, 2), Scalar(-1), Scalar(1))), (reference.p - state.p).norm(),
      motors[0], motors[1], motors[2], motors[3]}) output << ',' << value;
    output << '\n';
  }
}
