// Copyright (c) 2026, Isaac Lab IPC Contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Reduced-attitude construction adapted from PX4 AttitudeControl.cpp:
// Copyright (c) 2019 PX4 Development Team. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name PX4 nor the names of its contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <algorithm>
#include <cmath>

#include "flightlib/common/command.hpp"
#include "flightlib/common/quad_state.hpp"
#include "utils/geometry/quadrotor_flatness.hpp"

namespace isaaclab_marsim {

inline flightlib::Command track_thrust_direction(
    const flightlib::QuadState& state, flightlib::Vector<3> force,
    flightlib::Vector<3> force_dot, flightlib::Scalar yaw, flightlib::Scalar yaw_rate,
    flightlib::Scalar attitude_gain, flightlib::Scalar rate_limit,
    flightlib::Scalar collective_limit, flatness::FlatnessMap& flatness) {
  using Scalar = flightlib::Scalar;
  using Vec3 = flightlib::Vector<3>;
  using Quat = flightlib::Quaternion;
  const auto canonical = [](Quat q) {
    q.normalize();
    if (q.w() < 0) q.coeffs() *= -1;
    return q;
  };

  // Position tracking uses upright thrust. Project its derivative consistently.
  // This controller constraint does not change Flightmare's physical limits.
  if (force.z() < 0) {
    force.z() = 0;
    force_dot.z() = 0;
  }
  if (force.norm() < 1e-5) return flightlib::Command(state.t, 0, Vec3::Zero());

  double unused_thrust;
  Eigen::Vector4d quaternion;
  Eigen::Vector3d angular_velocity;
  flatness.forward(state.v.cast<double>(), (force - Vec3(0, 0, 9.81)).cast<double>(),
    force_dot.cast<double>(), yaw, yaw_rate, unused_thrust, quaternion, angular_velocity);
  const Quat full = canonical(Quat(quaternion[0], quaternion[1], quaternion[2], quaternion[3]));
  const auto actual = state.R();
  const auto desired = full.toRotationMatrix();
  const Quat tilt = canonical(Quat::FromTwoVectors(Vec3(actual.col(2)), Vec3(desired.col(2))));
  Quat weighted = full;
  if (std::abs(tilt.w()) > 1e-5) {
    const Quat reduced = canonical(tilt * state.q());
    const Quat remaining = canonical(reduced.conjugate() * full);
    const Scalar angle = 2 * std::atan2(remaining.z(), remaining.w());
    // Deliberately reduce yaw bandwidth as well as its attitude-error weight.
    weighted = canonical(reduced * Quat(Eigen::AngleAxis<Scalar>(0.4 * angle, Vec3::UnitZ())));
  }
  const Quat error = canonical(state.q().conjugate() * weighted);
  Vec3 feedback = Scalar(2) * attitude_gain * error.vec();
  const Scalar feedback_max = feedback.cwiseAbs().maxCoeff();
  if (feedback_max > rate_limit) feedback *= rate_limit / feedback_max;

  // Keep the planner's complete flatness derivative in actual body axes.
  // Reserve half the native rate range for correcting attitude errors.
  Vec3 feedforward = actual.transpose() * desired * angular_velocity.cast<Scalar>();
  const Scalar feedforward_limit = Scalar(0.5) * rate_limit;
  if (feedforward.norm() > feedforward_limit) feedforward *= feedforward_limit / feedforward.norm();
  Scalar scale = 1;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(feedforward[i]) > 1e-6) {
      scale = std::min(scale, (rate_limit - std::abs(feedback[i])) / std::abs(feedforward[i]));
    }
  }
  const Vec3 rates = (feedback + std::clamp(scale, Scalar(0), Scalar(1)) * feedforward)
    .cwiseMax(Vec3::Constant(-rate_limit)).cwiseMin(Vec3::Constant(rate_limit));
  const Scalar collective = std::clamp(force.dot(actual.col(2)), Scalar(0), collective_limit);
  return flightlib::Command(state.t, collective, rates);
}

}  // namespace isaaclab_marsim
