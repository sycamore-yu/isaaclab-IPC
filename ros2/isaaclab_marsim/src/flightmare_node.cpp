// Copyright (c) 2026, Isaac Lab IPC Contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <mars_quadrotor_msgs/msg/position_command.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

#include "flightlib/common/command.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "position_command_tracking.hpp"

namespace {
using Vec3 = flightlib::Vector<3>;
using Quat = flightlib::Quaternion;
using Scalar = flightlib::Scalar;
using PositionCommand = mars_quadrotor_msgs::msg::PositionCommand;

template <typename T>
Vec3 vector(const T& value) {
  return Vec3(value.x, value.y, value.z);
}

template <typename T>
void assign(T& target, const Vec3& value) {
  target.x = value.x();
  target.y = value.y();
  target.z = value.z();
}

class FlightmareNode : public rclcpp::Node {
 public:
  FlightmareNode() : Node("flightmare_node") {
    const auto config_path = declare_parameter<std::string>("config_path", "");
    const auto config = YAML::LoadFile(config_path);
    const auto source = declare_parameter<std::string>("command_source", "ctbr");
    if (source != "ctbr" && source != "position_command") {
      throw std::invalid_argument("command_source must be ctbr or position_command");
    }
    position_control_ = source == "position_command";
    const double frequency = declare_parameter<double>("control_frequency", 100.0);
    if (!std::isfinite(frequency) || frequency <= 0.0) {
      throw std::invalid_argument("control_frequency must be positive");
    }
    dt_ = 1.0 / frequency;
    kp_ = declare_parameter<double>("position_gain", 4.0);
    kd_ = declare_parameter<double>("velocity_gain", 3.0);
    attitude_gain_ = declare_parameter<double>("attitude_gain", 6.0);
    const auto mass = declare_parameter<double>("mass", 0.75);
    const auto inertia = declare_parameter<std::vector<double>>("inertia", {0.012, 0.012, 0.012});
    const auto motor_tau = declare_parameter<double>("motor_tau", 0.01);
    const auto rate_gain = declare_parameter<double>("rate_gain", 45.0);
    const auto max_thrust = declare_parameter<double>("max_rotor_thrust", 3.0 * 0.75 * 9.81 / 4.0);
    const auto rate_limit = declare_parameter<double>("body_rate_limit", 10.0);
    if (inertia.size() != 3 || mass <= 0.0 || motor_tau <= 0.0 ||
        rate_gain <= 0.0 || max_thrust <= 0.0 || rate_limit <= 0.0) {
      throw std::invalid_argument("Invalid physical parameters");
    }

    // Newton specifies N/(rev/s)^2; Flightmare uses N/(rad/s)^2.
    const double thrust_constant = 8.175e-5 / (4.0 * std::acos(-1.0) * std::acos(-1.0));
    YAML::Node parameters;
    auto dynamics_config = parameters["quadrotor_dynamics"];
    dynamics_config["mass"] = mass;
    dynamics_config["arm_l"] = 0.2;
    dynamics_config["motor_omega_min"] = 0.0;
    dynamics_config["motor_omega_max"] = std::sqrt(max_thrust / thrust_constant);
    dynamics_config["motor_tau"] = motor_tau;
    dynamics_config["thrust_map"] = std::vector<double>{thrust_constant, 0.0, 0.0};
    dynamics_config["kappa"] = 0.01;
    dynamics_config["omega_max"] = std::vector<double>{rate_limit, rate_limit, rate_limit};
    flightlib::QuadrotorDynamics dynamics;
    if (!dynamics.updateParams(parameters) ||
        !dynamics.setInertiaDiagonal(Vec3(inertia[0], inertia[1], inertia[2]))) {
      throw std::invalid_argument("Flightmare rejected physical parameters");
    }
    quad_ = std::make_unique<flightlib::Quadrotor>(dynamics);
    quad_->setBodyrateGains(Vec3::Constant(rate_gain));
    flatness_.reset(mass, 9.81, 0.0, 0.0, 0.0, 1e-4);
    rate_limit_ = rate_limit;
    collective_limit_ = 4.0 * max_thrust / mass;
    flightlib::Matrix<3, 2> world_box;
    world_box.col(0).setConstant(-1e6);
    world_box.col(1).setConstant(1e6);
    quad_->setWorldBox(world_box);

    state_.setZero();
    const auto initial = config["init_position"];
    state_.p = Vec3(initial && initial["x"] ? initial["x"].as<double>() : 0.0,
                    initial && initial["y"] ? initial["y"].as<double>() : 0.0,
                    initial && initial["z"] ? initial["z"].as<double>() : 1.5);
    const double yaw = config["init_yaw"] ? config["init_yaw"].as<double>() : 0.0;
    state_.q(Quat(Eigen::AngleAxis<Scalar>(yaw, Vec3::UnitZ())));
    if (!quad_->reset(state_)) throw std::runtime_error("Flightmare reset failed");
    quad_->resetMotorThrusts(flightlib::Vector<4>::Constant(mass * 9.81 / 4.0));
    assign(target_.position, state_.p);
    target_.yaw = yaw;
    collective_ = 9.81;
    rates_.setZero();

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/marsim/odometry", 10);
    legacy_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/lidar_slam/odom", 10);
    accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>("/marsim/acceleration", 10);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/lidar_slam/pose", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    if (position_control_) {
      ctbr_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/marsim/ctbr", 10);
      position_sub_ = create_subscription<PositionCommand>(
        "/planning/pos_cmd", rclcpp::QoS(10).best_effort(),
        [this](PositionCommand::ConstSharedPtr msg) {
          if (vector(msg->position).allFinite() && vector(msg->velocity).allFinite() &&
              vector(msg->acceleration).allFinite() && vector(msg->jerk).allFinite() &&
              std::isfinite(msg->yaw) && std::isfinite(msg->yaw_dot)) target_ = *msg;
        });
    } else {
      ctbr_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/marsim/ctbr", 1, [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {
          if (std::isfinite(msg->twist.linear.z) && vector(msg->twist.angular).allFinite()) {
            collective_ = msg->twist.linear.z;
            rates_ = vector(msg->twist.angular);
          }
        });
    }
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt_)),
      [this]() { step(); });
    RCLCPP_INFO(get_logger(),
      "Native Flightmare: %s, %.1f Hz, mass %.3f kg, motor tau %.3f s; RK4 substeps <= 0.0025 s",
      source.c_str(), frequency, mass, motor_tau);
  }

 private:
  void track_position() {
    const Vec3 force = vector(target_.acceleration) +
      Scalar(kp_) * (vector(target_.position) - state_.p) +
      Scalar(kd_) * (vector(target_.velocity) - state_.v) + Vec3(0.0, 0.0, 9.81);
    // Differentiate the complete feedback force; state_.a is net world acceleration.
    const Vec3 force_dot = vector(target_.jerk) +
      Scalar(kp_) * (vector(target_.velocity) - state_.v) +
      Scalar(kd_) * (vector(target_.acceleration) - state_.a);
    const auto command = isaaclab_marsim::track_thrust_direction(
      state_, force, force_dot, target_.yaw, target_.yaw_dot,
      attitude_gain_, rate_limit_, collective_limit_, flatness_);
    rates_ = command.omega;
    collective_ = command.collective_thrust;
  }

  void step() {
    if (position_control_) {
      track_position();
      geometry_msgs::msg::TwistStamped command;
      command.header.stamp = now();
      command.header.frame_id = "base_link";
      command.twist.linear.z = collective_;
      assign(command.twist.angular, rates_);
      ctbr_pub_->publish(command);
    }
    const Vec3 previous_rates = state_.w;
    if (!quad_->run(flightlib::Command(state_.t, collective_, rates_), dt_) ||
        !quad_->getState(&state_)) {
      RCLCPP_FATAL(get_logger(), "Flightmare integration failed");
      rclcpp::shutdown();
      return;
    }
    const auto stamp = now();
    const auto q = state_.q();
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "world";
    odom.child_frame_id = "base_link";
    assign(odom.pose.pose.position, state_.p);
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    assign(odom.twist.twist.linear, Vec3(state_.R().transpose() * state_.v));
    assign(odom.twist.twist.angular, state_.w);
    odom_pub_->publish(odom);

    // SUPER's established input uses world-frame velocity. Keep it explicit.
    auto legacy = odom;
    legacy.child_frame_id = "world";
    assign(legacy.twist.twist.linear, state_.v);
    assign(legacy.twist.twist.angular, Vec3(state_.R() * state_.w));
    legacy_odom_pub_->publish(legacy);
    geometry_msgs::msg::AccelStamped acceleration;
    acceleration.header = odom.header;
    assign(acceleration.accel.linear, state_.a);
    assign(acceleration.accel.angular, Vec3(state_.R() * ((state_.w - previous_rates) / Scalar(dt_))));
    accel_pub_->publish(acceleration);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom.header;
    pose.pose = odom.pose.pose;
    pose_pub_->publish(pose);
    geometry_msgs::msg::TransformStamped transform;
    transform.header = odom.header;
    transform.child_frame_id = "base_link";
    assign(transform.transform.translation, state_.p);
    transform.transform.rotation = odom.pose.pose.orientation;
    tf_->sendTransform(transform);
  }

  std::unique_ptr<flightlib::Quadrotor> quad_;
  flatness::FlatnessMap flatness_;
  flightlib::QuadState state_;
  PositionCommand target_;
  Vec3 rates_;
  Scalar collective_, rate_limit_, collective_limit_;
  double dt_, kp_, kd_, attitude_gain_;
  bool position_control_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr ctbr_sub_;
  rclcpp::Subscription<PositionCommand>::SharedPtr position_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_, legacy_odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr accel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr ctbr_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};
}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<FlightmareNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("flightmare_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
