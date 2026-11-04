// Copyright (c) 2026, Isaac Lab IPC Contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <yaml-cpp/yaml.h>

#include "marsim_render/marsim_render.hpp"

namespace {
class MarsimLidarNode : public rclcpp::Node {
 public:
  MarsimLidarNode() : Node("marsim_lidar_node") {
    const auto path = declare_parameter<std::string>("config_path", "");
    const auto config = YAML::LoadFile(path);
    const double sensing_rate = config["sensing_rate"] ? config["sensing_rate"].as<double>() : 10.0;
    if (!std::isfinite(sensing_rate) || sensing_rate <= 0.0) {
      throw std::invalid_argument("sensing_rate must be positive");
    }
    // The original MARSIM renderer owns the PCD, scan pattern and OpenGL context.
    renderer_ = std::make_unique<marsim::MarsimRender>(path);
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/cloud_registered", rclcpp::QoS(100).best_effort().durability_volatile());
    global_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/global_pc", rclcpp::QoS(1).transient_local());
    render_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/marsim/lidar_pose", 10);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      declare_parameter<std::string>("odom_topic", "/marsim/odometry"),
      rclcpp::SensorDataQoS(), [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
        const auto& q = message->pose.pose.orientation;
        const auto& p = message->pose.pose.position;
        Eigen::Quaternionf orientation(q.w, q.x, q.y, q.z);
        Eigen::Vector3f position(p.x, p.y, p.z);
        if (position.allFinite() && orientation.coeffs().allFinite() && orientation.norm() > 1e-6) {
          position_ = position;
          orientation_ = orientation.normalized();
          pose_stamp_ = rclcpp::Time(message->header.stamp);
          have_pose_ = true;
        }
      });
    pcl::PointCloud<marsim::PointType>::Ptr global;
    renderer_->getGlobalMap(global);
    if (!global || global->empty()) {
      throw std::runtime_error("MARSIM loaded an empty PCD map from config: " + path);
    }
    pcl::toROSMsg(*global, global_map_);
    global_map_.header.frame_id = "world";
    global_map_.header.stamp = now();
    global_pub_->publish(global_map_);
    global_timer_ = create_wall_timer(std::chrono::seconds(5), [this]() {
      global_map_.header.stamp = now();
      global_pub_->publish(global_map_);
    });
    scan_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / sensing_rate)),
      [this]() { scan(); });
    RCLCPP_INFO(get_logger(), "Original MARSIM LiDAR at %.1f Hz, driven by actual odometry", sensing_rate);
  }

 private:
  void scan() {
    if (!have_pose_) return;
    const auto stamp = pose_stamp_;
    pcl::PointCloud<marsim::PointType>::Ptr cloud(new pcl::PointCloud<marsim::PointType>);
    renderer_->renderOnceInWorld(position_, orientation_, stamp.seconds(), cloud);
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(*cloud, message);
    message.header.frame_id = "world";
    message.header.stamp = stamp;
    cloud_pub_->publish(message);
    geometry_msgs::msg::PoseStamped render_pose;
    render_pose.header = message.header;
    render_pose.pose.position.x = position_.x();
    render_pose.pose.position.y = position_.y();
    render_pose.pose.position.z = position_.z();
    render_pose.pose.orientation.w = orientation_.w();
    render_pose.pose.orientation.x = orientation_.x();
    render_pose.pose.orientation.y = orientation_.y();
    render_pose.pose.orientation.z = orientation_.z();
    render_pose_pub_->publish(render_pose);
  }

  std::unique_ptr<marsim::MarsimRender> renderer_;
  Eigen::Vector3f position_{Eigen::Vector3f::Zero()};
  Eigen::Quaternionf orientation_{Eigen::Quaternionf::Identity()};
  bool have_pose_{false};
  rclcpp::Time pose_stamp_{0, 0, RCL_ROS_TIME};
  sensor_msgs::msg::PointCloud2 global_map_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_, global_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr render_pose_pub_;
  rclcpp::TimerBase::SharedPtr scan_timer_, global_timer_;
};
}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    // MARSIM creates and uses its OpenGL context on this same thread.
    rclcpp::spin(std::make_shared<MarsimLidarNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("marsim_lidar_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
