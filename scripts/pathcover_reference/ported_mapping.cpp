#include <cstdint>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "nanovoxmap/nanovoxmap.hpp"
int main() {
  double resolution; int count; Eigen::Vector3d origin;
  std::cin >> resolution >> origin.x() >> origin.y() >> origin.z() >> count;
  std::vector<Eigen::Vector3d> cloud(count);
  for (auto& p:cloud) std::cin >> p.x() >> p.y() >> p.z();
  NanoVoxMap::OccupancyMapd map(resolution,.7,.4,.1192,.971,NanoVoxMap::Backend::kCpu,false);
  map.insertPointCloud(origin,cloud);
  std::cout << std::setprecision(17);
  for (const auto& p:map.getOccupiedVoxels()) std::cout << p.transpose() << '\n';
}
