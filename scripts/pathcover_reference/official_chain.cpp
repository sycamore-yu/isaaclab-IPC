// Standalone oracle: only unmodified pinned PathCover headers and sources.
// The calling script supplies scalar parameters and the same observed cloud.
#include <math.h>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <Eigen/Dense>
#include "nanovoxmap/nanovoxmap.hpp"
#include "occupancy_inflation/point_cloud_inflater.h"
#include "planner/VoxelGrid.hpp"
#include "planner/PathCover.hpp"
#include <jps_planner/jps_planner.h>
#include <distance_map_planner/distance_map_planner.h>
#include "gcopter_solver.hpp"

using V=Eigen::Vector3d;
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point t) {
 return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
struct Reader {
 const std::vector<V>& points;
 std::size_t size() const { return points.size(); }
 bool point(std::size_t i,float&x,float&y,float&z) const {
  x=points[i].x();y=points[i].y();z=points[i].z();return true;
 }
};

int main() {
 std::cout<<std::setprecision(17);std::cerr<<std::setprecision(17);
 V lower,upper,goal;Eigen::Matrix3d initial,terminal=Eigen::Matrix3d::Zero();
 for(int i=0;i<3;++i)std::cin>>lower(i);for(int i=0;i<3;++i)std::cin>>upper(i);
 double map_res,search_res,radius,rz,alpha,potential,search;
 int horizon,seed,count;
 std::cin>>map_res>>search_res>>radius>>rz>>horizon>>alpha>>potential>>search>>seed;
 for(int c=0;c<3;++c)for(int r=0;r<3;++r)std::cin>>initial(r,c);
 for(int i=0;i<3;++i)std::cin>>goal(i);
 std::cin>>count;std::vector<V> cloud(count);
 for(auto &p:cloud)for(int i=0;i<3;++i)std::cin>>p(i);
 double time_weight,margin,tolerance;int pieces,quadrature,iterations;
 std::cin>>time_weight>>margin>>pieces>>quadrature>>tolerance>>iterations;
 Eigen::VectorXd limits(5),weights(5),physics(6);
 for(int i=0;i<5;++i)std::cin>>limits(i);
 for(int i=0;i<5;++i)std::cin>>weights(i);
 for(int i=0;i<6;++i)std::cin>>physics(i);
 try {
  auto begin=Clock::now();
  NanoVoxMap::OccupancyMapd map(map_res,.7,.4,.1192,.971,NanoVoxMap::Backend::kCpu,false);
  map.insertPointCloud(initial.col(0),cloud);
  const auto occupied=map.getOccupiedVoxels();
  std::cerr<<"TIME mapping_ms "<<milliseconds(begin)<<"\n";
  begin=Clock::now();
  occupancy_inflation::InflationParams params;
  params.resolution=map_res;params.radius_xy=radius;params.radius_z=rz;
  params.shape=occupancy_inflation::InflationShape::kEllipsoid;
  params.use_cuda=false;params.num_threads=1;params.incremental=false;
  params.output_bounds_enable=true;params.output_min=lower;params.output_max=upper;
  occupancy_inflation::PointCloudInflater inflater;std::string error;
  if(!inflater.configure(params,&error))throw std::runtime_error(error);
  occupancy_inflation::InflationStats stats;inflater.inflate(Reader{occupied},&stats);
  if(!stats.ok)throw std::runtime_error("official inflation failed");
  Eigen::Matrix<float,Eigen::Dynamic,3,Eigen::RowMajor> packed(inflater.occupiedCount(),3);
  inflater.serialize(reinterpret_cast<uint8_t*>(packed.data()),3*sizeof(float),false);
  std::vector<V> inflated(packed.rows());
  for(int i=0;i<packed.rows();++i)inflated[i]=packed.row(i).cast<double>();
  std::cerr<<"TIME inflation_ms "<<milliseconds(begin)<<"\n";
  for(const auto&p:occupied)std::cout<<"OCCUPIED "<<p.transpose()<<"\n";
  for(const auto&p:inflated)std::cout<<"INFLATED "<<p.transpose()<<"\n";
  begin=Clock::now();
  std::vector<signed char> cells;auto dim=VoxelGrid::initializeGrid<double,3>(lower,upper,search_res,cells);
  std::vector<int> touched;VoxelGrid::setOccupancy<double,3>(inflated,lower,search_res,cells,dim,touched);
  auto grid=std::make_shared<JPS::VoxelMapUtil>();grid->setMap(lower,dim,cells,search_res);
  JPSPlanner<3> jps(false);jps.setMapUtil(grid);jps.updateMap();
  std::cerr<<"TIME grid_ms "<<milliseconds(begin)<<"\n";
  begin=Clock::now();if(!jps.plan(initial.col(0),goal,1.,true))throw std::runtime_error("official JPS failed");
  std::cerr<<"TIME jps_ms "<<milliseconds(begin)<<"\n";
  begin=Clock::now();
  DMPlanner<3> dmp(false);dmp.setPotentialRadius(V::Constant(potential));dmp.setSearchRadius(V::Constant(search));
  // Upstream node leaves DMP's heuristic at its original zero default.
  dmp.setMap(grid,initial.col(0));
  if(!dmp.computePath(initial.col(0),goal,jps.getRawPath()))throw std::runtime_error("official DMP failed");
  std::cerr<<"TIME dmp_ms "<<milliseconds(begin)<<"\n";
  std::vector<V> path{initial.col(0)};for(const auto&p:dmp.getRawPath())path.push_back(p);path.push_back(goal);
  for(const auto&p:path)std::cout<<"PATH "<<p.transpose()<<"\n";
  begin=Clock::now();
  Eigen::Matrix<double,Eigen::Dynamic,3> bounds(6,3);Eigen::VectorXd offsets(6);
  bounds.topRows(3)=-Eigen::Matrix3d::Identity();bounds.bottomRows(3)=Eigen::Matrix3d::Identity();
  offsets.head(3)=-lower;offsets.tail(3)=upper;
  std::vector<Eigen::Matrix<double,Eigen::Dynamic,3>> As;
  std::vector<Eigen::VectorXd> bs;std::vector<V> seeds;std::vector<std::vector<int>> decay;
  std::srand(seed);
  PathCover::pathCover<double,3>(inflated,path,bounds,offsets,As,bs,seeds,decay,horizon,alpha);
  std::cerr<<"TIME corridor_ms "<<milliseconds(begin)<<"\n";
  for(const auto&p:seeds)std::cout<<"SEED "<<p.transpose()<<"\n";
  std::vector<Eigen::MatrixX4d> corridors;
  for(int i=0;i<As.size();++i) {
   Eigen::MatrixX4d h(As[i].rows(),4);h.leftCols<3>()=As[i];h.col(3)=-bs[i];
   h.array().colwise()/=As[i].rowwise().norm().array();corridors.push_back(h);
   for(int r=0;r<h.rows();++r)std::cout<<"FACE "<<i<<" "<<h.row(r)<<"\n";
  }
  terminal.col(0)=seeds.back();trajectory_server::GcopterSolver<3> solver;
  begin=Clock::now();
  if(!solver.setup(time_weight,initial,terminal,corridors,margin,2.,pieces,.01,quadrature,limits,weights,physics))
   throw std::runtime_error("official GCOPTER setup failed");
  Trajectory<5> trajectory;double cost=solver.optimize(trajectory,tolerance,iterations);
  std::cerr<<"TIME trajectory_ms "<<milliseconds(begin)<<"\n";
  if(!std::isfinite(cost)||trajectory.getPieceNum()==0)throw std::runtime_error("official GCOPTER optimization failed");
  std::cout<<"OBJECTIVE "<<cost<<"\n";
  for(int i=0;i<trajectory.getPieceNum();++i) {
   std::cout<<"PIECE "<<i<<" "<<trajectory[i].getDuration()<<" ";
   const auto coefficients=trajectory[i].getCoeffMat();
   for(int r=0;r<3;++r)for(int c=0;c<6;++c)std::cout<<coefficients(r,c)<<" ";
   std::cout<<"\n";
  }
 }catch(const std::exception& error) {
  std::cerr<<"ERROR "<<error.what()<<"\n";return 4;
 }
}
