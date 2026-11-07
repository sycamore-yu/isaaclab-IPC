"""Integer occupancy snapshots retain complete-map replacement and geometry."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <Eigen/Core>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "nanovoxmap/nanovoxmap.hpp"
#include "occupancy_inflation/point_cloud_inflater.h"
#include "pathcover_grid.h"
#include "distance_map_planner/distance_map_planner.h"

using V=Eigen::Vector3i;
using P=Eigen::Vector3d;
using Map=NanoVoxMap::OccupancyMapd;
using Grid=occupancy_inflation::VoxelBitGrid;
using Inflater=occupancy_inflation::PointCloudInflater;
using Params=occupancy_inflation::InflationParams;
using Stats=occupancy_inflation::InflationStats;

void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
P centre(const V& p,double resolution){return (p.cast<double>().array()+.5).matrix()*resolution;}
struct Reader{
    const std::vector<P>& cloud;
    std::size_t size()const{return cloud.size();}
    bool point(std::size_t i,float&x,float&y,float&z)const{
        x=cloud[i].x();y=cloud[i].y();z=cloud[i].z();return true;
    }
};
void same(const Grid&a,const Grid&b){
    require(a.window().base==b.window().base&&a.window().dims==b.window().dims,"window");
    require(a.wordCount()==b.wordCount(),"word count");
    require(a.wordCount()==0||std::memcmp(a.data(),b.data(),a.wordCount()*8)==0,"grid bits");
}
Params params(double resolution=.1){
    Params p;p.resolution=resolution;p.radius_xy=.2;p.radius_z=.2;
    p.shape=occupancy_inflation::InflationShape::kEllipsoid;p.input_voxel_volumes=true;
    p.incremental=false;p.use_cuda=false;p.num_threads=1;return p;
}
struct Comparison{
    Inflater point,integer;Stats a,b;
    explicit Comparison(const Params& p){
        std::string error;
        require(point.configure(p,&error)&&integer.configure(p,&error),"configuration");
    }
    void step(const Map& map){
        const auto cloud=map.getOccupiedVoxels();point.inflate(Reader{cloud},&a);
        integer.inflateOccupancySnapshot(map,&b);
        require(std::tie(a.input_points,a.filtered_points,a.new_seed_voxels,a.seed_voxels,
                    a.inflated_voxels,a.output_voxels,a.changed,a.ok,a.error)==
                std::tie(b.input_points,b.filtered_points,b.new_seed_voxels,b.seed_voxels,
                    b.inflated_voxels,b.output_voxels,b.changed,b.ok,b.error),"stats");
        same(point.seed_,integer.seed_);same(point.outputGrid(),integer.outputGrid());
    }
};
void replacement(){
    auto p=params();Comparison c(p);Map map(.1,.7,.4,.1192,.971,NanoVoxMap::Backend::kCpu,false);
    c.step(map);require(!c.b.changed,"initial empty");
    const std::vector<V> indices{{-65,-9,-9},{-64,0,0},{-8,-8,-8},{-1,-1,-1},
                                 {0,0,0},{7,7,7},{8,8,8},{63,-1,1},{64,8,-8}};
    for(const auto&v:indices)map.setOccupied(centre(v,.1));
    c.step(map);require(c.b.changed&&c.b.seed_voxels==indices.size(),"initial indices");
    for(const auto&v:indices)require(c.integer.seed_.testGlobal(v),"missing independent seed");
    c.step(map);require(!c.b.changed,"unchanged snapshot");
    for(const auto&v:indices)if(v!=V::Zero())map.setFree(centre(v,.1));
    c.step(map);require(c.b.changed&&c.b.seed_voxels==1,"removals");
    require(c.integer.seed_.window().dims==V(64,8,8),"window failed to shrink");
    map.setFree(centre(V::Zero(),.1));map.setOccupied(centre(V(1,1,1),.1));
    c.step(map);require(c.b.changed&&!c.integer.seed_.testGlobal(V::Zero()),"same-count moved bit");
    map.clear();c.step(map);require(c.b.changed&&c.b.output_voxels==0,"clear retained old obstacle");
    c.step(map);require(!c.b.changed,"repeated empty");
    const P origin=centre(V(-5,1,2),.1),end=centre(V(5,1,2),.1);
    for(int i=0;i<5;++i)map.insertRay(origin,end);
    c.step(map);require(c.b.seed_voxels==1,"hit");
    for(int i=0;i<12;++i){map.insertPointCloud(origin,{end},false);c.step(map);}
    require(c.b.seed_voxels==0&&c.b.output_voxels==0,"miss did not remove seed");
    // Snapshot replacement does not inherit the point API's incremental union.
    p.incremental=true;Comparison full(p);
    map.setOccupied(centre(V::Zero(),.1));full.integer.inflateOccupancySnapshot(map,&full.b);
    map.clear();full.integer.inflateOccupancySnapshot(map,&full.b);
    require(full.b.changed&&full.b.output_voxels==0,"snapshot used incremental union");
}
void bounds(){
    for(double resolution:{.1,.25,.3})for(const auto&r:std::vector<P>{{0,0,0},{.2,0,0},{0,0,.2},{.2,0,.2}}){
        auto p=params(resolution);p.radius_xy=r.x();p.radius_z=r.z();Comparison c(p);
        Map map(resolution,.7,.4,.1192,.971,NanoVoxMap::Backend::kCpu,false);
        for(const V&v:std::vector<V>{{-65,-9,-1},{-1,1,1},{7,7,7},{8,8,8},{64,1,-8}})
            map.setOccupied(centre(v,resolution));
        c.step(map);
        if(r.x()==0&&r.z()==0)same(c.integer.seed_,c.integer.outputGrid());
    }
    auto p=params();p.output_bounds_enable=true;p.output_min=P::Zero();p.output_max=P::Constant(.3);
    Comparison c(p);Map map(.1,.7,.4,.1192,.971,NanoVoxMap::Backend::kCpu,false);
    map.setOccupied(centre(V(-1,1,1),.1));c.step(map);
    require(c.integer.outputGrid().testGlobal(V(0,1,1)),"outside seed clipped before inflation");
    require(!c.integer.outputGrid().testGlobal(V(-1,1,1)),"output clipping changed");
    map.clear();c.step(map);require(c.b.output_voxels==0,"bounded clear");
}
void errors(){
    auto p=params();Comparison c(p);Map map(.1,.7,.4,.1192,.971,NanoVoxMap::Backend::kCpu,false);
    for(int x:{-1048575,1048574,-1048576,1048575}){
        map.clear();map.setOccupied(centre(V(x,0,0),.1));c.step(map);
        require(c.b.ok==(x>=-1048575&&x<=1048574),"supported lattice range");
    }
    p.max_grid_voxels=64;Comparison small(p);map.clear();
    map.setOccupied(centre(V::Zero(),.1));map.setOccupied(centre(V(64,1,1),.1));small.step(map);
    require(!small.b.ok&&small.b.error==occupancy_inflation::InflationError::kGridTooLarge,"grid budget");
    Inflater empty;Stats stats;empty.inflateOccupancySnapshot(map,&stats);
    require(!stats.ok&&stats.error==occupancy_inflation::InflationError::kNotConfigured,"unconfigured");
    p=params();p.input_filter_enable=true;std::string error;require(empty.configure(p,&error),"filter config");
    bool rejected=false;try{empty.inflateOccupancySnapshot(map,&stats);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"filtered lattice snapshot silently accepted");
}
void downstream(){
    auto p=params();Comparison c(p);Map map(.1,.7,.4,.1192,.971,NanoVoxMap::Backend::kCpu,false);
    for(int z=0;z<20;++z)for(int y=-20;y<=20;++y)map.setOccupied(centre(V(0,y,z),.1));
    c.step(map);
    const auto a=pathcover_boxes::compress(c.point.outputGrid()),b=pathcover_boxes::compress(c.integer.outputGrid());
    require(a.size()==b.size(),"box count");
    for(std::size_t i=0;i<a.size();++i)require(a[i].lower==b[i].lower&&a[i].upper==b[i].upper,"box order");
    const V dims(32,32,16),origin_index(-16,-16,-4);
    std::vector<int8_t> ca(dims.prod(),0),cb=ca;std::vector<int> ia,ib;
    for(int z=0;z<dims.z();++z)for(int y=0;y<dims.y();++y)for(int x=0;x<dims.x();++x)
        if(x==0||y==0||z==0||x+1==dims.x()||y+1==dims.y()||z+1==dims.z()){
            const int i=x+dims.x()*(y+dims.y()*z);ca[i]=cb[i]=100;
        }
    const auto ratio=pathcover_grid::resolutionRatio(.1,.25);
    pathcover_grid::coarsenOccupiedBoxes(a,c.point.outputGrid().window(),ratio,origin_index,dims,ca,ia);
    pathcover_grid::coarsenOccupiedBoxes(b,c.integer.outputGrid().window(),ratio,origin_index,dims,cb,ib);
    require(ca==cb&&ia==ib,"coarse occupancy");
    auto ma=std::make_shared<JPS::VoxelMapUtil>(),mb=std::make_shared<JPS::VoxelMapUtil>();
    const P start(-2,0,1),goal(2,0,1),origin=origin_index.cast<double>()*.25;
    ma->setMap(origin,dims,ca,.25);mb->setMap(origin,dims,cb,.25);
    DMPlanner3D da,db;for(auto*d:{&da,&db}){d->setEps(1.5);d->setXYHeuristic(true);d->setPotentialRadius(P::Constant(1.0));}
    da.setMap(ma,start);db.setMap(mb,start);
    require(da.computeGlobalRawPath(start,goal)&&db.computeGlobalRawPath(start,goal),"DMP solve");
    require(da.getPathCost()==db.getPathCost(),"DMP cost");
    const auto pa=da.getRawPath(),pb=db.getRawPath();require(pa.size()==pb.size(),"DMP path size");
    for(std::size_t i=0;i<pa.size();++i)require(pa[i]==pb[i],"DMP waypoint");
}
int main(int argc,char**argv){
    try{
        require(argc==2,"case required");const std::string name=argv[1];
        if(name=="replacement")replacement();else if(name=="bounds")bounds();
        else if(name=="errors")errors();else if(name=="downstream")downstream();else return 2;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def snapshot_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native"
    directory = tmp_path_factory.mktemp("occupancy_snapshot")
    source = directory / "snapshot.cpp"
    executable = directory / "snapshot"
    source.write_text(SOURCE)
    result = subprocess.run([
        compiler, "-O2", "-std=c++17", "-fno-access-control", "-pthread",
        "-I/usr/include/eigen3", f"-I{root / 'native'}",
        f"-I{vendor / 'nanovoxmap/include'}",
        f"-I{vendor / 'occupancy_inflation/include'}",
        f"-I{vendor / 'jps/include'}", str(source),
        str(vendor / "occupancy_inflation/src/voxel_bit_grid.cpp"),
        str(vendor / "occupancy_inflation/src/point_cloud_inflater.cpp"),
        str(vendor / "jps/src/distance_map_planner/graph_search.cpp"),
        str(vendor / "jps/src/distance_map_planner/distance_map_planner.cpp"),
        "-o", str(executable),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", ["replacement", "bounds", "errors", "downstream"])
def test_complete_integer_occupancy_snapshot(snapshot_binary, case):
    result = subprocess.run([str(snapshot_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

