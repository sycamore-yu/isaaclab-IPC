"""Contracts for the opt-in 3D XY lower bound and its per-planner cache."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <distance_map_planner/distance_map_planner.h>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
bool near(double a, double b) { return a==b || std::abs(a-b)<1e-9; }
using Cache=DMP::XYLowerBound3D;

void fees_and_weights() {
    Cache cache;
    std::vector<int8_t> map{90,0};
    check(!cache.update(map.data(),2,1,1,.1,1,{{1,0,0}}).cache_hit,"first table must miss");
    check(cache.data()[0]==1 && cache.data()[1]==0,"reverse fee or goal fee wrong");
    DMP::GraphSearch ordinary(map.data(),2,1,1,1.5,.1,false);
    ordinary.setXYLowerBound(cache.data());
    check(ordinary.plan(0,0,0,1,0,0)==91,"original start fee changed");
    for(auto node:ordinary.getAllSet())check(node->h==(node->x==0?1.5:0),"heuristic scaled twice");
    map={0,90};cache.mapChanged();cache.update(map.data(),2,1,1,.1,1,{{1,0,0}});
    check(cache.data()[0]==10,"reverse edge charged predecessor");
    map={0,100,0};cache.mapChanged();cache.update(map.data(),3,1,1,.1,0,{{2,0,0}});
    DMP::GraphSearch zero(map.data(),3,1,1,0,.1,false);zero.setXYLowerBound(cache.data());
    check(!std::isfinite(zero.plan(0,0,0,2,0,0)),"blocked graph connected");
    for(auto node:zero.getAllSet())check(node->h==0,"eps zero produced NaN or nonzero");
    map.assign(7,0);cache.mapChanged();cache.update(map.data(),7,1,1,.1,1,{{1,0,100},{6,0,0}});
    check(cache.data()[0]==6,"virtual goal connector ignored");
}

void cache_keys() {
    Cache cache;
    std::vector<int8_t> map(24,0);for(int i=12;i<24;++i)map[i]=20;
    std::vector<Cache::Goal> goals{{3,2,0}};
    auto update=[&](){return cache.update(map.data(),4,3,2,.1,1.5,goals).cache_hit;};
    check(!update() && update(),"unchanged input should hit");
    cache.mapChanged();check(update(),"identical installed map should hit");
    map[15]=30;cache.mapChanged();check(update(),"nonminimum height should not invalidate XY graph");
    map[3]=5;cache.mapChanged();check(!update(),"column minimum change not invalidated");
    map[2]=map[14]=100;cache.mapChanged();check(!update(),"column occupancy change not invalidated");
    goals={{2,2,0}};check(!update(),"goal XY change not invalidated");
    goals={{2,2,1}};check(!update(),"goal fee change not invalidated");
    goals={{2,2,9},{2,2,1},{2,2,4}};check(update(),"dominated same-XY fee changed relaxed goal");
    check(!cache.update(map.data(),4,3,2,.2,1.5,goals).cache_hit,"potential weight change not invalidated");
    check(!cache.update(map.data(),4,3,2,.2,1.0,goals).cache_hit,"heuristic weight change not invalidated");
    std::vector<int8_t> reshaped(24,0);
    check(!cache.update(reshaped.data(),4,2,3,.2,1.0,{{2,1,1}}).cache_hit,"dimensions change not invalidated");
    Cache independent;
    check(!independent.update(reshaped.data(),4,2,3,.2,1.0,{{2,1,1}}).cache_hit,"cache leaked across instances");
    Cache recovered;
    std::vector<int8_t> small{0,0};
    recovered.update(small.data(),2,1,1,.1,1,{{1,0,0}});
    small[1]=50;recovered.mapChanged();
    bool rejected=false;
    try{recovered.update(small.data(),2,1,1,.1,1,{{1,0,-1}});}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"invalid goal fee accepted");
    check(!recovered.update(small.data(),2,1,1,.1,1,{{1,0,0}}).cache_hit,"exception retained stale reverse distances");
    check(recovered.data()[0]==6,"recovered cache used old potential");
}

void public_lifecycle() {
    const Vec3i dimensions(6,4,3);std::vector<int8_t> values(dimensions.prod(),0);
    auto map=std::make_shared<JPS::VoxelMapUtil>();map->setMap(Vec3f::Zero(),dimensions,values,.5);
    DMPlanner3D planner;planner.setEps(1.5);planner.setPotentialMap(map,values);
    check(!planner.hasXYHeuristic(),"public default changed");
    check(planner.computeGlobalRawPath(Vec3f(.25,.25,.25),Vec3f(2.75,1.75,1.25)),"baseline failed");
    const auto expected=planner.getRawPath();const double expected_cost=planner.getPathCost();
    planner.setXYHeuristic(true);
    check(planner.computeGlobalRawPath(Vec3f(.25,.25,.25),Vec3f(2.75,1.75,1.25)),"XY solve failed");
    check(!planner.xyHeuristicCacheHit(),"first planner table should miss");
    check(planner.getPathCost()==expected_cost && planner.getRawPath()==expected,"open fixture path changed");
    check(planner.computeGlobalRawPath(Vec3f(.75,.25,.25),Vec3f(2.75,1.75,1.25)) && planner.xyHeuristicCacheHit(),"moving start did not reuse table");
    planner.setPotentialMap(map,values);
    check(planner.computeGlobalRawPath(Vec3f(.75,.25,.25),Vec3f(2.75,1.75,1.25)) && planner.xyHeuristicCacheHit(),"equal reinstalled map missed");
    check(planner.computeGlobalRawPath(Vec3f(.75,.25,.25),Vec3f(2.25,1.75,1.25)) && !planner.xyHeuristicCacheHit(),"changed goal reused stale distances");
    DMPlanner3D::EndpointAnchors starts{{Vec3i(0,0,0),.2},{Vec3i(1,0,0),.5}};
    DMPlanner3D::EndpointAnchors goals{{Vec3i(5,3,0),.7},{Vec3i(5,3,2),.4}};
    check(planner.computeGlobalRawPath(starts,goals) && !planner.xyHeuristicCacheHit(),"multi anchors failed");
    const double first_cost=planner.getPathCost();
    std::swap(goals[0],goals[1]);
    check(planner.computeGlobalRawPath(starts,goals) && planner.xyHeuristicCacheHit(),"equivalent goal-column costs missed");
    check(planner.getPathCost()==first_cost,"target order changed objective");
    goals[0].second=.3;
    check(planner.computeGlobalRawPath(starts,goals) && !planner.xyHeuristicCacheHit(),"connector metres/res change reused table");
    planner.setXYHeuristic(false);
    check(planner.computeGlobalRawPath(starts,goals) && !planner.xyHeuristicCacheHit(),"disabled mode retained hit metric");
    DMPlanner2D two;two.setXYHeuristic(true);check(!two.hasXYHeuristic(),"2D activated 3D feature");
    values[0]=-1;planner.setPotentialMap(map,values);
    planner.setXYHeuristic(true);
    bool rejected=false;try{planner.computeGlobalRawPath(Vec3f(.75,.25,.25),Vec3f(2.25,1.75,1.25));}
    catch(const std::invalid_argument&){rejected=true;}check(rejected,"negative unknown potential silently projected");
}

double metric(int x,int y,int z) {
    x=std::abs(x);y=std::abs(y);z=std::abs(z);
    int lo=std::min({x,y,z}),hi=std::max({x,y,z}),mid=x+y+z-lo-hi;
    return hi+(std::sqrt(2.)-1)*mid+(std::sqrt(3.)-std::sqrt(2.))*lo;
}
void lower_bound_property() {
    const int nx=5,ny=4,nz=3;std::mt19937 random(87341);
    for(int fixture=0;fixture<5;++fixture){
        std::vector<int8_t> map(nx*ny*nz);for(auto& v:map)v=random()%5==0?100:random()%70;map.back()=0;
        Cache cache;cache.update(map.data(),nx,ny,nz,.1,1,{{nx-1,ny-1,0}});
        auto id=[&](int x,int y,int z){return x+nx*(y+ny*z);};
        auto free=[&](int x,int y,int z){return x>=0&&x<nx&&y>=0&&y<ny&&z>=0&&z<nz&&map[id(x,y,z)]<100;};
        auto h=[&](int x,int y,int z){return std::max(metric(x-nx+1,y-ny+1,z-nz+1),cache.data()[x+nx*y]);};
        for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){
            if(!free(x,y,z))continue;
            DMP::GraphSearch baseline(map.data(),nx,ny,nz,0,.1,false), candidate(map.data(),nx,ny,nz,1,.1,false);
            candidate.setXYLowerBound(cache.data());
            const double exact=baseline.plan(x,y,z,nx-1,ny-1,nz-1),got=candidate.plan(x,y,z,nx-1,ny-1,nz-1);
            check(near(exact,got),"A* disagrees with original Dijkstra");
            check(h(x,y,z)<=exact-map[id(x,y,z)]+1e-9,"projected heuristic is not a lower bound");
            for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)for(int dz=-1;dz<=1;++dz){
                if(!dx&&!dy&&!dz)continue;bool valid=true;
                for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c)
                    valid=valid&&free(x+a*dx,y+b*dy,z+c*dz);
                if(!valid)continue;
                double edge=std::sqrt(double(dx*dx+dy*dy+dz*dz))+.1*map[id(x+dx,y+dy,z+dz)];
                check(h(x,y,z)<=edge+h(x+dx,y+dy,z+dz)+1e-9,"projected heuristic is inconsistent");
            }
        }
    }
}

int main(int argc,char**argv){
 try{
  const std::string name=argc>1?argv[1]:"";
  if(name=="fees")fees_and_weights();else if(name=="cache")cache_keys();
  else if(name=="lifecycle")public_lifecycle();else if(name=="bounds")lower_bound_property();
  else throw std::invalid_argument("unknown case");
  std::cout<<name<<" passed\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for native DMP contracts")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native/jps"
    directory = tmp_path_factory.mktemp("dmp_xy")
    source = directory / "test.cpp"
    source.write_text(SOURCE)
    target = directory / "test"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-DNDEBUG", "-I/usr/include/eigen3",
        f"-I{vendor / 'include'}", str(source),
        str(vendor / "src/distance_map_planner/graph_search.cpp"),
        str(vendor / "src/distance_map_planner/distance_map_planner.cpp"),
        "-o", str(target),
    ], check=True, capture_output=True, text=True)
    return target


@pytest.mark.parametrize("case", ["fees", "cache", "lifecycle", "bounds"])
def test_xy_heuristic_contract(executable, case):
    subprocess.run([str(executable), case], check=True, capture_output=True, text=True)
