"""CPU regressions for endpoint connectors on the unchanged legal DMP graph."""

import shutil
import subprocess
from pathlib import Path

import pytest


SOURCE = r"""
#include <distance_map_planner/distance_map_planner.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual,double expected) {
    if (std::abs(actual-expected)>1e-12) {
        std::cerr << "cost " << actual << ", expected " << expected << '\n';
        throw std::runtime_error("path cost differs");
    }
}

void disconnected_start() {
    const int width=8,height=7;
    std::vector<int8_t> cells(width*height,0);
    for(int y=0;y<height;++y) cells[3+width*y]=100;
    DMP::GraphSearch graph(cells.data(),width,height,1.5,0.1);
    check(!std::isfinite(graph.plan(1,3,6,3)),"nearest start must be disconnected");
    const double cost=graph.plan({{1,3,0,0.1},{4,3,0,5.0}},{{6,3,0,0.2}});
    near(cost,7.2);
    check(graph.getStartAnchorIndex()==1 && graph.getGoalAnchorIndex()==0,"wrong start component");
    check(graph.getPath().back()->x==4 && graph.getPath().front()->x==6,"wrong anchor path ends");
    for(const auto& node:graph.getPath()) check(node->x>=4,"path crossed an occupied wall");
}

void disconnected_goal() {
    const int width=7,height=6;
    std::vector<int8_t> cells(width*height,0);
    for(int y=0;y<height;++y) cells[3+width*y]=100;
    DMP::GraphSearch graph(cells.data(),width,height,1.5,0.1);
    const double cost=graph.plan({{2,1,0,0.0}},{{4,1,0,0.0},{1,4,0,0.3}});
    check(std::isfinite(cost),"valid goal component was ignored");
    check(graph.getStartAnchorIndex()==0 && graph.getGoalAnchorIndex()==1,"wrong goal component");
    for(const auto& node:graph.getPath()) check(node->x<3,"path crossed an occupied wall");
}

void virtual_goal() {
    std::vector<int8_t> cells(7,0);
    DMP::GraphSearch graph(cells.data(),7,1,1.5,0.1);
    // x=1 is popped before x=6. Returning at the first anchor would cost 101.
    near(graph.plan({{0,0,0,0.0}},{{1,0,0,100.0},{6,0,0,0.0}}),6.0);
    check(graph.getGoalAnchorIndex()==1,"search stopped at first goal anchor");
    bool first_closed=false;
    for(const auto& node:graph.getCloseSet()) {
        check(node->id<7,"virtual goal leaked through grid-state getter");
        first_closed=first_closed || node->x==1;
    }
    check(first_closed,"test did not visit the expensive goal anchor first");
    for(const auto& node:graph.getAllSet()) check(node->id<7,"virtual goal is not a voxel");
}

void connector_costs() {
    std::vector<int8_t> cells(6,0);
    DMP::GraphSearch graph(cells.data(),6,1,1.5,0.1);
    near(graph.plan({{0,0,0,0.0},{4,0,0,10.0}},{{5,0,0,1.0}}),6.0);
    check(graph.getStartAnchorIndex()==0,"start connector cost was ignored");
    near(graph.plan({{1,0,0,5.0},{1,0,0,1.0}},{{3,0,0,2.0},{3,0,0,0.5}}),3.5);
    check(graph.getStartAnchorIndex()==1 && graph.getGoalAnchorIndex()==1,"duplicate anchor chose costly connector");
    near(graph.plan({{2,0,0,1.0}},{{2,0,0,0.5}}),1.5);
    check(graph.getPath().size()==1,"same-cell connectors need one coarse waypoint");
}

void no_corner_cut() {
    std::vector<int8_t> cells(27,0);
    auto id=[](int x,int y,int z){return x+3*y+9*z;};
    for(const auto& offset:std::vector<std::array<int,3>>{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}})
        cells[id(1+offset[0],1+offset[1],1+offset[2])]=100;
    DMP::GraphSearch graph(cells.data(),3,3,3,1.5,0.1);
    check(!std::isfinite(graph.plan({{1,1,1,0.0}},{{2,2,2,0.0}})),"multi-anchor search escaped through a corner");
    check(graph.getPath().empty() && graph.getStartAnchorIndex()==-1 && graph.getGoalAnchorIndex()==-1,"failed search retained a path");
}

void metric_units() {
    const Vec3i dim(4,3,2);
    std::vector<int8_t> cells(dim.prod(),0);
    cells[1+dim.x()]=7;
    auto map=std::make_shared<JPS::MapUtil<3>>();
    map->setMap(Vec3f(-0.5,1.0,0.0),dim,cells,0.25);
    DMPlanner3D planner; planner.setEps(1.5); planner.setCweight(0.1); planner.setPotentialMap(map,cells);
    DMPlanner3D::EndpointAnchors starts{{Vec3i(1,1,0),0.5}},goals{{Vec3i(3,1,0),0.25}};
    check(planner.computeGlobalPath(starts,goals),"metric connector wrapper failed");
    // 7 start potential + 0.5/.25 + 2 coarse steps + .25/.25.
    near(planner.getPathCost(),12.0);
    check(planner.getStartAnchorIndex()==0 && planner.getGoalAnchorIndex()==0,"wrapper index mismatch");
    const auto path=planner.getRawPath();
    check((path.front()-map->intToFloat(starts[0].first)).norm()==0.0,"wrapper inserted endpoint outside coarse path");
    check((path.back()-map->intToFloat(goals[0].first)).norm()==0.0,"wrong coarse goal centre");
    check(planner.getPriorPath().empty() && planner.getSearchRegion().size()==cells.size(),"global getter contract changed");
    check(!planner.computeGlobalPath(DMPlanner3D::EndpointAnchors{},goals),"empty start list succeeded");
    check(planner.getRawPath().empty() && planner.getStartAnchorIndex()==-1,"empty input retained selected anchor");
}

void legacy_regression() {
    const Vec3i dim(9,7,3);
    std::vector<int8_t> cells(dim.prod(),0);
    for(int z=0;z<dim.z();++z) for(int y=0;y<dim.y();++y) for(int x=0;x<dim.x();++x)
        cells[x+dim.x()*(y+dim.y()*z)] = x==4 && y<6 ? 100 : (x+2*y+3*z)%9;
    auto map=std::make_shared<JPS::MapUtil<3>>();
    map->setMap(Vec3f(-1.0,-0.75,0.0),dim,cells,0.25);
    DMPlanner3D planner;planner.setEps(1.5);planner.setCweight(0.1);planner.setPotentialMap(map,cells);
    // Frozen before this change from the old public single-endpoint API.
    const std::vector<Vec3i> expected{{1,1,1},{2,2,1},{2,3,1},{2,4,1},{2,5,1},
        {3,6,1},{4,6,1},{5,6,1},{6,5,1},{7,4,1},{7,3,1},{7,2,1},{7,1,1}};
    auto verify=[&]() {
        near(planner.getPathCost(),22.756854249492381);
        const auto path=planner.getRawPath();check(path.size()==expected.size(),"legacy path size changed");
        for(size_t i=0;i<path.size();++i)
            check((map->floatToInt(path[i])-expected[i]).norm()==0,"legacy raw path changed");
    };
    check(planner.computeGlobalPath(map->intToFloat(expected.front()),map->intToFloat(expected.back())),"legacy solve failed");
    verify();
    check(planner.computeGlobalPath(DMPlanner3D::EndpointAnchors{{expected.front(),0.0}},
                                   DMPlanner3D::EndpointAnchors{{expected.back(),0.0}}),"single zero-connector multi solve failed");
    verify();
    check(planner.getStartAnchorIndex()==0 && planner.getGoalAnchorIndex()==0,"zero-connector selection mismatch");
    check(planner.computeGlobalPath(map->intToFloat(expected.front()),map->intToFloat(expected.back())),"legacy solve after multi failed");
    verify();
    check(planner.getStartAnchorIndex()==-1 && planner.getGoalAnchorIndex()==-1,"legacy API retained stale anchor indices");
}

void reuse_and_validation() {
    std::vector<int8_t> cells(7,0);
    DMP::GraphSearch graph(cells.data(),7,1,1.5,0.1);
    near(graph.plan(0,0,6,0),6.0);
    near(graph.plan({{0,0,0,0.0}},{{6,0,0,0.0}}),6.0);
    near(graph.plan(6,0,0,0),6.0);
    check(graph.getPath().back()->x==6 && graph.getPath().front()->x==0,"reused search retained closed parents");
    check(graph.getStartAnchorIndex()==-1 && graph.getGoalAnchorIndex()==-1,"legacy graph retained anchor selection");
    bool rejected=false;
    try { graph.plan({{0,0,0,-1.0}},{{6,0,0,0.0}}); }
    catch(const std::invalid_argument&) { rejected=true; }
    check(rejected,"negative connector cost accepted");
}

void raw_only() {
    const Vec2i dim(8,7);
    std::vector<int8_t> cells(dim.prod(),0);
    for(int y=0;y<dim.y();++y) cells[3+dim.x()*y]=100;
    auto map=std::make_shared<JPS::MapUtil<2>>();
    map->setMap(Vec2f(-1,-1),dim,cells,0.25);
    DMPlanner2D planner;planner.setEps(1.5);planner.setCweight(0.1);planner.setPotentialMap(map,cells);
    const DMPlanner2D::EndpointAnchors starts{{Vec2i(1,3),0.025},{Vec2i(4,3),1.25}};
    const DMPlanner2D::EndpointAnchors goals{{Vec2i(1,1),100.0},{Vec2i(6,3),0.05}};
    check(planner.computeGlobalPath(starts,goals),"regular anchor plan failed");
    const auto expected=planner.getRawPath();
    const double cost=planner.getPathCost();
    check(!planner.getPath().empty(),"legacy simplified output disappeared");
    check(planner.getStartAnchorIndex()==1 && planner.getGoalAnchorIndex()==1,"unexpected anchor baseline");
    check(planner.computeGlobalRawPath(starts,goals),"raw anchor plan failed");
    check(planner.getPath().empty(),"raw API retained simplified output");
    check(planner.getPathCost()==cost && planner.getStartAnchorIndex()==1 && planner.getGoalAnchorIndex()==1,"raw anchor metadata changed");
    auto compare=[&](const vec_Vec2f& reference) {
        const auto actual=planner.getRawPath();check(actual.size()==reference.size(),"raw path length changed");
        for(std::size_t i=0;i<actual.size();++i) check((actual[i]-reference[i]).norm()==0.0,"raw path order changed");
    };
    compare(expected);
    check(planner.computeGlobalPath(starts,goals) && !planner.getPath().empty(),"regular API after raw failed");
    compare(expected);
    const Vec2f start=map->intToFloat(Vec2i(4,3)),goal=map->intToFloat(Vec2i(6,3));
    check(planner.computeGlobalPath(start,goal),"regular point plan failed");
    const auto single=planner.getRawPath();const double single_cost=planner.getPathCost();
    check(planner.computeGlobalRawPath(start,goal),"raw point plan failed");
    check(planner.getPath().empty() && planner.getPriorPath().empty(),"raw point API retained a prior output");
    check(planner.getPathCost()==single_cost,"raw point cost changed");
    check(planner.getStartAnchorIndex()==-1 && planner.getGoalAnchorIndex()==-1,"raw point API retained anchor indices");
    compare(single);
    check(!planner.computeGlobalRawPath(DMPlanner2D::EndpointAnchors{},goals),"empty raw anchor input succeeded");
    check(planner.getRawPath().empty() && planner.getPath().empty(),"failed raw call retained outputs");
}

int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {
        const std::string name=argv[1];
        if(name=="disconnected_start") disconnected_start();
        else if(name=="disconnected_goal") disconnected_goal();
        else if(name=="virtual_goal") virtual_goal();
        else if(name=="connector_costs") connector_costs();
        else if(name=="no_corner_cut") no_corner_cut();
        else if(name=="metric_units") metric_units();
        else if(name=="legacy_regression") legacy_regression();
        else if(name=="reuse_and_validation") reuse_and_validation();
        else if(name=="raw_only") raw_only();
        else return 3;
        std::cout << name << " passed\n";
    } catch(const std::exception& error) { std::cerr << error.what() << '\n';return 1; }
}
"""


@pytest.fixture(scope="module")
def dmp_endpoint_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native/jps"
    directory = tmp_path_factory.mktemp("dmp_endpoints")
    source = directory / "dmp_endpoints.cpp"
    executable = directory / "dmp_endpoints"
    source.write_text(SOURCE)
    result = subprocess.run([
        compiler, "-O1", "-std=c++17", "-I/usr/include/eigen3", f"-I{vendor / 'include'}",
        str(source), str(vendor / "src/distance_map_planner/graph_search.cpp"),
        str(vendor / "src/distance_map_planner/distance_map_planner.cpp"), "-o", str(executable),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "disconnected_start", "disconnected_goal", "virtual_goal", "connector_costs",
    "no_corner_cut", "metric_units", "legacy_regression", "reuse_and_validation",
    "raw_only",
])
def test_dmp_endpoint_connectors(dmp_endpoint_binary, case):
    result = subprocess.run([str(dmp_endpoint_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
