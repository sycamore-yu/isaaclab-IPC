"""Finite-feature clearance, closed contacts and continuous tip intervals."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

SOURCE = r"""
#include "pathcover_clearance.h"
#include <iomanip>
#include <iostream>
#include <string>

using namespace pathcover_clearance;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance) {
        std::cerr << std::setprecision(18) << message << ": " << actual << " vs " << expected << '\n';
        throw std::runtime_error(message);
    }
}
void witness(const Closest& result, const Box& box) {
    require(result.simplex_point.allFinite() && result.box_point.allFinite(), "nonfinite witness");
    require((result.box_point.array() >= box.lower.array()-1e-12).all()
        && (result.box_point.array() <= box.upper.array()+1e-12).all(), "box witness outside");
    near((result.simplex_point-result.box_point).squaredNorm(), result.squared_distance,
         2e-12, "witness distance");
}
void point_segment() {
    const Box box{Vec3(-1,-1,-1), Vec3(1,1,1)};
    const auto point = pointBoxClosest(Vec3(2,3,4), box);
    near(point.squared_distance, 14.0, 1e-14, "point distance"); witness(point, box);
    const auto crossing = segmentBoxClosest(Vec3(-2,0,0), Vec3(2,0,0), box);
    near(crossing.squared_distance, 0.0, 0.0, "segment intersects face"); witness(crossing, box);
    const auto parallel = segmentBoxClosest(Vec3(-2,1.5,0), Vec3(2,1.5,0), box);
    near(parallel.squared_distance, .25, 1e-14, "parallel segment interior"); witness(parallel, box);
    const auto repeated = segmentBoxClosest(Vec3(2,3,4), Vec3(2,3,4), box);
    near(repeated.squared_distance, 14.0, 1e-14, "zero length segment");
    for (const Vec3 p : {Vec3(1,0,0),Vec3(1,1,0),Vec3(1,1,1)})
        require(!pointFree(p,box,Vec3::Zero()), "zero-radius contact was free");
    require(!segmentFree(Vec3(-2,1,1),Vec3(2,1,1),box,Vec3::Zero()), "edge contact was free");
    require(pointFree(Vec3(1+1e-9,0,0),box,Vec3::Zero()), "zero radius replaced by positive body");
}
void triangle_features() {
    const Box box{Vec3(-.25,-.25,-.25),Vec3(.25,.25,.25)};
    const Triangle crossing{Vec3(-2,-2,0),Vec3(2,-2,0),Vec3(0,2,0)};
    const auto hit = triangleBoxClosest(crossing,box);
    near(hit.squared_distance,0.0,0.0,"triangle SAT missed interior intersection"); witness(hit,box);
    const Triangle parallel{Vec3(-5,-5,3),Vec3(5,-5,3),Vec3(0,5,3)};
    const auto above = triangleBoxClosest(parallel,box);
    near(above.squared_distance,2.75*2.75,1e-13,"face interior distance"); witness(above,box);
    const Box unit{Vec3::Zero(),Vec3::Ones()};
    const Triangle edge{Vec3(3,0,.5),Vec3(0,3,.5),Vec3(3,3,3)};
    const auto closest_edge = triangleBoxClosest(edge,unit);
    near(closest_edge.squared_distance,.5,1e-13,"interior edge-edge minimum"); witness(closest_edge,unit);
    const Triangle repeated{Vec3(2,0,0),Vec3(2,0,0),Vec3(2,0,0)};
    near(triangleBoxClosest(repeated,unit).squared_distance,1.0,1e-14,"point triangle");
    const Triangle line{Vec3(-2,1.5,.5),Vec3(2,1.5,.5),Vec3(0,1.5,.5)};
    near(triangleBoxClosest(line,unit).squared_distance,.25,1e-14,"collinear triangle");
    const Triangle thin{Vec3(-2,1.5,.5),Vec3(2,1.5,.5),Vec3(0,1.5+1e-13,.5)};
    near(triangleBoxClosest(thin,unit).squared_distance,.25,1e-13,"thin triangle");
}
void tetra_features() {
    const Box box{Vec3(-.5,-.5,-.5),Vec3(.5,.5,.5)};
    const Tetrahedron enclosing{Vec3(-4,-4,-4),Vec3(12,-4,-4),Vec3(-4,12,-4),Vec3(-4,-4,12)};
    for (int skip=0;skip<4;++skip) {
        Triangle face; int j=0;
        for(int i=0;i<4;++i) if(i!=skip) face[j++]=enclosing[i];
        require(triangleBoxClosest(face,box).squared_distance>0,"fixture face unexpectedly intersects");
    }
    const auto inside=tetraBoxClosest(enclosing,box);
    near(inside.squared_distance,0.0,0.0,"box-inside-tetra containment missed"); witness(inside,box);
    const Tetrahedron flat{Vec3(-2,-2,2),Vec3(2,-2,2),Vec3(2,2,2),Vec3(-2,2,2)};
    const auto planar=tetraBoxClosest(flat,box);
    near(planar.squared_distance,2.25,1e-13,"coplanar tetra hull"); witness(planar,box);
    const Tetrahedron line{Vec3(-2,2,0),Vec3(2,2,0),Vec3(0,2,0),Vec3(1,2,0)};
    near(tetraBoxClosest(line,box).squared_distance,2.25,1e-13,"line tetra hull");
    const Tetrahedron contained{Vec3::Zero(),Vec3(.1,0,0),Vec3(0,.1,0),Vec3(0,0,.1)};
    near(tetraBoxClosest(contained,box).squared_distance,0.0,0.0,"tetra inside box");
}
void anisotropic_and_support() {
    const Box point{Vec3::Zero(),Vec3::Zero()};
    const Vec3 radii(2,1,.5), p(1.5,.8,0);
    const Triangle triangle{p,p,p};
    const auto result=triangleClosest(triangle,point,radii);
    require(result.free,"anisotropic free point rejected");
    near(result.squared_distance,1.2025,1e-14,"anisotropic metric");
    const Vec3 expected=p.cwiseQuotient(radii.cwiseProduct(radii)).normalized();
    near((result.normal-expected).norm(),0.0,1e-13,"world separation normal uses wrong metric");
    require(result.seed_support>result.obstacle_support,"missing separating support gap");
    const double exact_support=radii.cwiseProduct(result.normal).norm();
    require(result.obstacle_support>=exact_support,"obstacle support is not an upper bound");
    require(result.seed_support<=result.normal.dot(p),"seed support is not a lower bound");
    near(result.obstacle_support,exact_support,2e-14,"excess artificial body margin");
    require(!pointFree(Vec3(1.5,.5,0),point,radii),"anisotropic interior was free");
    require(!pointFree(Vec3(2,0,0),point,radii),"ellipsoid face contact was free");
    require(!pointFree(Vec3(0,0,.5),point,radii),"vertical ellipsoid contact was free");
    require(pointFree(Vec3(0,0,.500001),point,radii),"vertical free point rejected");
    const Box translated{Vec3(-2,3,-4),Vec3(-1,5,-2)};
    const Vec3 n(.3,-.2,.7);
    const Vec3 corner(-1,3,-2);
    const long double expected_support=n.cast<long double>().dot(corner.cast<long double>())
        + radii.cast<long double>().cwiseProduct(n.cast<long double>()).norm();
    require(static_cast<long double>(boxSupportUpper(n,translated,radii))>=expected_support,
            "public support upper bound underestimated");
}
void dense_corner() {
    // Literal coordinates of the real V28 failure. This is the raw .1 m voxel,
    // not its re-rasterized 125-cell dilation; one-ULP lattice construction is
    // separately checked by the frontend integration tests.
    const Vec3 p(-.8190380334854126,26.266569137573242,2.998993158340454);
    const Box raw{Vec3(-1.1,26.0,2.7),Vec3(-1.0,26.1,2.8)};
    const auto distance=pointBoxClosest(p,raw);
    near(std::sqrt(distance.squared_distance),.316371281855716,2e-14,"real raw-cell clearance");
    require(pointFree(p,raw,Vec3::Constant(.2)),"real conservative raster corner remained blocked");
    const Box occupied_raster{Vec3(-.9,26.2,2.9),Vec3(-.8,26.3,3.0)};
    require(!pointFree(p,occupied_raster,Vec3::Zero()),"real inflated-grid fixture not occupied");
}
void continuous_intervals() {
    const Vec3 base(-1,0,0);
    const std::array<Vec3,2> tips{Vec3(1,-2,0),Vec3(1,2,0)};
    const Box point{Vec3::Zero(),Vec3::Zero()};
    const Vec3 radii=Vec3::Constant(.1);
    const auto interval=collisionInterval(base,base,tips,point,radii);
    require(interval.has_value(),"interior collision interval missed");
    const double half=.1/(2*std::sqrt(1-.1*.1));
    require(interval->lower<=.5-half && interval->upper>=.5+half,"interval did not enclose analytic roots");
    near(interval->lower,.5-half,2e-10,"lower collision root");
    near(interval->upper,.5+half,2e-10,"upper collision root");
    require(!collisionInterval(base,base,tips,point,radii,{0,.4}),"free subdomain blocked");
    const auto singleton=collisionInterval(base,base,tips,point,Vec3::Zero());
    require(singleton && singleton->lower<=.5 && singleton->upper>=.5,"zero-radius contact singleton missed");
    require(singleton->upper-singleton->lower<=2e-10,"zero radius replaced by finite body radius");
    const auto narrow=collisionInterval(base,base,tips,point,Vec3::Constant(1e-5));
    require(narrow && narrow->lower<.5 && narrow->upper>.5,"thin interval missed by sampling");
    require(narrow->upper-narrow->lower<1.01e-5,"thin interval width changed");
    const auto full=collisionInterval(Vec3::Zero(),base,tips,point,radii);
    require(full && full->lower==0 && full->upper==1,"fixed-base collision must block every parameter");
    const Box low{Vec3(0,-.3,0),Vec3(0,-.3,0)}, high{Vec3(0,.3,0),Vec3(0,.3,0)};
    const Vec3 small=Vec3::Constant(.03);
    const auto first=collisionInterval(base,base,tips,low,small);
    const auto second=collisionInterval(base,base,tips,high,small);
    require(first && second && first->upper<second->lower,"two obstacles should have disjoint forbidden intervals");
    const Triangle middle{base,base,Vec3(1,0,0)};
    require(triangleFree(middle,low,small) && triangleFree(middle,high,small),"free gap between obstacles lost");
}
void invalid_inputs() {
    const Box box{Vec3::Zero(),Vec3::Ones()};
    int rejected=0;
    for(const Vec3 r : {Vec3(1,1,0),Vec3(0,0,1),Vec3(-1,1,1)}) {
        try {pointFree(Vec3(2,2,2),box,r);} catch(const std::invalid_argument&) {++rejected;}
    }
    require(rejected==3,"partial-zero or negative radii accepted");
    try {pointFree(Vec3(std::numeric_limits<double>::quiet_NaN(),0,0),box,Vec3::Zero());}
    catch(const std::invalid_argument&) {++rejected;}
    try {collisionInterval(Vec3::Zero(),Vec3::Ones(),{Vec3::Zero(),Vec3::Ones()},box,Vec3::Ones(),{.7,.2});}
    catch(const std::invalid_argument&) {++rejected;}
    require(rejected==5,"nonfinite point or reversed domain accepted");
}
void oracle_query() {
    int count;
    while(std::cin>>count) {
        Box box;
        for(int k=0;k<3;++k)std::cin>>box.lower[k];
        for(int k=0;k<3;++k)std::cin>>box.upper[k];
        std::array<Vec3,4> p;
        for(int i=0;i<count;++i)for(int k=0;k<3;++k)std::cin>>p[i][k];
        Closest result;
        if(count==1)result=pointBoxClosest(p[0],box);
        else if(count==2)result=segmentBoxClosest(p[0],p[1],box);
        else if(count==3)result=triangleBoxClosest(Triangle{p[0],p[1],p[2]},box);
        else if(count==4)result=tetraBoxClosest(p,box);
        else throw std::invalid_argument("oracle point count");
        std::cout<<std::setprecision(17)<<result.squared_distance;
        for(int k=0;k<3;++k)std::cout<<' '<<result.simplex_point[k];
        for(int k=0;k<3;++k)std::cout<<' '<<result.box_point[k];
        std::cout<<'\n';
    }
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)return 2;
        const std::string name=argv[1];
        if(name=="point_segment")point_segment();
        else if(name=="triangle_features")triangle_features();
        else if(name=="tetra_features")tetra_features();
        else if(name=="anisotropic_and_support")anisotropic_and_support();
        else if(name=="dense_corner")dense_corner();
        else if(name=="continuous_intervals")continuous_intervals();
        else if(name=="invalid_inputs")invalid_inputs();
        else if(name=="oracle") {oracle_query();return 0;}
        else return 3;
        std::cout<<name<<" passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def clearance_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("pathcover_clearance")
    source = directory / "clearance.cpp"
    executable = directory / "clearance"
    source.write_text(SOURCE)
    result = subprocess.run([
        compiler, "-O2", "-std=c++17", "-ffp-contract=off", "-I/usr/include/eigen3",
        f"-I{root / 'native'}", str(source), "-o", str(executable),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "point_segment", "triangle_features", "tetra_features", "anisotropic_and_support",
    "dense_corner", "continuous_intervals", "invalid_inputs",
])
def test_clearance_geometry(clearance_binary, case):
    result = subprocess.run([str(clearance_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_closest_distance_against_independent_convex_program(clearance_binary):
    optimize = pytest.importorskip("scipy.optimize")
    random = np.random.default_rng(285)
    problems = []
    for count in (1, 2, 3, 4):
        for shift in (np.zeros(3), np.array([2.5, 0.4, -0.3]), np.array([-1.0, 2.3, 1.4])):
            lower = random.uniform(-0.9, -0.4, 3)
            upper = random.uniform(0.4, 0.9, 3)
            points = random.uniform(-1.4, 1.4, (count, 3)) + shift
            problems.append((points, lower, upper))
    lines = [" ".join(map(str, [len(p), *lo, *hi, *p.ravel()])) for p, lo, hi in problems]
    query = subprocess.run([str(clearance_binary), "oracle"], input="\n".join(lines)+"\n",
                           capture_output=True, text=True)
    assert query.returncode == 0, query.stdout + query.stderr
    values = np.asarray([[float(v) for v in line.split()] for line in query.stdout.splitlines()])
    assert values.shape == (len(problems), 7)
    for (points, lower, upper), result in zip(problems, values, strict=True):
        count = len(points)
        matrix = np.column_stack((points.T, -np.eye(3)))
        x0 = np.r_[np.full(count, 1/count), np.clip(points.mean(0), lower, upper)]
        equality = np.r_[np.ones(count), np.zeros(3)]
        solved = optimize.minimize(
            lambda x, matrix=matrix: float((matrix @ x) @ (matrix @ x)), x0,
            jac=lambda x, matrix=matrix: 2 * matrix.T @ (matrix @ x), method="SLSQP",
            bounds=[(0.0, 1.0)] * count + list(zip(lower, upper, strict=True)),
            constraints={"type": "eq", "fun": lambda x, equality=equality: float(equality @ x - 1),
                         "jac": lambda x, equality=equality: equality},
            options={"ftol": 1e-13, "maxiter": 200},
        )
        assert solved.success, solved.message
        assert abs(result[0] - solved.fun) < 2e-10
        np.testing.assert_allclose(np.sum((result[1:4]-result[4:7])**2), result[0], atol=2e-12)
        assert np.all(result[4:7] >= lower-2e-12) and np.all(result[4:7] <= upper+2e-12)
        # Independently verify the returned simplex witness has nonnegative
        # barycentric coordinates; a distance alone does not prove a valid pair.
        membership = optimize.linprog(
            np.zeros(count), A_eq=np.vstack((points.T, np.ones(count))),
            b_eq=np.r_[result[1:4], 1.0], bounds=(0, None), method="highs",
        )
        assert membership.success, membership.message
