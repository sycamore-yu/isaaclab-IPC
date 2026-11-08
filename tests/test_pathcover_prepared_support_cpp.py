"""Prepared support keeps directed rounding; only exact point simplices collapse."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include "pathcover_clearance.h"
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

using namespace pathcover_clearance;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::uint64_t bits(double value) {
    std::uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
void same(double first, double second, const char* message) {
    require(bits(first) == bits(second), message);
}
void same(const Vec3& first, const Vec3& second, const char* message) {
    for (int i=0;i<3;++i) same(first[i],second[i],message);
}
void same(const Closest& first, const Closest& second) {
    same(first.squared_distance,second.squared_distance,"point distance bits changed");
    same(first.simplex_point,second.simplex_point,"point witness bits changed");
    same(first.box_point,second.box_point,"box witness bits changed");
}
void same(const Clearance& first, const Clearance& second) {
    same(first.squared_distance,second.squared_distance,"clearance distance bits changed");
    same(first.simplex_point,second.simplex_point,"clearance point witness bits changed");
    same(first.box_point,second.box_point,"clearance box witness bits changed");
    same(first.normal,second.normal,"clearance normal bits changed");
    same(first.seed_support,second.seed_support,"seed support bits changed");
    same(first.obstacle_support,second.obstacle_support,"obstacle support bits changed");
    require(first.free==second.free,"closed-contact decision changed");
}
template<class Function> int rejection(Function function) {
    try {function();return 0;}
    catch(const std::invalid_argument&) {return 1;}
    catch(const std::overflow_error&) {return 2;}
}

void support_bits() {
    const double tiny=std::numeric_limits<double>::denorm_min();
    const std::array<Vec3,6> normals{Vec3(1,0,0),Vec3(-.3,.2,-.7),
        Vec3(0.0,-0.0,0.0),Vec3(tiny,-tiny,tiny),Vec3(1e100,-1e100,.125),
        Vec3(1e308,0,0)};
    const std::array<Vec3,3> radii{Vec3::Zero(),Vec3::Constant(.2),Vec3(.1,.3,.7)};
    const std::array<Box,3> boxes{
        Box{Vec3(-2,3,-4),Vec3(-1,5,-2)},
        Box{Vec3(-.1,-.2,-.3),Vec3(.2,.3,.4)},
        Box{Vec3(-0.0,0.0,-0.0),Vec3(0.0,0.0,0.0)}};
    for(const auto& normal:normals) for(const auto& radius:radii) {
        const PreparedBoxSupport prepared(normal,radius);
        // Reuse one prepared cut across distinct boxes; signed-zero and
        // overflow results keep the original public function's exact bits.
        for(const auto& box:boxes)
            same(prepared.upper(box),boxSupportUpper(normal,box,radius),
                 "prepared support changed directed arithmetic");
    }
}

void support_rejections() {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    const Vec3 normal(.3,-.2,.7),radius(.2,.2,.2);
    const Box valid{Vec3::Zero(),Vec3::Ones()};
    auto check=[&](const Vec3& n,const Vec3& r,const Box& b) {
        const auto original=rejection([&] {volatile double v=boxSupportUpper(n,b,r);(void)v;});
        const auto prepared=rejection([&] {volatile double v=PreparedBoxSupport(n,r).upper(b);(void)v;});
        require(original==1 && prepared==original,"prepared support changed invalid-input class");
    };
    for(const auto& n:std::array<Vec3,2>{Vec3(nan,0,0),Vec3(inf,0,0)})check(n,radius,valid);
    for(const auto& r:std::array<Vec3,5>{Vec3(0,.2,.2),Vec3(.2,0,0),
        Vec3(-.2,.2,.2),Vec3(nan,.2,.2),Vec3(inf,.2,.2)})check(normal,r,valid);
    for(const auto& b:std::array<Box,3>{Box{Vec3(2,0,0),Vec3::Ones()},
        Box{Vec3(nan,0,0),Vec3::Ones()},Box{Vec3::Zero(),Vec3(inf,1,1)}})check(normal,radius,b);
}

void repeated_points() {
    const Box box{Vec3::Zero(),Vec3::Ones()};
    const std::array<Vec3,7> points{Vec3(.5,.5,.5),Vec3(1,.5,.5),Vec3(1,1,1),
        Vec3(2,3,4),Vec3(-2,.3,.6),Vec3(1.2,.5,.5),
        Vec3(std::nextafter(1.2,2.0),.5,.5)};
    for(const auto& p:points) {
        const Triangle triangle{p,p,p};
        same(triangleBoxClosest(triangle,box),pointBoxClosest(p,box));
        for(const auto& r:std::array<Vec3,3>{Vec3::Zero(),Vec3::Constant(.2),Vec3(.2,.3,.1)})
            same(triangleClosest(triangle,box,r),detail::clearance(std::array<Vec3,1>{p},box,r));
    }
    const Vec3 contact(1.2,.5,.5);
    require(!triangleFree({contact,contact,contact},box,Vec3::Constant(.2)),
            "positive-radius contact was changed to free");
    // A one-ULP vertex difference is not an exact repeated point. Its closer
    // vertex must still determine the full finite-feature minimum.
    const Vec3 p(2,.5,.5),q(std::nextafter(2.0,0.0),.5,.5),r(2,.5+1e-13,.5);
    const auto thin=triangleBoxClosest({p,q,r},box);
    same(thin,pointBoxClosest(q,box));
    require(bits(thin.squared_distance)!=bits(pointBoxClosest(p,box).squared_distance),
            "near-degenerate triangle was incorrectly collapsed");
    for(const auto& bad:std::array<Vec3,3>{Vec3(std::numeric_limits<double>::quiet_NaN(),0,0),
        Vec3(std::numeric_limits<double>::infinity(),0,0),Vec3::Constant(1e200)}) {
        const auto single=rejection([&] {(void)pointBoxClosest(bad,box);});
        const auto repeated=rejection([&] {(void)triangleBoxClosest({bad,bad,bad},box);});
        require(single!=0 && repeated==single,"point shortcut changed numeric rejection class");
    }
}

void adjacent_contract() {
    // Numeric boundary behavior includes errno and the floating environment.
    // Compare the platform function directly, without reproducing its formula.
    double (*volatile reference)(double,double) = &std::nextafter;
    std::vector<std::uint64_t> patterns;
    const std::array<std::uint64_t,14> edges{
        0,1,UINT64_C(0x000ffffffffffffe),UINT64_C(0x000fffffffffffff),
        UINT64_C(0x0010000000000000),UINT64_C(0x0010000000000001),
        UINT64_C(0x3fefffffffffffff),UINT64_C(0x3ff0000000000000),
        UINT64_C(0x3ff0000000000001),UINT64_C(0x7feffffffffffffe),
        UINT64_C(0x7fefffffffffffff),UINT64_C(0x7ff0000000000000),
        UINT64_C(0x7ff8000000000042),UINT64_C(0x7ff0000000000042)};
    for (const auto sign : {UINT64_C(0),UINT64_C(0x8000000000000000)})
        for (const auto edge : edges) patterns.push_back(sign | edge);
    std::uint64_t random = UINT64_C(0x79c3a63d10e59b71);
    for (int i=0;i<2048;++i) {
        random ^= random << 13; random ^= random >> 7; random ^= random << 17;
        patterns.push_back(random);
    }
    fenv_t environment;
    require(fegetenv(&environment)==0,"could not save floating environment");
    for (const int rounding : {FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}) {
        require(fesetround(rounding)==0,"rounding mode unavailable");
        for (const int preset : {0,FE_INEXACT | FE_DIVBYZERO})
            for (const int direction : {-1,1}) for (const auto pattern : patterns) {
                double value;
                std::memcpy(&value,&pattern,sizeof(value));
                const double target = direction > 0 ? std::numeric_limits<double>::infinity()
                                                    : -std::numeric_limits<double>::infinity();
                feclearexcept(FE_ALL_EXCEPT);
                if (preset) feraiseexcept(preset);
                errno=EDOM;
                const double expected=reference(value,target);
                const int expected_errno=errno, expected_flags=fetestexcept(FE_ALL_EXCEPT);
                feclearexcept(FE_ALL_EXCEPT);
                if (preset) feraiseexcept(preset);
                errno=EDOM;
                const double actual=direction > 0 ? detail::up(value) : detail::down(value);
                const int actual_errno=errno, actual_flags=fetestexcept(FE_ALL_EXCEPT);
                same(actual,expected,"adjacent double bits changed");
                require(actual_errno==expected_errno,"adjacent errno changed");
                require(actual_flags==expected_flags,"adjacent floating exceptions changed");
            }
    }
    require(fesetenv(&environment)==0,"could not restore floating environment");
}

void support_filter_contract() {
    fenv_t environment;
    require(fegetenv(&environment)==0,"could not save floating environment");
    for (const int rounding : {FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}) {
        require(fesetround(rounding)==0,"rounding mode unavailable");
        for (const int exponent : {-500,-150,0,150,500}) for (int i=0;i<256;++i) {
            const double scale=std::ldexp(1.0,exponent);
            const Vec3 center=Vec3((i%23)-11.0,(i%17)-8.0,(i%13)-6.0)*scale;
            const Vec3 width=Vec3(.05+(i%5)*.25,.25,.5)*scale;
            const Box box{center-width,center+width};
            const Vec3 normal=Vec3(.2+(i%3)*.1,-.7,.125)*scale;
            for (const auto& radii : {Vec3::Zero().eval(),Vec3(.2,.3,.1)}) {
                const PreparedBoxSupport prepared(normal,radii);
                const double expected=prepared.upper(box);
                const std::array<double,7> thresholds{expected,
                    std::nextafter(expected,std::numeric_limits<double>::infinity()),
                    std::nextafter(expected,-std::numeric_limits<double>::infinity()),
                    -0.0,0.0,-std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN()};
                for (const double threshold : thresholds) {
                    require(prepared.below(box,threshold)==(expected<threshold),
                            "checked filter changed an old support decision");
                    require(prepared.belowValidated(box,threshold)==(expected<threshold),
                            "validated filter changed an old support decision");
                }
            }
        }
    }
    require(fesetenv(&environment)==0,"could not restore floating environment");
    const PreparedBoxSupport prepared(Vec3(1,0,0),Vec3::Constant(.2));
    const Box invalid{Vec3(2,0,0),Vec3::Ones()};
    require(rejection([&] {(void)prepared.below(invalid,0.0);})==1,
            "checked filter accepted an invalid box");
}

int main(int argc,char** argv) {
    try {
        if(argc!=2)return 2;
        const std::string name=argv[1];
        if(name=="support_bits")support_bits();
        else if(name=="support_rejections")support_rejections();
        else if(name=="repeated_points")repeated_points();
        else if(name=="adjacent_contract")adjacent_contract();
        else if(name=="support_filter_contract")support_filter_contract();
        else return 3;
        std::cout<<name<<" passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def prepared_support_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("prepared_box_support")
    source = directory / "prepared_box_support.cpp"
    executable = directory / "prepared_box_support"
    source.write_text(SOURCE)
    result = subprocess.run([
        compiler, "-O3", "-DNDEBUG", "-std=c++17", "-ffp-contract=off", "-frounding-math",
        "-fno-builtin-nextafter", "-I/usr/include/eigen3",
        f"-I{root / 'native'}", str(source), "-o", str(executable),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "support_bits", "support_rejections", "repeated_points", "adjacent_contract", "support_filter_contract",
])
def test_prepared_support_and_exact_point_contract(prepared_support_binary, case):
    result = subprocess.run([str(prepared_support_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
