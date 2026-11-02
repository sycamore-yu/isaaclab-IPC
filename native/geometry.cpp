// Local bindings to hku-mars/IPC. Upstream notices remain in generated headers.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "local_astar.cpp"
#include "emvp.hpp"
namespace py=pybind11;
using Points=std::vector<Eigen::Vector3d>;
Eigen::MatrixXd corridor(const Eigen::Vector3d&a,const Eigen::Vector3d&b,const Points&points,double radius){
    if(!a.allFinite()||!b.allFinite()||!std::isfinite(radius)||radius<0) throw std::invalid_argument("Invalid corridor input");
    Eigen::Matrix<double,6,4> bd; bd.setZero();
    bd.block<3,3>(0,0)=Eigen::Matrix3d::Identity();
    bd.block<3,3>(3,0)=-Eigen::Matrix3d::Identity();
    bd.col(3)<<-a.x()-10,-a.y()-10,-3,a.x()-10,a.y()-10,-0.5;
    if(points.empty()) return bd;
    // EMVP's tie breaking depends on the original nonempty-cloud plane order.
    Eigen::Matrix<double,6,4> ordered;
    ordered<<bd.row(0),bd.row(3),bd.row(1),bd.row(4),bd.row(2),bd.row(5);
    bd=ordered;
    Eigen::Matrix3Xd cloud(3,points.size());
    for(size_t i=0;i<points.size();++i){
        if(!points[i].allFinite()) throw std::invalid_argument("Nonfinite point");
        cloud.col(i)=points[i];
    }
    Polytope p;
    if(!emvp::emvp(bd,cloud,a,b,p,radius,false,1)) return Eigen::MatrixXd(0,4);
    return p.GetPlanes();
}
PYBIND11_MODULE(_geometry,m){
    m.def("corridor",&corridor,py::call_guard<py::gil_scoped_release>());
    py::class_<LoaclAstarClass>(m,"Grid")
        .def(py::init([](){auto p=std::make_unique<LoaclAstarClass>(); p->InitMap(.1,{-10,-10,0},{10,10,3});p->Reset();return p;}))
        .def("center",&LoaclAstarClass::SetCenter)
        .def("cloud",&LoaclAstarClass::setObsVector,py::call_guard<py::gil_scoped_release>())
        .def("free",&LoaclAstarClass::CheckPoint)
        .def("endpoint_free",&LoaclAstarClass::CheckStartEnd)
        .def("line_free",&LoaclAstarClass::CheckLineObstacleFree)
        .def("path_free",&LoaclAstarClass::CheckPathFree)
        .def("search",[](LoaclAstarClass&g,Eigen::Vector3d a,Eigen::Vector3d b){Points path,way; if(g.SearchPath(a,b)){g.GetPath(path);g.FloydHandle(path,way);}g.Reset();return way;},py::call_guard<py::gil_scoped_release>());
}
