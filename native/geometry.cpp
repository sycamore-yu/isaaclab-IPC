// Local bindings to hku-mars/IPC. Upstream notices remain in generated headers.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "local_astar.cpp"
#include "PathCover.hpp"
namespace py=pybind11;
using Points=std::vector<Eigen::Vector3d>;
struct CorridorError:std::runtime_error{using std::runtime_error::runtime_error;};
Eigen::MatrixXd corridor(const Eigen::Vector3d&a,const Eigen::Vector3d&end,const Points&points,double radius){
 if(!a.allFinite()||!end.allFinite()||!std::isfinite(radius)||radius<0) throw std::invalid_argument("Invalid corridor input");
 Eigen::Matrix<double,-1,3> bounds(6,3); bounds<<1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1;
 Eigen::VectorXd limits(6); limits<<a.x()+10,10-a.x(),a.y()+10,10-a.y(),3,-.5;
 for(const auto&p:points){if(!p.allFinite())throw std::invalid_argument("Nonfinite point");if((p-a).norm()<1e-8)throw CorridorError("Obstacle coincides with SFC seed");}
 Eigen::Matrix<double,-1,3>A;Eigen::VectorXd b;std::vector<int> remaining;
 if(points.empty()){A=bounds;b=limits;}
 else{
  try { PathCover::RISP<double,3>(points,a,bounds,limits,A,b,remaining); }
  catch(const std::exception&e){throw CorridorError(e.what());}
  // The IPC cloud is not inflated; preserve its metric robot clearance here.
  b.array()-=radius*A.rowwise().norm().array();
 }
 if(!A.allFinite()||!b.allFinite()||((A*a-b).array()>1e-8).any())throw CorridorError("SFC seed violates clearance");
 if(((A*end-b).array()>1e-8).any())return Eigen::MatrixXd(0,4);
 Eigen::MatrixXd planes(A.rows(),4);planes.leftCols(3)=A;planes.col(3)=-b;return planes;
}
PYBIND11_MODULE(_geometry,m){
    py::register_exception<CorridorError>(m,"CorridorError");
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
