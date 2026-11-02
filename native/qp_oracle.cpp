// Test-only adapter to the original Docker OSQP-Eigen build.
#include <Eigen/Eigen>
#include <OsqpEigen/OsqpEigen.h>
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 if(argc!=3 && argc!=4)return 2;
 std::ifstream in(argv[1]); int n,m;in>>n>>m;
 if(n!=45||m<135||m>10000)return 3;
 Eigen::MatrixXd H(n,n),A(m,n);Eigen::VectorXd g(n),l(m),u(m);
 for(int i=0;i<n;++i)for(int j=0;j<n;++j)in>>H(i,j);
 for(int i=0;i<n;++i)in>>g[i];
 for(int i=0;i<m;++i)for(int j=0;j<n;++j)in>>A(i,j);
 for(int i=0;i<m;++i)in>>l[i];for(int i=0;i<m;++i)in>>u[i];
 if(!in)return 4;
 OsqpEigen::Solver s; s.settings()->setVerbosity(false);s.settings()->setWarmStart(true);
 if(argc==4){s.settings()->setAbsoluteTolerance(1e-8);s.settings()->setRelativeTolerance(1e-8);s.settings()->setMaxIteration(20000);}
 s.data()->setNumberOfVariables(n);s.data()->setNumberOfConstraints(m);
 Eigen::SparseMatrix<double> hs=H.sparseView(),as=A.sparseView();
 s.data()->setHessianMatrix(hs);s.data()->setGradient(g);s.data()->setLinearConstraintsMatrix(as);s.data()->setLowerBound(l);s.data()->setUpperBound(u);
 if(!s.initSolver())return 5; bool ok=s.solve();
 std::ofstream out(argv[2]);out<<std::setprecision(17)<<ok<<"\n"<<s.getSolution()<<"\n";
 return 0;
}
