#include "RadialProfile.hpp"
#include <filesystem>
#include <iostream>
int main(int argc,char **argv) {
  using namespace Theseus;
  auto check=[](bool ok){if(!ok) throw std::runtime_error("Radial profile assertion failed");};
  auto close=[&](double x,double y){check(std::abs(x-y)<1e-10);};
  RadialProfile p; p.rows={{0.1,1000,20,-2},{0.3,2000,40,-4}};
  auto a=p.Evaluate(0); close(a[1],1000);close(a[2],20);close(a[3],0);
  a=p.Evaluate(0.05);close(a[3],-1);
  a=p.Evaluate(0.2);close(a[1],1500);close(a[2],30);close(a[3],-3);
  bool rejected=false;try{p.Evaluate(0.4);}catch(const std::runtime_error&){rejected=true;}check(rejected);
  const auto u=CPGState(10000,1000,20,-2,1.4,287.05);
  close(u[1]/u[0],20);close(u[2]/u[0],-2);
  close((u[3]-(u[1]*u[1]+u[2]*u[2])/(2*u[0]))*.4,10000);
  auto real=RadialProfile::Read(argv[1],0), duplicate=RadialProfile::Read(argv[1],1);
  check(real.rows==duplicate.rows); check(real.rows.size()==335);
  close(real.Evaluate(0)[3],0);check(real.Evaluate(.0868)[1]>0);
  rejected=false;try{RadialProfile::Read(argv[1],2);}catch(const std::runtime_error&){rejected=true;}check(rejected);
  std::filesystem::path bad=std::filesystem::temp_directory_path()/"theseus-profile-invalid.dat";
  {std::ofstream f(bad);f<<"2 1\n.1 0 300 1 0 0\n.1 0 300 1 0 0\n";}
  rejected=false;try{RadialProfile::Read(bad.string(),0);}catch(const std::runtime_error&){rejected=true;}check(rejected);
  {std::ofstream f(bad);f<<"2 1\n.1 0 -1 1 0 0\n.2 0 300 1 0 0\n";}
  rejected=false;try{RadialProfile::Read(bad.string(),0);}catch(const std::runtime_error&){rejected=true;}check(rejected);
  std::filesystem::remove(bad);
  std::cout<<"Profile interpolation, parity, signed velocities, CPG closure and invalid inputs passed\n";
}
