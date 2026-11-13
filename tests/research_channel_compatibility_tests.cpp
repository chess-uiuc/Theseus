#include "PirozzoliFlux.hpp"
#include "LTETransport.hpp"
#include <cmath>
#include <iostream>

struct Table { struct Layout { static constexpr int mu_idx=0, lambda_idx=1; } L; };
struct EOS
{
  template<typename State>
  MFEM_HOST_DEVICE mfem::real_t temperature(const Theseus::PhysicsConstants &,
    const Theseus::StateLayout &, const State &, const Table &) const { return 400.0; }
  template<typename State>
  MFEM_HOST_DEVICE mfem::real_t property_lookup(int, const Theseus::PhysicsConstants &,
    const Theseus::StateLayout &, const State &, const Table &) const { return 0.125; }
};
int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend=argc>1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if(backend=="cuda" && !mfem::Device::Allows(mfem::Backend::CUDA)) return 1;
  Theseus::PhysicsConstants physics(1.4, .72, 287.05, .02, .162668791274317);
  Theseus::StateLayout layout(2,1);
  Theseus::IdealGasModel gas(physics,layout);
  mfem::Vector states(8), normal(2), output(6);
  for(auto *v:{&states,&normal,&output}) v->UseDevice();
  const double rho[2]={1.2,.95}, ux[2]={20,-5}, uy[2]={3,7}, pressure[2]={101325,99000};
  for(int i=0;i<2;++i)
    {
      states.HostReadWrite()[4*i]=rho[i];
      states.HostReadWrite()[4*i+1]=rho[i]*ux[i];
      states.HostReadWrite()[4*i+2]=rho[i]*uy[i];
      states.HostReadWrite()[4*i+3]=pressure[i]/.4+.5*rho[i]*(ux[i]*ux[i]+uy[i]*uy[i]);
    }
  normal.HostReadWrite()[0]=.3; normal.HostReadWrite()[1]=-1.2;
  const auto *u=states.Read(), *n=normal.Read(); auto *result=output.Write();
  mfem::forall(1,[=] MFEM_HOST_DEVICE(int)
  {
    Theseus::PirozzoliFlux::ComputeFaceFluxKernel(gas,u,u+4,n,result);
    const Theseus::LTETransport transport;
    result[4]=transport.viscosity(physics,layout,EOS{},Theseus::PointStateView{u},Table{});
    result[5]=transport.thermal_conductivity(physics,layout,EOS{},Theseus::PointStateView{u},Table{});
  });
  const double vn=.5*((ux[0]+ux[1])*.3+(uy[0]+uy[1])*(-1.2));
  const double rm=.5*(rho[0]+rho[1]), pm=.5*(pressure[0]+pressure[1]);
  const double h=.5*3.5*(pressure[0]/rho[0]+pressure[1]/rho[1]);
  const double ke=.25*(ux[0]*ux[0]+uy[0]*uy[0]+ux[1]*ux[1]+uy[1]*uy[1]);
  const double speed0=std::abs(ux[0]*.3+uy[0]*(-1.2))+std::sqrt(1.4*pressure[0]/rho[0])*std::sqrt(1.53);
  const double speed1=std::abs(ux[1]*.3+uy[1]*(-1.2))+std::sqrt(1.4*pressure[1]/rho[1])*std::sqrt(1.53);
  const double lambda=.5*std::max(speed0,speed1);
  double expected[6]={rm*vn,rm*vn*.5*(ux[0]+ux[1])+pm*.3,
    rm*vn*.5*(uy[0]+uy[1])+pm*(-1.2),rm*(h+ke)*vn,0,0};
  for(int q=0;q<4;++q) expected[q]-=lambda*(states.HostRead()[4+q]-states.HostRead()[q]);
  expected[4]=physics.mu0*(physics.T0+physics.Ts)*std::pow(400/physics.T0,1.5)/(400+physics.Ts);
  expected[5]=expected[4]*3.5*287.05/.72;
  const auto *actual=output.HostRead();
  for(int q=0;q<6;++q)
    if(!std::isfinite(actual[q]) || std::abs(actual[q]-expected[q])>2e-12*(1+std::abs(expected[q])))
      {std::cerr<<q<<": "<<actual[q]<<" expected "<<expected[q]<<'\n';return 1;}
  std::cout<<"Research Pirozzoli/Lax and Sutherland fixed-cp conductivity: PASS\n";
  return 0;
}
