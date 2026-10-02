// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"
#include "InitialCondition.hpp"
#include "BoundaryStateConfig.hpp"
#include "BoundaryStatePacking.hpp"
#include "bc_kernels.hpp"
#include <filesystem>
#include <limits>
using namespace Theseus;

namespace {
template<class F> bool rejects(F f)
{
  try { f(); } catch (const std::invalid_argument &) { return true; }
  return false;
}
// Analytic table: p=10*rho*T, e=500+2*T. The energy reference offset
// deliberately differs from CPG and inverse interpolation is exact.
struct TableFixture
{
  LTETable::Data data;
  LTETable::LTETables tables{3,3};
  TableFixture()
  {
    data.rho_grid.SetSize(3); data.T_grid.SetSize(3); data.e_grid.SetSize(3);
    data.lte_table.SetSize(81); data.lte_table = 0.0;
    data.inv_table.SetSize(9);
    for (int i=0;i<3;++i) {
      data.rho_grid[i]=1+i; data.T_grid[i]=100+100*i; data.e_grid[i]=700+200*i;
    }
    for (int j=0;j<3;++j) for (int i=0;i<3;++i) {
      const auto set = [&](int k, double v) { data.lte_table[tables.L.property_index(k,i,j)]=v; };
      set(tables.L.P_idx,10*data.rho_grid[i]*data.T_grid[j]);
      set(tables.L.e_idx,500+2*data.T_grid[j]); set(tables.L.cv_idx,2);
      data.inv_table[tables.L.property_index(0,i,j)]=data.T_grid[j];
    }
    tables.tables={data.lte_table.GetData(),data.inv_table.GetData(),
      data.rho_grid.GetData(),data.T_grid.GetData(),data.e_grid.GetData()};
  }
};
}

TEST(PhysicalState_CPGPairsAndPrimitiveDispatch)
{
  for(int dim=1;dim<=3;++dim) {
    auto gas=std::make_shared<IdealGasModel>(PhysicsConstants(1.4,0.72,287,0.02),StateLayout(dim,7));
    GasModelInterfaceT<IdealGasModel> bridge(gas);
    const double rho=2, T=300, p=rho*287*T;
    std::vector<double> velocity(dim,-3); velocity[0]=10;
    for (const ThermodynamicInput &pair : std::vector<ThermodynamicInput>{
         PressureTemperature{p,T},DensityTemperature{rho,T},DensityPressure{rho,p}}) {
      mfem::Vector u;
      bridge.ConservativeFromPhysical({pair,velocity},u);
      EXPECT_EQ(u.Size(),dim+2);
      PointStateView s(u.GetData());
      EXPECT_CLOSE(gas->density(s),rho,1e-12);
      EXPECT_CLOSE(gas->pressure(s),p,1e-8);
      EXPECT_CLOSE(gas->temperature(s),T,1e-10);
      EXPECT_CLOSE(u[dim+1],p/0.4+rho*0.5*(100+9*(dim-1)),1e-8);
      mfem::Vector primitive(dim+2), converted(dim+2);
      primitive[0]=rho; for(int d=0;d<dim;++d) primitive[d+1]=velocity[d];
      primitive[dim+1]=p;
      PointPrimitiveView prim(primitive.GetData()); PointStateViewRW cons(converted.GetData());
      gas->primitive_to_conserved(prim,cons);
      for(int q=0;q<u.Size();++q) EXPECT_CLOSE(converted[q],u[q],1e-8);
    }
  }
  return 0;
}

TEST(PhysicalState_LTEPairsRoundTripAndEndpoints)
{
  TableFixture fixture;
  for (int dim=1;dim<=3;++dim) {
    auto gas=std::make_shared<LTEGas>(PhysicsConstants(1.4,0.72,10,0.02),StateLayout(dim,7),fixture.tables);
    GasModelInterfaceT<LTEGas> bridge(gas);
    for (double rho : {1.,1.5,3.}) for (double T : {100.,175.,300.}) {
      for (const ThermodynamicInput &pair : std::vector<ThermodynamicInput>{
           PressureTemperature{10*rho*T,T},DensityTemperature{rho,T},DensityPressure{rho,10*rho*T}}) {
        mfem::Vector u;
        bridge.ConservativeFromPhysical({pair,std::vector<double>(dim,2)},u);
        PointStateView s(u.GetData());
        EXPECT_CLOSE(u[0],rho,1e-12);
        EXPECT_CLOSE(u[dim+1],rho*(500+2*T+2*dim),1e-10);
        EXPECT_CLOSE(gas->temperature(s),T,1e-10);
        EXPECT_CLOSE(gas->pressure(s),10*rho*T,1e-9);
      }
    }
  }
  return 0;
}

TEST(PhysicalState_InvalidInputsAndAtomicOutput)
{
  IdealGasModel cpg(PhysicsConstants(1.4,0.72,287,0.02),StateLayout(2,1));
  mfem::Vector out(1); out[0]=123;
  const auto nan=std::numeric_limits<double>::quiet_NaN();
  for (const PhysicalStateInput &bad : std::vector<PhysicalStateInput>{
      {PressureTemperature{-1,300},{0,0}}, {PressureTemperature{1,nan},{0,0}},
      {DensityTemperature{0,300},{0,0}}, {DensityPressure{1,0},{0,0}},
      {PressureTemperature{1,300},{0}}, {PressureTemperature{1,300},{0,nan}},
      {PressureTemperature{1,300},{1e308,0}}})
    EXPECT_TRUE(rejects([&]{cpg.ConservativeFromPhysical(bad,out);}));
  IdealGasModel scalars(cpg.phys,StateLayout(2,1,1));
  EXPECT_TRUE(rejects([&]{scalars.ConservativeFromPhysical({DensityTemperature{1,300},{0,0}},out);}));
  EXPECT_EQ(out.Size(),1); EXPECT_EQ(out[0],123);
  return 0;
}

TEST(PhysicalState_LTEBoundsAndAmbiguousRoots)
{
  TableFixture fixture;
  LTEGas gas(PhysicsConstants(1.4,0.72,10,0.02),StateLayout(2,1),fixture.tables);
  mfem::Vector out(1); out[0]=123;
  for (const ThermodynamicInput &bad : std::vector<ThermodynamicInput>{
       DensityTemperature{0.5,200},DensityTemperature{2,301},
       PressureTemperature{100000,200},DensityPressure{2,100000}})
    EXPECT_TRUE(rejects([&]{gas.ConservativeFromPhysical({bad,{0,0}},out);}));
  fixture.data.e_grid[0]=800;
  EXPECT_TRUE(rejects([&]{gas.ConservativeFromPhysical({DensityTemperature{2,100},{0,0}},out);}));
  fixture.data.e_grid[0]=700;
  fixture.data.inv_table=400.0;
  EXPECT_TRUE(rejects([&]{gas.ConservativeFromPhysical({DensityTemperature{2,200},{0,0}},out);}));
  for(int j=0;j<3;++j) for(int i=0;i<3;++i)
    fixture.data.inv_table[fixture.tables.L.property_index(0,i,j)]=fixture.data.T_grid[j];

  // Pressure has two roots in rho for this temperature.
  for(int j=0;j<3;++j) for(int i=0;i<3;++i)
    fixture.data.lte_table[fixture.tables.L.property_index(0,i,j)]=(i==1?2000:1000);
  EXPECT_TRUE(rejects([&]{gas.ConservativeFromPhysical({PressureTemperature{1500,200},{0,0}},out);}));
  fixture.data.lte_table=1000;
  EXPECT_TRUE(rejects([&]{gas.ConservativeFromPhysical({PressureTemperature{1000,200},{0,0}},out);}));
  EXPECT_EQ(out.Size(),1); EXPECT_EQ(out[0],123);
  return 0;
}

TEST(PhysicalState_JSONValidationAndProfiles)
{
  using nlohmann::json;
  const StateLayout layout(2,1);
  const json good={{"pressure",60000},{"temperature",1200},{"velocity",{10,-2}}};
  const auto parsed=ParsePhysicalState(good,layout,"test.state");
  EXPECT_CLOSE(std::get<PressureTemperature>(parsed.thermo).pressure,60000,1e-12);
  for (auto bad : std::vector<json>{
      {{"pressure",1},{"velocity",{0,0}}},
      {{"pressure",1},{"temperature",300},{"density",1},{"velocity",{0,0}}},
      {{"pressure",true},{"temperature",300},{"velocity",{0,0}}},
      {{"pressure",1},{"temperature",300},{"velocity",{0,0}},{"species",{1}}},
      {{"pressure",1},{"temperature",300}}})
    EXPECT_TRUE(rejects([&]{ParsePhysicalState(bad,layout,"test.state");}));
  auto gas=std::make_shared<IdealGasModel>(PhysicsConstants(1.4,0.72,287,0.02),layout);
  GasModelInterfaceT<IdealGasModel> bridge(gas);
  EXPECT_TRUE(rejects([&]{MakeInitialCondition({{"state",good},{"function","RampIC"}},{},bridge);}));
  EXPECT_TRUE(rejects([&]{MakeInitialCondition({{"state",good},{"physical_profile",{}}},{},bridge);}));
  const json blob={{"type","thermal-blob"},{"radius",0.5},{"pressure",60000},
                   {"ambient_temperature",1200},{"peak_temperature",1500}};
  const auto f=ParsePhysicalProfile(blob,layout);
  mfem::Vector x(2); x=0.0;
  EXPECT_CLOSE(std::get<PressureTemperature>(f(x).thermo).temperature,1500,1e-12);
  x[0]=0.5;
  EXPECT_CLOSE(std::get<PressureTemperature>(f(x).thermo).temperature,1200+300/std::exp(1.),1e-12);
  const json vortex={{"type","vortex"},{"radius",0.5},{"speed",10},{"strength",0.2},
                     {"density",0.2},{"temperature",1200},{"shape_gamma",1.4},{"shape_gas_constant",287}};
  const auto v=ParsePhysicalProfile(vortex,layout)(x);
  EXPECT_TRUE(std::get<DensityTemperature>(v.thermo).temperature < 1200);
  EXPECT_TRUE(v.velocity[1]>0);
  return 0;
}

TEST(PhysicalState_LTERecoveryFromInexactInverse)
{
  TableFixture fixture;
  fixture.data.inv_table = 175.0; // Valid but deliberately inexact first guess.
  LTEGas gas(PhysicsConstants(1.4,0.72,10,0.02),StateLayout(2,1),fixture.tables);
  mfem::Vector out;
  gas.ConservativeFromPhysical({DensityTemperature{2,200},{3,-2}},out);
  EXPECT_CLOSE(gas.temperature(PointStateView(out.GetData())),200,1e-10);
  const auto set_cv = [&](double cv) {
    for (int j=0;j<3;++j) for (int i=0;i<3;++i)
      fixture.data.lte_table[fixture.tables.L.property_index(fixture.tables.L.cv_idx,i,j)] = cv;
  };
  mfem::Vector saved(out);
  // cv=1 oscillates between two in-range temperatures; it must terminate with
  // a host error. Smaller cv exits the table and must fail before another lookup.
  for (double cv : {1.,0.1,0.}) {
    set_cv(cv);
    EXPECT_TRUE(rejects([&]{gas.ConservativeFromPhysical({DensityTemperature{2,200},{3,-2}},out);}));
    for (int q=0;q<out.Size();++q) EXPECT_EQ(out[q],saved[q]);
  }
  return 0;
}

TEST(InitialCondition_UnifiedFactoryCompatibility)
{
  using nlohmann::json;
  auto gas=std::make_shared<IdealGasModel>(PhysicsConstants(1.4,0.72,287,0.02),StateLayout(2,1));
  GasModelInterfaceT<IdealGasModel> bridge(gas);
  auto mesh=mfem::Mesh::MakeCartesian2D(1,1,mfem::Element::QUADRILATERAL);
  auto &transformation=*mesh.GetElementTransformation(0);
  const auto &point=mfem::Geometries.GetCenter(mfem::Geometry::SQUARE);
  auto evaluate = [&](const json &config, const json &runtime) {
    auto coefficient=MakeInitialCondition(config,runtime,bridge);
    mfem::Vector result;
    coefficient->Eval(result,transformation,point);
    return result;
  };
  const json runtime={{"gamma",1.4},{"R_gas",287}};
  const auto legacy=evaluate({{"cpg_state",{{"pressure",60000},{"temperature",1200},{"ux",10},{"ur",-2}}}},runtime);
  const auto physical=evaluate({{"state",{{"pressure",60000},{"temperature",1200},{"velocity",{10,-2}}}}},runtime);
  for(int q=0;q<4;++q) EXPECT_CLOSE(legacy[q],physical[q],1e-9);

  // Exercise every legacy signature through the same public entry point.
  // Distinct weights detect reordered parameters and preserve raw conservative
  // output (including values that would be rejected as physical input).
  auto emit = [](double value) {
    return [value](const mfem::Vector &,mfem::Vector &out) { out=value; };
  };
  auto &factory=Prandtl::ConditionFactory::Instance();
  const std::string key="FactoryCompatibilityFixture";
  factory.RegisterInitialCondition0(key,[=]{return emit(-1);});
  factory.RegisterInitialCondition1(key,[=](double a){return emit(a);});
  factory.RegisterInitialCondition2(key,[=](double a,double b){return emit(a+10*b);});
  factory.RegisterInitialCondition3(key,[=](double a,double b,double c){return emit(a+10*b+100*c);});
  factory.RegisterInitialCondition4(key,[=](double a,double b,double c,double d){return emit(a+10*b+100*c+1000*d);});
  factory.RegisterInitialCondition5(key,[=](double a,double b,double c,double d,double e){return emit(a+10*b+100*c+1000*d+10000*e);});
  const double expected[]={-1,1,21,321,4321,54321};
  for(int signature=0;signature<=5;++signature) {
    const auto result=evaluate({{"function",key},{"signature",signature},
      {"params",{{"x1",1},{"x2",2},{"x3",3},{"x4",4},{"x5",5}}}},runtime);
    for(int q=0;q<4;++q) EXPECT_EQ(result[q],expected[signature]);
  }
  EXPECT_TRUE(rejects([&]{evaluate({{"function","MissingFixture"}},runtime);}));
  EXPECT_TRUE(rejects([&]{evaluate({{"function",key},{"signature",6}},runtime);}));
  EXPECT_TRUE(rejects([&]{evaluate({{"cpg_state",{{"pressure",60000},{"temperature",1200}}}},{{"gas_model","lte"}});}));
  return 0;
}

TEST(BoundaryState_PhysicalPackingAndLegacyCompatibility)
{
  using nlohmann::json;
  TableFixture fixture;
  auto lte = std::make_shared<LTEGas>(PhysicsConstants(1.4, 0.72, 10, 0.02), StateLayout(2, 1),
                                      fixture.tables);
  auto cpg = std::make_shared<IdealGasModel>(lte->phys, lte->L);
  GasModelInterfaceT<LTEGas> lte_interface(lte);
  GasModelInterfaceT<IdealGasModel> cpg_interface(cpg);
  const auto file =
      (std::filesystem::path(__FILE__).parent_path() / "data/physical-boundary-profile.dat")
          .string();
  const json config = {{"type", "radial-profile"}, {"file", file}, {"pressure", 4000}};
  const auto radial = ParseBoundaryState(config, lte_interface);
  const auto constant = ParseBoundaryState(
      {{"type", "exterior-state"},
       {"state", {{"pressure", 4000}, {"temperature", 200}, {"velocity", {1, 0}}}}},
      lte_interface);
  mfem::Vector input(radial.values);
  const int constant_offset = AppendBCVectorPayload(input, constant.values);
  mfem::Array<BCDescriptor> descriptors(2);
  descriptors[0] = {int(BCType::PrescribedState), int(radial.kind), 0, 0, -1, 0};
  descriptors[1] = {int(BCType::PrescribedState), int(constant.kind), constant_offset, 0, -1, 0};
  mfem::Array<int> markers(3);
  markers[0] = 0;
  markers[1] = 1;
  markers[2] = -1;
  mfem::Vector xyz(18);
  xyz = 0.0;
  xyz[1] = 0;
  xyz[3] = .5;
  xyz[5] = .8;
  mfem::Array<BCDescriptor> points;
  mfem::Vector packed;
  PackBoundaryPointStates(descriptors, input, {"Inflow", "Top"}, markers, 3, xyz, lte_interface,
                          points, packed);
  for (int point = 0; point < 3; ++point)
    {
      EXPECT_EQ(points[point].data_kind, int(BCDataKind::VectorConstant));
      PointStateView state(packed.HostRead() + points[point].data_index);
      const double temperatures[] = {150, 200, 250};
      const double axial[] = {-4, 1, 6};
      const double radial_velocity[] = {0, 0, -2};
      EXPECT_CLOSE(lte->temperature(state), temperatures[point], 1e-10);
      EXPECT_CLOSE(lte->pressure(state), 4000, 1e-9);
      EXPECT_CLOSE(state.velocity(lte->L, 0), axial[point], 1e-12);
      EXPECT_CLOSE(state.velocity(lte->L, 1), radial_velocity[point], 1e-12);
    }
  // Midpoint conversion uses interpolated primitives, not interpolated energy/rho.
  EXPECT_CLOSE(packed[points[1].data_index], 2, 1e-12);
  for (int point = 3; point < 6; ++point)
    {
      EXPECT_EQ(points[point].data_index, constant_offset);
    }
  for (int point = 6; point < 9; ++point)
    {
      EXPECT_EQ(points[point].type, int(BCType::Invalid));
    }

  mfem::Array<BCDescriptor> cpg_points, legacy_points;
  mfem::Vector cpg_packed, legacy_packed;
  PackBoundaryPointStates(descriptors, input, {"Inflow", "Top"}, markers, 3, xyz, cpg_interface,
                          cpg_points, cpg_packed);
  mfem::Vector legacy(12);
  legacy[0] = 4000;
  legacy[1] = 1.4;
  legacy[2] = 10;
  legacy[3] = 2;
  for (int q = 0; q < 8; ++q)
    {
      legacy[4 + q] = radial.values[2 + q];
    }
  descriptors[0].data_kind = int(BCDataKind::RadialCPG);
  markers.SetSize(1);
  markers[0] = 0;
  PackBoundaryPointStates(descriptors, legacy, {"Inflow"}, markers, 3, xyz, cpg_interface,
                          legacy_points, legacy_packed);
  for (int point = 0; point < 3; ++point)
    {
      for (int q = 0; q < 4; ++q)
        {
          EXPECT_CLOSE(cpg_packed[cpg_points[point].data_index + q],
                       legacy_packed[legacy_points[point].data_index + q], 1e-8);
        }
    }

  descriptors[0].data_kind = int(BCDataKind::RadialPhysical);
  xyz[3] = .9;
  try
    {
      PackBoundaryPointStates(descriptors, input, {"Inflow"}, markers, 3, xyz, lte_interface,
                              points, packed);
      EXPECT_TRUE(false);
    }
  catch (const std::invalid_argument &error)
    {
      const std::string message = error.what();
      EXPECT_TRUE(message.find("Inflow") != std::string::npos);
      EXPECT_TRUE(message.find("radius 0.9") != std::string::npos);
    }
  for (const json &bad : std::vector<json>{
           {{"type", "exterior-state"}, {"state", {{"pressure", 4000}, {"velocity", {0, 0}}}}},
           {{"type", "exterior-state"},
            {"state", {{"pressure", 4000}, {"temperature", 400}, {"velocity", {0, 0}}}}},
           {{"type", "radial-profile"}, {"file", file}, {"pressure", -1}},
           {{"type", "radial-profile"}, {"file", file}, {"pressure", 4000}, {"species", {1}}},
           {{"type", "radial-profile"}, {"file", file}, {"pressure", 4000}, {"flag", 0.5}}})
    {
      EXPECT_TRUE(rejects([&] { ParseBoundaryState(bad, lte_interface); }));
    }
  return 0;
}

namespace
{
  template<typename Gas>
  struct WallBoundaryCache
  {
    Gas gas;
    int dim = 2;
    int num_equations = 4;
    const mfem::real_t *bc_vector_d;
  };
}

TEST(IsothermalWall_UsesSelectedEntropyScaling)
{
  TableFixture fixture;
  const PhysicsConstants physics(1.4, 0.72, 10, 0.02);
  const IdealGasModel cpg(physics, StateLayout(2, 1));
  const LTEGas lte(physics, StateLayout(2, 1), fixture.tables);
  const mfem::real_t wall_data[] = {3, -4, 250};
  const BCDescriptor boundary{int(BCType::NoSlipIso), int(BCDataKind::VectorConstant),
                              0, 0, -1, 0};

  const auto check = [&](const auto &gas, mfem::real_t normalization)
    {
      WallBoundaryCache<std::decay_t<decltype(gas)>> cache{gas, 2, 4, wall_data};
      const auto interior_beta = 1 / (normalization * 200);
      const auto wall_beta = 1 / (normalization * 250);
      mfem::real_t entropy[] = {0, interior_beta, -2 * interior_beta, -interior_beta};
      mfem::real_t correction[4];
      BC::ComputeBdrFaceGradFlux(cache, boundary, entropy, correction);
      EXPECT_CLOSE(correction[0], 0, 1e-15);
      EXPECT_CLOSE(correction[1], 3 * wall_beta - interior_beta, 1e-15);
      EXPECT_CLOSE(correction[2], -4 * wall_beta + 2 * interior_beta, 1e-15);
      EXPECT_CLOSE(correction[3], -wall_beta + interior_beta, 1e-15);

      entropy[1] = 3 * wall_beta;
      entropy[2] = -4 * wall_beta;
      entropy[3] = -wall_beta;
      BC::ComputeBdrFaceGradFlux(cache, boundary, entropy, correction);
      for (int equation = 0; equation < 4; ++equation)
        {
          EXPECT_CLOSE(correction[equation], 0, 1e-15);
        }
      return 0;
    };

  EXPECT_EQ(check(cpg, physics.R_gas), 0);
  EXPECT_EQ(check(lte, 1), 0);
  return 0;
}
