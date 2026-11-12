// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "LTEGasModel.hpp"
#include "helmholtz_fixture.hpp"
#include "LTEHelmholtzTableBuilder.hpp"
#include "ViscousInvariants.hpp"
#ifdef USE_PLATO
#include "plato_helpers.hpp"
#endif
#include <iostream>
#include <iomanip>
#include <string>

namespace
{
  using namespace Theseus;
  namespace HT = Theseus::HelmholtzTable;
  int checks = 0;
  int failures = 0;

  void Check(bool pass, const char *name)
  {
    ++checks;
    if (!pass)
      {
        ++failures;
        std::cerr << "FAIL: " << name << '\n';
      }
  }

  void Near(double value, double expected, double tolerance, const char *name)
  {
    if (!std::isfinite(value) || std::abs(value - expected) > tolerance*std::max(1.0, std::abs(expected)))
      {
        std::cerr << std::setprecision(17) << name << ": " << value << " expected " << expected << '\n';
      }
    Check(std::isfinite(value) && std::abs(value - expected) <= tolerance*std::max(1.0, std::abs(expected)), name);
  }

  void ConversionAndQueries(LTEHelmholtzGas &gas, double r, double t)
  {
    const auto expected = HelmholtzEOS::At(gas.T, r, t);
    for (const ThermodynamicInput input : {ThermodynamicInput{DensityTemperature{r, t}},
                                          ThermodynamicInput{DensityPressure{r, expected.pressure()}},
                                          ThermodynamicInput{PressureTemperature{expected.pressure(), t}}})
      {
        PhysicalStateInput physical{input, {3, -2, 1}};
        mfem::Vector conservative;
        gas.ConservativeFromPhysical(physical, conservative);
        PointStateView state(conservative.HostRead());
        Near(gas.density(state), r, 1e-10, "physical density");
        Near(gas.temperature(state), t, 1e-10, "physical temperature round trip");
        Near(gas.specific_internal_energy(state), expected.energy(), 1e-10, "physical energy");
        Near(gas.pressure(state), expected.pressure(), 1e-10, "physical pressure");
        // The represented potential is C1, not C2. Query recovered coordinates
        // for derivative consistency; independent PLATO checks remain separate.
        const auto query = HelmholtzEOS::At(gas.T, gas.density(state), gas.temperature(state));
        Near(gas.cp(state), query.cp(), 1e-10, "gas cp");
        Near(gas.gamma(state), query.gamma(), 1e-10, "gas gamma");
        Near(gas.R_gas(state), expected.gas_constant(), 1e-10, "gas constant");
        Near(gas.sound_speed(state), std::sqrt(query.sound_speed_squared()), 1e-10, "gas acoustics");
        const auto properties = gas.EvaluateProperties(state, {true, true, true, true, true});
        Near(properties.pressure, gas.pressure(state), 1e-12, "batched pressure");
        Near(properties.temperature, gas.temperature(state), 1e-12, "batched temperature");
        Near(properties.sound_speed, gas.sound_speed(state), 1e-12, "batched acoustics");
        Near(properties.viscosity, gas.viscosity(state), 1e-12, "batched viscosity");
        Near(properties.thermal_conductivity, gas.thermal_conductivity(state), 1e-12, "batched conductivity");
        mfem::real_t entropy_values[MAXEQ]{};
        PointStateViewRW entropy(entropy_values);
        gas.entropy_state(state, entropy);
        Near(entropy.mass(gas.L), (expected.gibbs() - 7)/t, 1e-10, "entropy density slot");
        Near(entropy.energy(gas.L), -1/t, 1e-10, "entropy energy slot");
        ViscousInvariants::Tables(gas, "helmholtz-gas-test");
      }
  }

  void RootCases()
  {
    const mfem::real_t grid[]{0, 1};
    const auto root = HT::detail::SolvePressure(grid, 2, 3, 1,
                       [](auto x) { return 1 + (x - 0.5)*(x - 0.5); });
    Near(root, 0.5, 1e-10, "interior tangent pressure root");
    for (int scenario = 0; scenario < 3; ++scenario)
      {
        bool rejected = false;
        try
          {
            HT::detail::SolvePressure(grid, 2, 4, 1, [=](auto x)
              {
                if (scenario == 0)
                  {
                    return 1 + (x - 0.25)*(x - 0.75);
                  }
                return scenario == 1 ? 1.0 : 2.0;
              });
          }
        catch (const std::invalid_argument &)
          {
            rejected = true;
          }
        Check(rejected, "reject multiple/constant/absent pressure roots");
      }
  }

  void Kernel(HT::Data &data, const HT::Tables &tables)
  {
    PhysicsConstants physics{};
    physics.mu0 = 1;
    physics.T0 = 1;
    physics.Ts = 1;
    const StateLayout layout(3, 1);
    struct Owner { HT::Data *lteTableData; } owner{&data};
    LTEHelmholtzGas host(physics, layout, tables);
    const auto gas = host.to_device(owner);
    mfem::Vector output(5*8);
    auto *values = output.Write();
    mfem::forall(5, [=] MFEM_HOST_DEVICE (int point)
      {
        const mfem::real_t r = 0.75 + point*0.125;
        const mfem::real_t t = 3 + point*0.25;
        mfem::real_t u[MAXEQ]{};
        PointStateViewRW state(u);
        state.set_mass(layout, r);
        state.set_momentum(layout, 0, 3*r);
        state.set_momentum(layout, 1, -2*r);
        state.set_momentum(layout, 2, r);
        state.set_energy(layout, r*(2*r + r*r/2 + t*t + 7));
        values[point*8] = gas.temperature(state);
        values[point*8 + 1] = gas.pressure(state);
        values[point*8 + 2] = gas.viscosity(state);
        values[point*8 + 3] = gas.thermal_conductivity(state);
        mfem::real_t ev[MAXEQ]{};
        PointStateViewRW entropy(ev);
        gas.entropy_state(state, entropy);
        values[point*8 + 4] = entropy.mass(layout);
        mfem::real_t gv[MAXEQ]{0, 1, 2, 3, 0.5};
        PointStateView gradient(gv);
        mfem::real_t pv[MAXEQ]{};
        PointStateViewRW primitive(pv);
        PointStateView read_state(u);
        gas.grad_entropy_to_grad_prim(read_state, gradient, primitive);
        values[point*8 + 5] = primitive.mass(layout);
        values[point*8 + 6] = primitive.momentum(layout, 0);
        values[point*8 + 7] = primitive.energy(layout);
      });
    const auto *result = output.HostRead();
    for (int point = 0; point < 5; ++point)
      {
        const auto r = 0.75 + point*0.125;
        const auto t = 3 + point*0.25;
        Near(result[point*8], t, 1e-10, "device gas recovery");
        Near(result[point*8 + 1], r*r*(2 + r + t/4), 1e-10, "device gas pressure");
#ifdef SUTHERLAND
        Near(result[point*8 + 2], 2*t*std::sqrt(t)/(t + 1), 1e-10, "Sutherland device viscosity");
#else
        Near(result[point*8 + 2], 2 + r + t, 1e-10, "tabulated device viscosity");
#endif
        Near(result[point*8 + 3], 4 + 2*r + t, 1e-10, "tabulated device conductivity");
        Near(result[point*8 + 4], (4*r + 1.5*r*r + r*t/2 - t*t - 7)/t, 1e-10, "device entropy");
        Near(result[point*8 + 5], 0, 1e-12, "primitive density convention");
        Near(result[point*8 + 6], 2.5*t, 1e-10, "primitive velocity gradient");
        Near(result[point*8 + 7], 0.5*t*t, 1e-10, "primitive temperature gradient");
      }
  }

#ifdef USE_PLATO
  void RealAir()
  {
    const std::string database(Theseus::BuildConfig::PlatoDBPath);
    plato_initialize("LTE_table_rhoT_(air5)", "air5", "empty", "empty", database.c_str());
    HT::Data data;
    LTETable::uniform_grid(17, 0.8, 1.2, data.rho_grid);
    LTETable::uniform_grid(41, 250, 350, data.T_grid);
    const auto tables = HT::Build(data);
    Check(data.lte_table.Size() == HT::Layout::num_properties*17*41, "authoritative six-plane allocation");
    for (int i = 0; i < data.inv_table.Size(); ++i)
      {
        Check(std::isfinite(data.inv_table[i]) && data.inv_table[i] >= 250 && data.inv_table[i] <= 350,
              "bounded inverse guess");
      }
    double maximum_relative_error[5]{};
    PhysicsConstants physics{};
    const StateLayout layout(3, 1);
    LTEHelmholtzGas gas(physics, layout, tables);
    for (const auto r : {0.8, 0.9, 1.0125, 1.1, 1.2})
      {
        for (const auto t : {250.0, 280.0, 293.15, 320.0, 350.0})
          {
            PlatoMixture reference;
            plato_set_state(r, t, reference);
            const auto sample = HelmholtzEOS::At(tables, r, t);
            const double actual[]{sample.pressure(), sample.energy(), sample.cv(), sample.cp(),
                                  std::sqrt(sample.sound_speed_squared())};
            const double independent[]{reference.P, reference.e, reference.cv, reference.cp, reference.c};
            for (int property = 0; property < 5; ++property)
              {
                maximum_relative_error[property] = std::max(maximum_relative_error[property],
                  std::abs(actual[property] - independent[property])/std::abs(independent[property]));
              }
            Near(sample.pressure(), reference.P, 2e-5, "AIR5 independent pressure");
            Near(sample.energy(), reference.e, 2e-5, "AIR5 independent energy");
            Near(sample.cv(), reference.cv, 2e-3, "AIR5 independent cv");
            Near(sample.cp(), reference.cp, 2e-3, "AIR5 independent cp");
            Near(std::sqrt(sample.sound_speed_squared()), reference.c, 2e-3, "AIR5 independent acoustics");
            ConversionAndQueries(gas, r, t);
          }
      }
    std::cout << "AIR5 maximum relative errors (p,e,cv,cp,c):";
    for (const auto error : maximum_relative_error)
      {
        std::cout << ' ' << error;
      }
    std::cout << '\n';
    plato_finalize();
  }
#endif
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string mode = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(mode == "plato" ? "cpu" : mode.c_str());
  if (mode == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA))
    {
      return 1;
    }
#ifdef USE_PLATO
  if (mode == "plato")
    {
      RealAir();
    }
  else
#endif
    {
      HT::Data data;
      const auto tables = TheseusTest::AnalyticHelmholtz(data);
      PhysicsConstants physics{};
      physics.mu0 = 1;
      physics.T0 = 1;
      physics.Ts = 1;
      LTEHelmholtzGas gas(physics, StateLayout(3, 1), tables);
      ConversionAndQueries(gas, 1, 3);
      ConversionAndQueries(gas, 1.375, 4.125);
      RootCases();
      Kernel(data, tables);
    }
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
