// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"

using namespace Theseus;

namespace
{

  struct AnalyticTables
  {
    LTETable::Data data;
    LTETable::LTETables tables{3, 3};

    AnalyticTables()
    {
      data.rho_grid.SetSize(3);
      data.T_grid.SetSize(3);
      data.e_grid.SetSize(3);
      data.lte_table.SetSize(81);
      data.lte_table = 0.0;
      data.inv_table.SetSize(9);

      for (int index = 0; index < 3; ++index)
        {
          data.rho_grid[index] = 1 + index;
          data.T_grid[index] = 100 + 100 * index;
          data.e_grid[index] = 700 + 200 * index;
        }

      for (int temperature_index = 0; temperature_index < 3; ++temperature_index)
        {
          for (int density_index = 0; density_index < 3; ++density_index)
            {
              const auto density = data.rho_grid[density_index];
              const auto temperature = data.T_grid[temperature_index];
              const auto set_property = [&](int property, mfem::real_t value)
                {
                  const int index = tables.L.property_index(property, density_index,
                                                           temperature_index);
                  data.lte_table[index] = value;
                };

              set_property(tables.L.P_idx, 10 * density * temperature);
              set_property(tables.L.e_idx, 500 + 2 * temperature);
              set_property(tables.L.cv_idx, 2);
              set_property(tables.L.c_idx, 20 + density + temperature);
              set_property(tables.L.mu_idx, 0.01 * density + 0.001 * temperature);
              set_property(tables.L.lambda_idx, 2 * density + 0.1 * temperature);
              data.inv_table[temperature_index * 3 + density_index] = temperature;
            }
        }

      tables.tables = {data.lte_table.GetData(), data.inv_table.GetData(),
                       data.rho_grid.GetData(), data.T_grid.GetData(), data.e_grid.GetData()};
    }
  };

  struct CountingEOS : LTEGasEOS
  {
    mutable int temperature_recoveries = 0;

    template<typename StateView>
    mfem::real_t temperature(const PhysicsConstants &physics, const StateLayout &layout,
                             const StateView &state, const LTETable::LTETables &tables) const
    {
      ++temperature_recoveries;
      return LTEGasEOS::temperature(physics, layout, state, tables);
    }
  };

  GasPropertyRequest AllProperties()
  {
    GasPropertyRequest request;
    request.pressure = true;
    request.temperature = true;
    request.sound_speed = true;
    request.viscosity = true;
    request.thermal_conductivity = true;
    return request;
  }

}

TEST(gas_properties_cpg_matches_individual_queries)
{
  auto gas = std::make_shared<IdealGasModel>(PhysicsConstants(1.4, 0.72, 287, 0.02),
                                            StateLayout(2, 1));
  GasModelInterfaceT<IdealGasModel> interface(gas);
  PhysicalStateInput input;
  input.thermo = PressureTemperature{60000, 1200};
  input.velocity = {20, -3};

  mfem::Vector conservative;
  interface.ConservativeFromPhysical(input, conservative);
  const DofStateView state{conservative.HostRead(), 0};
  const auto properties = interface.EvaluateProperties(state, AllProperties());

  EXPECT_CLOSE(properties.pressure, gas->pressure(state), 1e-10);
  EXPECT_CLOSE(properties.temperature, 1200, 1e-10);
  EXPECT_CLOSE(properties.sound_speed, gas->sound_speed(state), 1e-10);
  EXPECT_CLOSE(properties.viscosity, gas->viscosity(state), 1e-12);
#ifdef SUTHERLAND
  const auto relative_temperature = 1200 / gas->phys.T0;
  const auto expected_viscosity = gas->phys.mu0 * (gas->phys.T0 + gas->phys.Ts) *
                                 std::pow(relative_temperature, 1.5) / (1200 + gas->phys.Ts);
#else
  const auto expected_viscosity = gas->phys.mu;
#endif
  EXPECT_CLOSE(properties.viscosity, expected_viscosity, 1e-12);
  EXPECT_CLOSE(properties.thermal_conductivity,
               expected_viscosity * gas->phys.cp / gas->phys.Pr, 1e-10);
  EXPECT_CLOSE(properties.thermal_conductivity, gas->thermal_conductivity(state), 1e-10);

  const auto empty = interface.EvaluateProperties(state, {});
  EXPECT_EQ(empty.pressure, 0);
  EXPECT_EQ(empty.temperature, 0);
  EXPECT_EQ(empty.sound_speed, 0);
  EXPECT_EQ(empty.viscosity, 0);
  EXPECT_EQ(empty.thermal_conductivity, 0);
  return 0;
}

TEST(gas_properties_lte_recovers_temperature_once_for_all_properties)
{
  AnalyticTables fixture;
  using CountingGas = LTEGasModel<CountingEOS, LTETransport>;
  auto gas = std::make_shared<CountingGas>(PhysicsConstants(1.4, 0.72, 10, 0.02),
                                           StateLayout(2, 1), fixture.tables);
  GasModelInterfaceT<CountingGas> interface(gas);

  for (const auto density : {1.0, 1.5, 3.0})
    {
      for (const auto temperature : {100.0, 175.0, 300.0})
        {
          PhysicalStateInput input;
          input.thermo = DensityTemperature{density, temperature};
          input.velocity = {20, -3};
          mfem::Vector conservative;
          interface.ConservativeFromPhysical(input, conservative);
          const DofStateView state{conservative.HostRead(), 0};

          gas->eos.temperature_recoveries = 0;
          const auto properties = interface.EvaluateProperties(state, AllProperties());
          EXPECT_EQ(gas->eos.temperature_recoveries, 1);
          EXPECT_CLOSE(properties.temperature, temperature, 1e-10);
          EXPECT_CLOSE(properties.pressure, 10 * density * temperature, 1e-10);
          EXPECT_CLOSE(properties.sound_speed, 20 + density + temperature, 1e-10);
          EXPECT_CLOSE(properties.thermal_conductivity, 2 * density + 0.1 * temperature, 1e-10);
#ifndef SUTHERLAND
          EXPECT_CLOSE(properties.viscosity, 0.01 * density + 0.001 * temperature, 1e-12);
#endif
          EXPECT_CLOSE(properties.pressure, gas->pressure(state), 1e-10);
          EXPECT_CLOSE(properties.sound_speed, gas->sound_speed(state), 1e-10);
          EXPECT_CLOSE(properties.viscosity, gas->viscosity(state), 1e-12);
          EXPECT_CLOSE(properties.thermal_conductivity, gas->thermal_conductivity(state), 1e-10);

          gas->eos.temperature_recoveries = 0;
          const auto empty = interface.EvaluateProperties(state, {});
          EXPECT_EQ(gas->eos.temperature_recoveries, 0);
          EXPECT_EQ(empty.temperature, 0);
          EXPECT_EQ(empty.pressure, 0);
        }
    }
  return 0;
}

TEST(gas_properties_lte_only_populates_requested_values)
{
  AnalyticTables fixture;
  LTEGas gas(PhysicsConstants(1.4, 0.72, 10, 0.02), StateLayout(2, 1), fixture.tables);
  PhysicalStateInput input;
  input.thermo = DensityTemperature{1.5, 175};
  input.velocity = {0, 0};
  mfem::Vector conservative;
  gas.ConservativeFromPhysical(input, conservative);
  const PointStateView state(conservative.HostRead());

  GasPropertyRequest temperature_only;
  temperature_only.temperature = true;
  const auto temperature = gas.EvaluateProperties(state, temperature_only);
  EXPECT_CLOSE(temperature.temperature, 175, 1e-10);
  EXPECT_EQ(temperature.pressure, 0);
  EXPECT_EQ(temperature.sound_speed, 0);
  EXPECT_EQ(temperature.viscosity, 0);
  EXPECT_EQ(temperature.thermal_conductivity, 0);

  GasPropertyRequest pressure_only;
  pressure_only.pressure = true;
  const auto pressure = gas.EvaluateProperties(state, pressure_only);
  EXPECT_CLOSE(pressure.pressure, 2625, 1e-10);
  EXPECT_EQ(pressure.temperature, 0);
  EXPECT_EQ(pressure.sound_speed, 0);
  EXPECT_EQ(pressure.viscosity, 0);
  EXPECT_EQ(pressure.thermal_conductivity, 0);
  return 0;
}

TEST(gas_properties_reject_unavailable_lte_table_property)
{
  AnalyticTables fixture;
  fixture.tables.L.num_properties = 7;
  LTEGas gas(PhysicsConstants(1.4, 0.72, 10, 0.02), StateLayout(2, 1), fixture.tables);
  PhysicalStateInput input;
  input.thermo = DensityTemperature{1.5, 175};
  input.velocity = {0, 0};
  mfem::Vector conservative;
  gas.ConservativeFromPhysical(input, conservative);

  GasPropertyRequest request;
  request.thermal_conductivity = true;
  bool rejected = false;
  try
    {
      gas.EvaluateProperties(PointStateView(conservative.HostRead()), request);
    }
  catch (const std::invalid_argument &)
    {
      rejected = true;
    }
  EXPECT_TRUE(rejected);
  return 0;
}
