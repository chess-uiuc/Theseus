// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "LTEGasModel.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace
{
using Real = mfem::real_t;

// Guard the selected logical edge, not merely the end of a large allocator block.
// Each CTest case owns its process: an unsafe read must remain a failing test.
class GuardedGrid
{
  void *mapping = MAP_FAILED;
  std::size_t page_bytes = 0;
  Real *values = nullptr;
public:
  explicit GuardedGrid(bool guard_upper)
  {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
      {
        throw std::runtime_error("Cannot determine page size");
      }
    page_bytes = static_cast<std::size_t>(page_size);
    mapping = mmap(nullptr, 3 * page_bytes, PROT_NONE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping == MAP_FAILED)
      {
        throw std::runtime_error("Cannot allocate guarded grid");
      }
    char *middle = static_cast<char *>(mapping) + page_bytes;
    if (mprotect(middle, page_bytes, PROT_READ | PROT_WRITE) != 0)
      {
        munmap(mapping, 3 * page_bytes);
        throw std::runtime_error("Cannot enable grid storage");
      }
    values = reinterpret_cast<Real *>(guard_upper ? middle + page_bytes - 3 * sizeof(Real)
                                                 : middle);
  }
  ~GuardedGrid()
  {
    if (mapping != MAP_FAILED)
      {
        munmap(mapping, 3 * page_bytes);
      }
  }
  GuardedGrid(const GuardedGrid &) = delete;
  GuardedGrid &operator=(const GuardedGrid &) = delete;
  Real *Data() { return values; }
};

struct Fixture
{
  GuardedGrid rho;
  GuardedGrid temperature;
  Theseus::LTETable::Data owner;
  Theseus::LTETable::LTETables tables{3, 3};
  explicit Fixture(bool upper) : rho(upper), temperature(upper)
  {
    owner.rho_grid.SetSize(3);
    owner.T_grid.SetSize(3);
    owner.e_grid.SetSize(3);
    owner.lte_table.SetSize(9 * 3 * 3);
    owner.inv_table.SetSize(3 * 3);
    for (int i = 0; i < 3; ++i)
      {
        rho.Data()[i] = 1 + i;
        temperature.Data()[i] = 100 + 100 * i;
        owner.rho_grid[i] = rho.Data()[i];
        owner.T_grid[i] = temperature.Data()[i];
        owner.e_grid[i] = 700 + 200 * i;
      }
    for (int j = 0; j < 3; ++j)
      {
        for (int i = 0; i < 3; ++i)
          {
            const Real t = temperature.Data()[j];
            // Distinct property planes make cross-property addressing visible.
            const Real properties[9] = {
              10 * rho.Data()[i] * t, 500 + 2 * t, 2, 3, 10, 1.4,
              1000 + t, 0.01 + t * 1e-5, 0.1 + t * 1e-4};
            for (int property = 0; property < 9; ++property)
              {
                owner.lte_table[tables.L.property_index(property, i, j)] = properties[property];
              }
            owner.inv_table[j * 3 + i] = t;
          }
      }
    tables.tables = {owner.lte_table.HostRead(), owner.inv_table.HostRead(),
                     rho.Data(), temperature.Data(), owner.e_grid.HostRead()};
  }

  void SetHeatCapacity(Real value)
  {
    for (int j = 0; j < 3; ++j)
      {
        for (int i = 0; i < 3; ++i)
          {
            owner.lte_table[tables.L.property_index(tables.L.cv_idx, i, j)] = value;
          }
      }
  }

  void SetDensityDependentEnergy()
  {
    // e=500+2T+100rho. The GLOBAL inverse rectangle includes (rho=3,e=800),
    // whose exact inverse is T=0, outside the forward temperature grid.
    for (int j = 0; j < 3; ++j)
      {
        owner.e_grid[j] = 800 + 300 * j;
        for (int i = 0; i < 3; ++i)
          {
            owner.lte_table[tables.L.property_index(tables.L.e_idx, i, j)] =
              500 + 2 * temperature.Data()[j] + 100 * rho.Data()[i];
            owner.inv_table[j * 3 + i] =
              (owner.e_grid[j] - 500 - 100 * rho.Data()[i]) / 2;
          }
      }
  }

  Theseus::LTETable::LTETables DeviceView()
  {
    auto result = tables;
    result.tables = {owner.lte_table.Read(), owner.inv_table.Read(),
                     owner.rho_grid.Read(), owner.T_grid.Read(), owner.e_grid.Read()};
    return result;
  }
};

Real Evaluate(const Theseus::LTEGas &gas, const Real *values, Real t, bool recovery,
              int property = 7)
{
  Theseus::PointStateView state{values};
  return recovery ? gas.temperature(state) :
    gas.eos.biinterp_lte_table(property, gas.phys, gas.L, state, t, gas.T);
}
}

int main(int argc, char **argv)
{
  if (argc != 3)
    {
      std::cerr << "Usage: lte_runtime_memory_tests CASE cpu|cuda\n";
      return 2;
    }
  const std::string scenario = argv[1];
  const std::string backend = argv[2];
  if (backend != "cpu" && backend != "cuda")
    {
      std::cerr << "Backend must be cpu or cuda\n";
      return 2;
    }
  const bool upper = scenario.find("high") != std::string::npos;
  mfem::Device device(backend.c_str());
  Fixture fixture(upper);
  Real density = 2;
  Real specific_energy = 900;
  Real t = 200;
  bool recovery = false;
  int property = fixture.tables.L.mu_idx;
  const bool control = scenario == "valid";
  if (scenario == "forward_temperature_low")
    {
      t = 99;
    }
  else if (scenario == "forward_temperature_high")
    {
      t = 301;
    }
  else if (scenario == "forward_density_low")
    {
      density = 0.5;
    }
  else if (scenario == "forward_density_high")
    {
      density = 3.5;
    }
  else if (scenario == "forward_temperature_nan")
    {
      t = std::numeric_limits<Real>::quiet_NaN();
    }
  else if (scenario == "forward_density_nan")
    {
      density = std::numeric_limits<Real>::quiet_NaN();
    }
  else if (scenario == "forward_property_low" || scenario == "forward_property_high")
    {
      property = scenario == "forward_property_low" ? -1 : fixture.tables.L.num_properties;
    }
  else if (scenario == "newton_negative_cv" || scenario == "newton_infinite_cv")
    {
      recovery = true;
      fixture.owner.inv_table = 150;
      fixture.SetHeatCapacity(scenario == "newton_negative_cv" ? Real(-2) :
                              std::numeric_limits<Real>::infinity());
    }
  else if (scenario == "inverse_guess_low")
    {
      recovery = true;
      fixture.owner.inv_table = 50;
    }
  else if (scenario == "inverse_guess_high")
    {
      recovery = true;
      fixture.owner.inv_table = 350;
    }
  else if (scenario == "inverse_guess_nan")
    {
      recovery = true;
      fixture.owner.inv_table = std::numeric_limits<Real>::quiet_NaN();
    }
  else if (scenario == "newton_overshoot_high")
    {
      recovery = true;
      fixture.owner.inv_table = 150;
      fixture.SetHeatCapacity(0.5);
    }
  else if (scenario == "newton_negative_temperature")
    {
      recovery = true;
      specific_energy = 700;
      fixture.owner.inv_table = 150;
      fixture.SetHeatCapacity(0.5);
    }
  else if (scenario == "newton_zero_cv" || scenario == "newton_nan_cv")
    {
      recovery = true;
      fixture.owner.inv_table = 150;
      fixture.SetHeatCapacity(scenario == "newton_zero_cv" ? 0 :
                              std::numeric_limits<Real>::quiet_NaN());
    }
  else if (scenario == "inverse_density_domain")
    {
      recovery = true;
      density = 3;
      specific_energy = 800;
      fixture.SetDensityDependentEnergy();
    }
  else if (!control)
    {
      std::cerr << "Unknown case: " << scenario << '\n';
      return 2;
    }

  const Theseus::PhysicsConstants physics(1.4, 0.72, 10, 0.02);
  const Theseus::StateLayout layout(2, 1);
  const Real values[4] = {density, 0, 0, density * specific_energy};
  std::cout << "CASE=" << scenario << " backend=" << backend
            << " rho=" << density << " e=" << specific_energy
            << " query_T=" << t << " forward_rho=[1,3] forward_T=[100,300]"
            << std::endl;
  Real result = 0;
  try
    {
      if (backend == "cpu")
        {
          const Theseus::LTEGas gas(physics, layout, fixture.tables);
          result = Evaluate(gas, values, t, recovery, property);
          if (control)
            {
              for (Real endpoint : {Real(100), Real(300)})
                {
                  const Real actual = Evaluate(gas, values, endpoint, false);
                  if (std::abs(actual - (0.01 + endpoint * 1e-5)) > 1e-12)
                    {
                      std::cerr << "Valid endpoint lookup failed\n";
                      return 1;
                    }
                }
              for (Real endpoint : {Real(100), Real(300)})
                {
                  const Real direction = endpoint == 100 ? Real(0) : Real(400);
                  const Real adjacent = std::nextafter(endpoint, direction);
                  const Real actual = Evaluate(gas, values, adjacent, false);
                  if (std::abs(actual - (0.01 + endpoint * 1e-5)) > 1e-12)
                    {
                      std::cerr << "Roundoff-adjacent endpoint lookup failed\n";
                      return 1;
                    }
                }
              const Real recovered = Evaluate(gas, values, t, true);
              if (std::abs(recovered - 200) > 1e-10)
                {
                  std::cerr << "Valid temperature recovery failed\n";
                  return 1;
                }
            }
        }
      else
        {
          const Theseus::LTEGas gas(physics, layout, fixture.DeviceView());
          mfem::Vector output(1);
          output.UseDevice(true);
          Real *out = output.Write();
          mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
          {
            const Real state_values[4] = {density, 0, 0, density * specific_energy};
            Theseus::PointStateView state{state_values};
            out[0] = recovery ? gas.temperature(state) :
              gas.eos.biinterp_lte_table(property, gas.phys, gas.L, state, t, gas.T);
          });
          result = output.HostRead()[0];
          if (control)
            {
              for (int query = 0; query < 5; ++query)
                {
                  const Real endpoint = query % 2 == 0 ? Real(100) : Real(300);
                  const Real query_temperature = query < 2 ? endpoint :
                    std::nextafter(endpoint, query % 2 == 0 ? Real(0) : Real(400));
                  Real *control_output = output.Write();
                  mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
                  {
                    const Real state_values[4] = {density, 0, 0, density * specific_energy};
                    Theseus::PointStateView state{state_values};
                    control_output[0] = query == 4 ? gas.temperature(state) :
                      gas.eos.biinterp_lte_table(gas.T.L.mu_idx, gas.phys, gas.L,
                                                state, query_temperature, gas.T);
                  });
                  const Real expected = query == 4 ? Real(200) : Real(0.01) + endpoint * Real(1e-5);
                  if (std::abs(output.HostRead()[0] - expected) > 1e-10)
                    {
                      std::cerr << "Device endpoint/recovery control failed\n";
                      return 1;
                    }
                }
            }
        }
    }
  catch (const std::invalid_argument &error)
    {
      if (control)
        {
          std::cerr << "Unexpected rejection: " << error.what() << '\n';
          return 1;
        }
      std::cout << "REJECTED before returning a property: " << error.what() << '\n';
      return 0;
    }
  if (control)
    {
      if (!std::isfinite(result) || std::abs(result - 0.012) > 1e-12)
        {
          std::cerr << "Valid interior lookup failed: " << result << '\n';
          return 1;
        }
      std::cout << "Valid lookup control PASS\n";
      return 0;
    }
  std::cerr << "FAIL: invalid query/iteration was not rejected; returned " << result << '\n';
  return 1;
}
