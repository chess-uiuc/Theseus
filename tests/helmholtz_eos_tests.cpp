// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "LTEHelmholtzEOS.hpp"
#include "helmholtz_difference_cases.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

namespace
{
  namespace Table = Theseus::HelmholtzTable;
  namespace EOS = Theseus::HelmholtzEOS;
  using real_t = mfem::real_t;
  int checks = 0;
  int failures = 0;

  void Check(bool condition, const std::string &name)
  {
    ++checks;
    if (!condition)
      {
        ++failures;
        if (failures < 20)
          {
            std::cerr << "FAIL: " << name << '\n';
          }
      }
  }

  void Near(real_t actual, real_t expected, const std::string &name, real_t extra_roundoff = 0)
  {
    // Small exactly represented polynomial fixture: this bounds arithmetic
    // roundoff, not the eventual CPU/GPU evolution acceptance criterion.
    const auto tolerance = 4096 * std::numeric_limits<real_t>::epsilon()
                           * std::max(real_t(1), std::abs(expected)) + extra_roundoff;
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
      {
        std::cerr << std::setprecision(17) << name << ": actual=" << actual
                  << " expected=" << expected << " tolerance=" << tolerance << '\n';
      }
    Check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, name);
  }

  // H = 2*r + r^2/2 + r*T/4 - T^2. This is an exactly representable
  // bicubic with positive cv and p_r over the complete test domain.
  // Independent derivatives/properties below come from this power expression.
  void Fill(Table::Data &data, const Table::Layout &layout)
  {
    data.rho_grid.SetSize(3);
    data.T_grid.SetSize(3);
    data.e_grid.SetSize(3);
    data.rho_grid[0] = 0.5;
    data.rho_grid[1] = 1.25;
    data.rho_grid[2] = 2;
    data.T_grid[0] = 2;
    data.T_grid[1] = 3;
    data.T_grid[2] = 5;
    data.e_grid[0] = 0;
    data.e_grid[1] = 16;
    data.e_grid[2] = 32;
    data.lte_table.SetSize(Table::Layout::num_properties * layout.nx * layout.ny);
    data.inv_table.SetSize(layout.nx * layout.ny);
    data.inv_table = 5; // Deliberately inexact starting guesses; Newton must work.
    for (int j = 0; j < layout.ny; ++j)
      {
        const auto t = data.T_grid[j];
        for (int i = 0; i < layout.nx; ++i)
          {
            const auto r = data.rho_grid[i];
            auto *p = data.lte_table.HostWrite();
            p[layout.property_index(Table::Layout::H_idx, i, j)] = 2*r + r*r/2 + r*t/4 - t*t;
            p[layout.property_index(Table::Layout::Hr_idx, i, j)] = 2 + r + t/4;
            p[layout.property_index(Table::Layout::HT_idx, i, j)] = r/4 - 2*t;
            p[layout.property_index(Table::Layout::HrT_idx, i, j)] = 0.25;
            p[layout.property_index(Table::Layout::mu_idx, i, j)] = 1 + 2*r + 3*t + r*t;
            p[layout.property_index(Table::Layout::lambda_idx, i, j)] = 4 + r + 2*t + 2*r*t;
          }
      }
  }

  void Properties(const Table::Tables &table)
  {
    for (const real_t r : {0.5, 0.875, 1.25, 1.375, 1.625, 2.0})
      {
        for (const real_t t : {2.0, 2.5, 3.0, 4.0, 4.125, 5.0})
          {
            const auto q = EOS::At(table, r, t);
            Check(q.valid, "valid density/temperature including upper endpoint");
            const auto pressure = r*r*(2 + r + t/4);
            const auto pr = 4*r + 3*r*r + r*t/2;
            const auto cv = 2*t;
            const auto cp = cv + t*r*r/(16*pr);
            Near(q.H(), 2*r + r*r/2 + r*t/4 - t*t, "potential");
            Near(q.energy(), 2*r + r*r/2 + t*t, "energy");
            Near(q.entropy(), 2*t - r/4, "entropy");
            Near(q.pressure(), pressure, "pressure");
            Near(q.gibbs(), 4*r + 1.5*r*r + r*t/2 - t*t, "Gibbs");
            Near(q.pressure_density_derivative(), pr, "p_r");
            Near(q.pressure_temperature_derivative(), r*r/4, "p_T");
            Near(q.gibbs_density_derivative(), 4 + 3*r + t/2, "g_r");
            Near(q.gibbs_temperature_derivative(), r/2 - 2*t, "g_T");
            Near(q.cv(), cv, "cv");
            Near(q.cp(), cp, "cp");
            Near(q.sound_speed_squared(), pr + r*r/32, "sound speed squared");
            Near(q.gamma(), cp/cv, "gamma");
            Near(q.gas_constant(), pressure/(r*t), "effective gas constant");
            Near(q.transport<Table::Layout::mu_idx>(table), 1 + 2*r + 3*t + r*t, "viscosity plane");
            Near(q.transport<Table::Layout::lambda_idx>(table), 4 + r + 2*t + 2*r*t, "conductivity plane");
            const auto recovered = EOS::TemperatureFromEnergy(table, r, 2*r + r*r/2 + t*t);
            Check(recovered.status == EOS::RecoveryStatus::converged, "temperature recovery status r=" + std::to_string(r) + " t=" + std::to_string(t) + " recovered=" + std::to_string(recovered.temperature));
            Near(recovered.temperature, t, "forward/inverse temperature round trip");
          }
      }
  }

  void InvalidInputs(Table::Data &data, const Table::Layout &layout)
  {
    auto table = Table::HostView(data, layout);
    const auto nan = std::numeric_limits<real_t>::quiet_NaN();
    const auto inf = std::numeric_limits<real_t>::infinity();
    for (const auto r : {0.49, 2.01, nan, inf})
      {
        Check(!EOS::At(table, r, 3).valid, "reject invalid density");
        Check(EOS::TemperatureFromEnergy(table, r, 10).status == EOS::RecoveryStatus::outside_table,
              "reject inverse invalid density");
      }
    for (const auto t : {1.99, 5.01, nan, inf})
      {
        Check(!EOS::At(table, 1, t).valid, "reject invalid temperature");
      }
    for (const auto e : {-1.0, 33.0, nan, inf})
      {
        Check(EOS::TemperatureFromEnergy(table, 1, e).status == EOS::RecoveryStatus::outside_table,
              "reject inverse invalid energy");
      }
    // Energy can be in the rectangular inverse grid but have no forward root
    // at this density. Never access a forward cell outside its domain.
    Check(EOS::TemperatureFromEnergy(table, 1, 0).status == EOS::RecoveryStatus::outside_table,
          "Newton leaves forward domain");
    data.inv_table = nan;
    table = Table::HostView(data, layout);
    Check(EOS::TemperatureFromEnergy(table, 1, 10).status == EOS::RecoveryStatus::outside_table,
          "reject nonfinite inverse guess");
    Fill(data, layout);
    data.lte_table = 0.0;
    table = Table::HostView(data, layout);
    Check(EOS::TemperatureFromEnergy(table, 1, 10).status == EOS::RecoveryStatus::invalid_thermodynamics,
          "reject zero cv");
    Fill(data, layout);
    const int size = layout.nx * layout.ny;
    for (int k = 0; k < 4*size; ++k)
      {
        data.lte_table[k] *= -1;
      }
    table = Table::HostView(data, layout);
    Check(EOS::TemperatureFromEnergy(table, 1, 10).status == EOS::RecoveryStatus::invalid_thermodynamics,
          "reject negative cv");
    Fill(data, layout);
    data.T_grid[1] = data.T_grid[0];
    bool rejected = false;
    try
      {
        Table::HostView(data, layout);
      }
    catch (const std::invalid_argument &)
      {
        rejected = true;
      }
    Check(rejected, "reject duplicate grid coordinate");
    Fill(data, layout);
    data.lte_table.SetSize(9*size);
    rejected = false;
    try
      {
        Table::HostView(data, layout);
      }
    catch (const std::invalid_argument &)
      {
        rejected = true;
      }
    Check(rejected, "reject legacy nine-plane data as Helmholtz table");
    Fill(data, layout);
  }

  void Means(const Table::Tables &table)
  {
    const real_t r1 = 0.875;
    const real_t t1 = 2.5;
    // Exercise both difference quotients, each independent fallback, and both
    // fallbacks together. Small cases straddle the donor thresholds.
    for (const real_t dr : {0.0, 1e-9, 1e-6, 0.5})
      {
        for (const real_t dt : {0.0, 1e-9, 1e-4, 1.5})
          {
            const auto r2 = r1 + dr;
            const auto t2 = t1 + dt;
            const auto beta1 = 1/t1;
            const auto beta2 = 1/t2;
            const auto db = beta2 - beta1;
            const bool use_r = dr >= 2.22e-15 + 1.49e-8*r2;
            const bool use_b = std::abs(db) >= 2.22e-15 + 1.49e-8*beta1;
            // Exact divided differences of the power-form f/T, independently
            // simplified before floating-point evaluation (no table lookups).
            const auto pr = use_r
              ? 0.5*(beta1 + beta2)*(2*(r1 + r2) + r1*r1 + r1*r2 + r2*r2) + (r1 + r2)/4
              : 0.5*((4*r1 + 3*r1*r1 + r1*t1/2)/t1 + (4*r2 + 3*r2*r2 + r2*t2/2)/t2);
            const auto gr = use_r
              ? 0.5*(beta1 + beta2)*(4 + 1.5*(r1 + r2)) + 0.5
              : 0.5*((4 + 3*r1 + t1/2)/t1 + (4 + 3*r2 + t2/2)/t2);
            const auto pb = 0.5*(2*r1*r1 + r1*r1*r1 + 2*r2*r2 + r2*r2*r2);
            const auto gb = 0.5*(4*r1 + 1.5*r1*r1 + 4*r2 + 1.5*r2*r2)
                            + (use_b ? t1*t2 : 0.5*(t1*t1 + t2*t2));
            const auto left = EOS::At(table, r1, t1);
            const auto right = EOS::At(table, r2, t2);
            const auto p = EOS::DiscreteGradient<EOS::Quantity::pressure>(table, left, right);
            const auto g = EOS::DiscreteGradient<EOS::Quantity::gibbs>(table, left, right);
            // Quotients amplify interpolation/subtraction roundoff by 1/delta.
            // This checks the preserved donor formula; it is not a parity waiver.
            const auto epsilon = std::numeric_limits<real_t>::epsilon();
            const auto r_error = use_r ? 256*epsilon/std::abs(r2 - r1) : 0;
            const auto b_error = use_b ? 256*epsilon/std::abs(db) : 0;
            Near(p.density, pr, "pressure density discrete gradient", r_error);
            Near(g.density, gr, "Gibbs density discrete gradient", r_error);
            Near(p.beta, pb, "pressure beta discrete gradient", b_error);
            Near(g.beta, gb, "Gibbs beta discrete gradient", b_error);
            const auto means = EOS::ThermodynamicMeans(table, left, right);
            const auto reverse = EOS::ThermodynamicMeans(table, right, left);
            const auto rho = pr/gr;
            Near(means.density, rho, "thermodynamic density mean", 8*r_error);
            Near(means.energy, gb - pb/rho, "thermodynamic energy mean", 32*(r_error + b_error));
            Near(reverse.density, means.density, "density mean symmetry");
            Near(reverse.energy, means.energy, "energy mean symmetry");
            if (dr == 0 && dt == 0)
              {
                Near(means.density, r1, "equal-state density consistency");
                Near(means.energy, 2*r1 + r1*r1/2 + t1*t1, "equal-state energy consistency");
              }
          }
      }
  }

  void NearStateReference()
  {
    // References use an independent 80-digit power-basis expansion of the
    // corner data. The existing roundoff tolerances are unchanged.
    for (const auto &fixture : HelmholtzDifferenceReference::cases)
      {
        Table::Data data;
        data.rho_grid.SetSize(2);
        data.T_grid.SetSize(2);
        data.e_grid.SetSize(2);
        data.rho_grid[0] = fixture.bounds[0];
        data.rho_grid[1] = fixture.bounds[1];
        data.T_grid[0] = fixture.bounds[2];
        data.T_grid[1] = fixture.bounds[3];
        data.e_grid[0] = -1e9;
        data.e_grid[1] = 1e9;
        data.inv_table.SetSize(4);
        data.inv_table = 0.5*(fixture.bounds[2] + fixture.bounds[3]);
        data.lte_table.SetSize(24);
        data.lte_table = 0.0;
        for (int i = 0; i < 16; ++i)
          {
            data.lte_table[i] = fixture.jets[i];
          }
        const auto tables = Table::DeviceView(data, Table::Layout{2, 2});
        const real_t rho1 = fixture.states[0];
        const real_t t1 = fixture.states[1];
        const real_t rho2 = fixture.states[2];
        const real_t t2 = fixture.states[3];
        mfem::Vector result(4);
        auto *out = result.Write(); // Every entry is written; no sentinels.
        mfem::forall(2, [=] MFEM_HOST_DEVICE (int reverse)
          {
            const auto first = EOS::At(tables, reverse ? rho2 : rho1, reverse ? t2 : t1);
            const auto second = EOS::At(tables, reverse ? rho1 : rho2, reverse ? t1 : t2);
            const auto means = EOS::ThermodynamicMeans(tables, first, second);
            out[2*reverse] = means.density;
            out[2*reverse + 1] = means.energy;
          });
        const auto *values = result.HostRead();
        Near(values[0], fixture.expected[0], "near-state density: 80-digit reference");
        Near(values[1], fixture.expected[1], "near-state energy: 80-digit reference");
        Near(values[2], fixture.expected[0], "reversed near-state density reference");
        Near(values[3], fixture.expected[1], "reversed near-state energy reference");
        Check(values[0] == values[2] && values[1] == values[3], "near-state exact reversal symmetry");
      }
  }

  void Kernel(Table::Data &data, const Table::Layout &layout)
  {
    const auto table = Table::DeviceView(data, layout);
    mfem::Vector results(25*9);
    auto *out = results.Write();
    mfem::forall(25, [=] MFEM_HOST_DEVICE (int index)
      {
        const real_t r = 0.5 + (index%5)*0.375;
        const real_t t = 2 + (index/5)*0.75;
        const auto q = EOS::At(table, r, t);
        const auto recovery = EOS::TemperatureFromEnergy(table, r, 2*r + r*r/2 + t*t);
        out[9*index] = q.pressure();
        out[9*index + 1] = q.energy();
        out[9*index + 2] = q.sound_speed_squared();
        out[9*index + 3] = recovery.temperature;
        out[9*index + 4] = recovery.status == EOS::RecoveryStatus::converged ? 1 : 0;
        const auto means = EOS::ThermodynamicMeans(table, q, EOS::At(table, 1.25, 3));
        out[9*index + 5] = means.density;
        out[9*index + 6] = means.energy;
        const auto reverse = EOS::ThermodynamicMeans(table, EOS::At(table, 1.25, 3), q);
        out[9*index + 7] = reverse.density;
        out[9*index + 8] = reverse.energy;
      });
    const auto *values = results.HostRead();
    for (int index = 0; index < 25; ++index)
      {
        const real_t r = 0.5 + (index%5)*0.375;
        const real_t t = 2 + (index/5)*0.75;
        Near(values[9*index], r*r*(2 + r + t/4), "kernel pressure");
        Near(values[9*index + 1], 2*r + r*r/2 + t*t, "kernel energy");
        Near(values[9*index + 2], 4*r + 3*r*r + r*t/2 + r*r/32, "kernel acoustic relation");
        Near(values[9*index + 3], t, "kernel recovery");
        Check(values[9*index + 4] == 1, "kernel recovery status");
        // Independent well-separated quotient references, also valid at equality.
        const real_t r2 = 1.25;
        const real_t t2 = 3;
        const auto pr = 0.5*(1/t + 1/t2)*(2*(r + r2) + r*r + r*r2 + r2*r2) + (r + r2)/4;
        const auto gr = 0.5*(1/t + 1/t2)*(4 + 1.5*(r + r2)) + 0.5;
        const auto pb = 0.5*(2*r*r + r*r*r + 2*r2*r2 + r2*r2*r2);
        const auto gb = 0.5*(4*r + 1.5*r*r + 4*r2 + 1.5*r2*r2) + t*t2;
        Near(values[9*index + 5], pr/gr, "kernel density mean");
        Near(values[9*index + 6], gb - pb*gr/pr, "kernel energy mean");
        Near(values[9*index + 7], values[9*index + 5], "kernel density mean symmetry");
        Near(values[9*index + 8], values[9*index + 6], "kernel energy mean symmetry");
      }
  }
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA))
    {
      std::cerr << "CUDA backend was not selected\n";
      return 1;
    }
  const Table::Layout layout{3, 3};
  Table::Data data;
  Fill(data, layout);
  Properties(Table::HostView(data, layout));
  Means(Table::HostView(data, layout));
  InvalidInputs(data, layout);
  Kernel(data, layout);
  NearStateReference();
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
