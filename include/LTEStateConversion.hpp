// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "PhysicalState.hpp"
#include "LTETable.hpp"
#include "LTETemperatureRecovery.hpp"
#include <algorithm>
#include <limits>
#include <sstream>

namespace Theseus
{
namespace detail
{
inline void CheckTableCoordinate(mfem::real_t x, const mfem::real_t *grid,
                                 int n, const char *name)
{
  if (!std::isfinite(x) || x < grid[0] || x > grid[n-1])
    {
      std::ostringstream message;
      message << "LTE " << name << "=" << x << " outside [" << grid[0]
              << ", " << grid[n-1] << "]";
      throw std::invalid_argument(message.str());
    }
}
inline void ValidateGrid(const mfem::real_t *grid, int n)
{
  if (!grid || n < 2) throw std::invalid_argument("Missing LTE grid or fewer than two samples");
  for (int i = 0; i < n; ++i)
    if (!std::isfinite(grid[i]) || (i && grid[i] <= grid[i-1]))
      throw std::invalid_argument("LTE grid must be finite and strictly increasing");
}

// A bilinear table restricted to one coordinate is piecewise linear. Solve
// each bracket exactly, rejecting multiple roots and constant-pressure spans.
// No unconstrained iteration and no evaluation outside the supplied grid.
template<class Evaluate>
mfem::real_t SolveTablePressure(const mfem::real_t *grid, int n,
                               mfem::real_t target, Evaluate evaluate)
{
  bool found = false;
  mfem::real_t root = 0;
  auto add_root = [&](mfem::real_t value) {
    if (found && value != root)
      throw std::invalid_argument("LTE pressure input has multiple table roots");
    found = true; root = value;
  };
  auto left = evaluate(grid[0]);
  for (int i = 0; i + 1 < n; ++i)
    {
      const auto right = evaluate(grid[i+1]);
      if (!std::isfinite(left) || !std::isfinite(right))
        throw std::invalid_argument("Nonfinite LTE pressure table value");
      if (left == target && right == target)
        throw std::invalid_argument("LTE pressure input has a nonunique constant interval");
      if (left == target) add_root(grid[i]);
      if (right == target) add_root(grid[i+1]);
      if ((left < target && target < right) || (right < target && target < left))
        add_root(grid[i] + (grid[i+1] - grid[i]) * ((target-left)/(right-left)));
      left = right;
    }
  if (!found) throw std::invalid_argument("LTE pressure input has no root inside the table");
  return root;
}
}

// Host-only adapter around the EOS's existing forward interpolation. The EOS
// retains ownership of thermodynamic semantics and the caller owns all tables.
template<class EOS>
ThermodynamicState LTEPhysicalThermodynamics(const EOS &eos,
    const PhysicsConstants &phys, const StateLayout &layout,
    const ThermodynamicInput &input, const LTETable::LTETables &tables)
{
  const auto &view = tables.tables;
  const auto &tl = tables.L;
  detail::ValidateGrid(view.rho_grid, tl.nx);
  detail::ValidateGrid(view.T_grid, tl.ny);
  detail::ValidateGrid(view.e_grid, tl.ny);
  if (!view.lte_table || !view.inv_table)
    throw std::invalid_argument("Missing LTE forward or inverse table");
  mfem::real_t values[MAXEQ] = {};
  PointStateViewRW state(values);
  auto property = [&](int index, mfem::real_t rho, mfem::real_t temperature) {
    detail::CheckTableCoordinate(rho, view.rho_grid, tl.nx, "density");
    detail::CheckTableCoordinate(temperature, view.T_grid, tl.ny, "temperature");
    state.set_mass(layout, rho);
    const auto value = eos.biinterp_lte_table(index, phys, layout, state, temperature, tables);
    if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite LTE property");
    return value;
  };
  mfem::real_t rho = 0, temperature = 0;
  std::visit([&](const auto &pair) {
    using Pair = std::decay_t<decltype(pair)>;
    if constexpr (std::is_same_v<Pair, PressureTemperature>)
      {
        temperature = pair.temperature;
        rho = detail::SolveTablePressure(view.rho_grid, tl.nx, pair.pressure,
          [&](auto r) { return property(tl.P_idx, r, temperature); });
      }
    else if constexpr (std::is_same_v<Pair, DensityPressure>)
      {
        rho = pair.density;
        temperature = detail::SolveTablePressure(view.T_grid, tl.ny, pair.pressure,
          [&](auto t) { return property(tl.P_idx, rho, t); });
      }
    else { rho = pair.density; temperature = pair.temperature; }
  }, input);
  const auto energy = property(tl.e_idx, rho, temperature);
  detail::CheckTableCoordinate(energy, view.e_grid, tl.ny, "specific internal energy");
  return {rho, temperature, energy};
}
// Verify the runtime inverse/Newton path on the host, guarding every forward
// evaluation. A forward-table state alone is insufficient: malformed inverse
// guesses or inconsistent cv data must fail before an unchecked kernel lookup.
template<class EOS, class State>
void ValidateLTERoundTrip(const EOS &eos, const PhysicsConstants &phys,
                         const StateLayout &layout, const State &state,
                         mfem::real_t expected_temperature,
                         const LTETable::LTETables &tables)
{
  const auto &v = tables.tables;
  const auto &l = tables.L;
  const auto energy = eos.specific_internal_energy(phys, layout, state, tables);
  detail::CheckTableCoordinate(energy, v.e_grid, l.ny, "packed specific internal energy");
  detail::LTETemperatureRecovery recovery(eos.biinterp_inverse_table(phys, layout, state, tables));
  while (recovery.NeedsIteration())
    {
      detail::CheckTableCoordinate(recovery.temperature, v.T_grid, l.ny, "inverse temperature");
      const auto guess = eos.biinterp_lte_table(l.e_idx, phys, layout, state, recovery.temperature, tables);
      const auto cv = eos.biinterp_lte_table(l.cv_idx, phys, layout, state, recovery.temperature, tables);
      RequirePositiveFinite(cv, "LTE heat capacity");
      recovery.Update(energy, guess, cv);
      detail::CheckTableCoordinate(recovery.temperature, v.T_grid, l.ny, "Newton temperature");
      if (recovery.ExceededIterations())
        throw std::invalid_argument("LTE physical state inverse iteration did not converge");
    }
  if (std::abs(recovery.temperature - expected_temperature) >
      1e-9 * std::max(mfem::real_t(1), std::abs(expected_temperature)))
    throw std::invalid_argument("LTE physical state fails temperature round trip");
}

}
