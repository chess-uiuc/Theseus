// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "GasState.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <variant>
#include <type_traits>
#include <vector>

namespace Theseus
{
// Host-only physical input. A variant makes under/over-specification impossible
// after parsing. Scalars/composition require an explicit future model contract.
struct PressureTemperature { mfem::real_t pressure, temperature; };
struct DensityTemperature { mfem::real_t density, temperature; };
struct DensityPressure { mfem::real_t density, pressure; };
using ThermodynamicInput = std::variant<PressureTemperature, DensityTemperature,
                                        DensityPressure>;
struct PhysicalStateInput
{
  ThermodynamicInput thermo;
  std::vector<mfem::real_t> velocity;
};
struct ThermodynamicState
{
  mfem::real_t density, temperature, specific_internal_energy;
};

inline void RequirePositiveFinite(mfem::real_t x, const char *name)
{
  if (!std::isfinite(x) || x <= 0)
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
}
inline void ValidatePhysicalState(const PhysicalStateInput &input,
                                  const StateLayout &layout)
{
  if (layout.dim < 1 || layout.dim > 3 || layout.num_scalars != 0)
    throw std::invalid_argument("Physical state conversion requires 1-3 dimensions and no scalars");
  if (input.velocity.size() != static_cast<size_t>(layout.dim))
    throw std::invalid_argument("Physical state velocity dimension does not match gas layout");
  for (auto u : input.velocity)
    if (!std::isfinite(u)) throw std::invalid_argument("Velocity must be finite");
  std::visit([](const auto &pair) {
    using Pair = std::decay_t<decltype(pair)>;
    if constexpr (!std::is_same_v<Pair, PressureTemperature>)
      RequirePositiveFinite(pair.density, "Density");
    if constexpr (!std::is_same_v<Pair, DensityTemperature>)
      RequirePositiveFinite(pair.pressure, "Pressure");
    if constexpr (!std::is_same_v<Pair, DensityPressure>)
      RequirePositiveFinite(pair.temperature, "Temperature");
  }, input.thermo);
}

// Assemble only after EOS conversion succeeds. Energy may be negative when an
// EOS uses a reference offset; positivity is not a universal energy constraint.
inline void PackPhysicalState(const PhysicalStateInput &input,
                              const ThermodynamicState &thermo,
                              const StateLayout &layout, mfem::Vector &out)
{
  RequirePositiveFinite(thermo.density, "Converted density");
  RequirePositiveFinite(thermo.temperature, "Converted temperature");
  mfem::Vector result(layout.nequations());
  PointStateViewRW state(result.GetData());
  state.set_mass(layout, thermo.density);
  mfem::real_t kinetic = 0;
  for (int d = 0; d < layout.dim; ++d)
    {
      state.set_momentum(layout, d, thermo.density * input.velocity[d]);
      kinetic += 0.5 * input.velocity[d] * input.velocity[d];
    }
  state.set_energy(layout, thermo.density * (thermo.specific_internal_energy + kinetic));
  for (int q = 0; q < result.Size(); ++q)
    if (!std::isfinite(result[q]))
      throw std::invalid_argument("Physical state produces nonfinite conservative values");
  out = result;
}
}
