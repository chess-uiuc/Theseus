// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "LTEHelmholtzEOS.hpp"
#include "PhysicalState.hpp"
#include <algorithm>
#include <limits>
#include <vector>
#include <utility>
#include <sstream>

namespace Theseus
{
  namespace HelmholtzTable
  {
    namespace detail
    {
      using Polynomial = std::vector<long double>;

      inline long double Value(const Polynomial &p, long double x)
      {
        long double result = 0;
        for (auto i = p.rbegin(); i != p.rend(); ++i)
          {
            result = result*x + *i;
          }
        return result;
      }

      // Isolate roots at derivative critical points, then bisect each monotone
      // span. Endpoint signs alone miss interior pairs and tangential roots.
      inline std::vector<long double> Roots(Polynomial p)
      {
        long double scale = 0;
        for (auto c : p)
          {
            scale += std::abs(c);
          }
        const auto tolerance = 64 * std::numeric_limits<mfem::real_t>::epsilon() * scale;
        while (p.size() > 1 && std::abs(p.back()) <= tolerance)
          {
            p.pop_back();
          }
        if (p.size() == 1)
          {
            if (std::abs(p[0]) <= tolerance)
              {
                throw std::invalid_argument("Helmholtz pressure input has a constant root interval");
              }
            return {};
          }
        Polynomial derivative(p.size() - 1);
        for (size_t i = 1; i < p.size(); ++i)
          {
            derivative[i - 1] = i*p[i];
          }
        std::vector<long double> points{0};
        if (derivative.size() > 1)
          {
            const auto critical = Roots(derivative);
            points.insert(points.end(), critical.begin(), critical.end());
          }
        points.push_back(1);
        std::sort(points.begin(), points.end());
        std::vector<long double> roots;
        auto add = [&](long double root)
          {
            if (roots.empty() || std::abs(root - roots.back()) > 512*std::numeric_limits<mfem::real_t>::epsilon())
              {
                roots.push_back(root);
              }
          };
        for (size_t i = 0; i + 1 < points.size(); ++i)
          {
            auto low = points[i];
            auto high = points[i + 1];
            auto f_low = Value(p, low);
            const auto f_high = Value(p, high);
            if (std::abs(f_low) <= tolerance)
              {
                add(low);
              }
            if (std::abs(f_low) > tolerance && std::abs(f_high) > tolerance &&
                ((f_low < 0) != (f_high < 0)))
              {
                for (int iteration = 0; iteration < 80; ++iteration)
                  {
                    const auto middle = (low + high)/2;
                    const auto f = Value(p, middle);
                    if ((f < 0) == (f_low < 0))
                      {
                        low = middle;
                        f_low = f;
                      }
                    else
                      {
                        high = middle;
                      }
                  }
                add((low + high)/2);
              }
            if (std::abs(f_high) <= tolerance)
              {
                add(high);
              }
          }
        return roots;
      }

      template<class Evaluate>
      mfem::real_t SolvePressure(const mfem::real_t *grid, int count,
                                 int degree, mfem::real_t target, Evaluate evaluate)
      {
        std::vector<mfem::real_t> roots;
        for (int cell = 0; cell + 1 < count; ++cell)
          {
            // A fixed-T cell pressure is quartic in rho; fixed-rho pressure
            // is cubic in T. Recover that polynomial, never a linear surrogate.
            Polynomial differences(degree + 1);
            for (int k = 0; k <= degree; ++k)
              {
                const auto x = mfem::real_t(k)/degree;
                const auto value = evaluate(grid[cell] + (grid[cell + 1] - grid[cell])*x);
                if (!std::isfinite(value))
                  {
                    throw std::invalid_argument("Nonfinite Helmholtz pressure during inversion");
                  }
                differences[k] = static_cast<long double>(value) - target;
              }
            for (int order = 1; order <= degree; ++order)
              {
                for (int k = degree; k >= order; --k)
                  {
                    differences[k] = (differences[k] - differences[k - 1])*degree/order;
                  }
              }
            Polynomial polynomial(degree + 1, 0);
            Polynomial basis{1};
            for (int k = 0; k <= degree; ++k)
              {
                for (size_t j = 0; j < basis.size(); ++j)
                  {
                    polynomial[j] += differences[k]*basis[j];
                  }
                Polynomial next(basis.size() + 1, 0);
                for (size_t j = 0; j < basis.size(); ++j)
                  {
                    next[j] -= basis[j]*k/degree;
                    next[j + 1] += basis[j];
                  }
                basis = std::move(next);
              }
            for (const auto root : Roots(polynomial))
              {
                const auto value = grid[cell] + (grid[cell + 1] - grid[cell])*root;
                const auto tolerance = 512*std::numeric_limits<mfem::real_t>::epsilon()
                                       * std::max(mfem::real_t(1), std::abs(mfem::real_t(value)));
                if (roots.empty() || std::abs(value - roots.back()) > tolerance)
                  {
                    roots.push_back(value);
                  }
              }
          }
        if (roots.size() != 1)
          {
            throw std::invalid_argument("Helmholtz pressure input requires exactly one table root");
          }
        return roots[0];
      }
    }
  }

  template<class EOS>
  ThermodynamicState LTEPhysicalThermodynamics(const EOS &, const PhysicsConstants &,
      const StateLayout &, const ThermodynamicInput &input, const HelmholtzTable::Tables &tables)
  {
    mfem::real_t density = 0;
    mfem::real_t temperature = 0;
    auto pressure = [&](mfem::real_t r, mfem::real_t t)
      {
        const auto state = HelmholtzEOS::At(tables, r, t);
        if (!state.valid)
          {
            throw std::invalid_argument("Physical state outside Helmholtz table");
          }
        return state.pressure();
      };
    std::visit([&](const auto &pair)
      {
        using Pair = std::decay_t<decltype(pair)>;
        if constexpr (std::is_same_v<Pair, PressureTemperature>)
          {
            temperature = pair.temperature;
            density = HelmholtzTable::detail::SolvePressure(tables.tables.rho_grid, tables.L.nx,
              4, pair.pressure, [&](auto r) { return pressure(r, temperature); });
          }
        else if constexpr (std::is_same_v<Pair, DensityPressure>)
          {
            density = pair.density;
            temperature = HelmholtzTable::detail::SolvePressure(tables.tables.T_grid, tables.L.ny,
              3, pair.pressure, [&](auto t) { return pressure(density, t); });
          }
        else
          {
            density = pair.density;
            temperature = pair.temperature;
          }
      }, input);
    const auto state = HelmholtzEOS::At(tables, density, temperature);
    if (!state.valid || !std::isfinite(state.energy()))
      {
        throw std::invalid_argument("Invalid Helmholtz physical state");
      }
    return {density, temperature, state.energy()};
  }

  template<class EOS, class State>
  void ValidateLTERoundTrip(const EOS &eos, const PhysicsConstants &phys,
      const StateLayout &layout, const State &state, mfem::real_t expected,
      const HelmholtzTable::Tables &tables)
  {
    const auto energy = eos.specific_internal_energy(phys, layout, state, tables);
    const auto recovery = HelmholtzEOS::TemperatureFromEnergy(tables, state.mass(layout), energy);
    if (recovery.status != HelmholtzEOS::RecoveryStatus::converged ||
        std::abs(recovery.temperature - expected) > 1e-9 + 1e-12*expected)
      {
        std::ostringstream message;
        message.precision(17);
        message << "Helmholtz temperature round trip: expected=" << expected
                << " recovered=" << recovery.temperature << " energy=" << energy
                << " status=" << int(recovery.status);
        throw std::invalid_argument(message.str());
      }
  }
}
