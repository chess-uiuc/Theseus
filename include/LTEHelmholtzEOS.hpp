// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "LTEHelmholtzInterpolation.hpp"
#include "LTEHelmholtzTable.hpp"

namespace Theseus
{
  namespace HelmholtzEOS
  {
    // Share one cell lookup across requested properties, without coefficient
    // blocks, derivative output buffers, or nested Gibbs-to-pressure calls.
    struct Evaluation
    {
      HelmholtzInterpolation::Cell cell{};
      mfem::real_t xi = 0;
      mfem::real_t eta = 0;
      mfem::real_t density = 0;
      mfem::real_t temperature = 0;
      bool valid = false;

      MFEM_HOST_DEVICE Evaluation() = default;

      template<int DensityDerivative = 0, int TemperatureDerivative = 0>
      MFEM_HOST_DEVICE mfem::real_t H() const
      {
        return HelmholtzInterpolation::Evaluate<DensityDerivative, TemperatureDerivative>(cell, xi, eta);
      }

      MFEM_HOST_DEVICE mfem::real_t pressure() const
      {
        return density * density * H<1, 0>();
      }

      MFEM_HOST_DEVICE mfem::real_t energy() const
      {
        return H() - temperature * H<0, 1>();
      }

      MFEM_HOST_DEVICE mfem::real_t entropy() const
      {
        return -H<0, 1>();
      }

      MFEM_HOST_DEVICE mfem::real_t gibbs() const
      {
        return H() + density * H<1, 0>();
      }

      MFEM_HOST_DEVICE mfem::real_t pressure_density_derivative() const
      {
        return 2 * density * H<1, 0>() + density * density * H<2, 0>();
      }

      MFEM_HOST_DEVICE mfem::real_t pressure_temperature_derivative() const
      {
        return density * density * H<1, 1>();
      }

      MFEM_HOST_DEVICE mfem::real_t gibbs_density_derivative() const
      {
        return 2 * H<1, 0>() + density * H<2, 0>();
      }

      MFEM_HOST_DEVICE mfem::real_t gibbs_temperature_derivative() const
      {
        return H<0, 1>() + density * H<1, 1>();
      }

      MFEM_HOST_DEVICE mfem::real_t cv() const
      {
        return -temperature * H<0, 2>();
      }

      MFEM_HOST_DEVICE mfem::real_t cp() const
      {
        const auto mixed = H<1, 1>();
        return cv() + temperature * density * density * mixed * mixed / pressure_density_derivative();
      }

      MFEM_HOST_DEVICE mfem::real_t sound_speed_squared() const
      {
        const auto mixed = H<1, 1>();
        return pressure_density_derivative() - density * density * mixed * mixed / H<0, 2>();
      }

      MFEM_HOST_DEVICE mfem::real_t gamma() const
      {
        return cp() / cv();
      }

      MFEM_HOST_DEVICE mfem::real_t gas_constant() const
      {
        return pressure() / (density * temperature);
      }

      template<int Property>
      MFEM_HOST_DEVICE mfem::real_t transport(const HelmholtzTable::Tables &tables) const
      {
        static_assert(Property == HelmholtzTable::Layout::mu_idx ||
                      Property == HelmholtzTable::Layout::lambda_idx,
                      "Only viscosity and conductivity are bilinear properties");
        const auto offset = cell.potential - tables.tables.lte_table;
        const auto *plane = tables.tables.lte_table + Property * tables.L.nx * tables.L.ny + offset;
        return plane[0] * (1 - xi) * (1 - eta) + plane[1] * xi * (1 - eta)
               + plane[tables.L.nx] * (1 - xi) * eta + plane[tables.L.nx + 1] * xi * eta;
      }
    };

    // Check valid before consuming the result. Reject coordinates before any
    // cell dereference, rather than silently clamping or extrapolating.
    MFEM_HOST_DEVICE inline Evaluation At(const HelmholtzTable::Tables &tables,
                                          mfem::real_t density, mfem::real_t temperature)
    {
      Evaluation result;
      const auto &view = tables.tables;
      const int i = HelmholtzTable::FindCell(view.rho_grid, tables.L.nx, density);
      const int j = HelmholtzTable::FindCell(view.T_grid, tables.L.ny, temperature);
      if (i < 0 || j < 0)
        {
          return result;
        }
      const int offset = j * tables.L.nx + i;
      const int size = tables.L.nx * tables.L.ny;
      const auto dr = view.rho_grid[i + 1] - view.rho_grid[i];
      const auto dt = view.T_grid[j + 1] - view.T_grid[j];
      result.cell = {view.lte_table + offset, view.lte_table + size + offset,
                     view.lte_table + 2 * size + offset, view.lte_table + 3 * size + offset,
                     tables.L.nx, dr, dt};
      result.xi = (density - view.rho_grid[i]) / dr;
      result.eta = (temperature - view.T_grid[j]) / dt;
      result.density = density;
      result.temperature = temperature;
      result.valid = true;
      return result;
    }

    enum class RecoveryStatus
    {
      converged,
      outside_table,
      invalid_thermodynamics,
      iteration_limit
    };

    struct Recovery
    {
      mfem::real_t temperature = 0;
      int iterations = 0;
      RecoveryStatus status = RecoveryStatus::outside_table;

      MFEM_HOST_DEVICE Recovery() = default;
    };

    // The inverse table supplies a guess; Newton solves the forward energy with
    // its analytic cv. The adapter owns error reporting. No kernel allocation.
    MFEM_HOST_DEVICE inline Recovery TemperatureFromEnergy(const HelmholtzTable::Tables &tables,
                                                           mfem::real_t density, mfem::real_t energy)
    {
      Recovery result;
      const auto &view = tables.tables;
      const int i = HelmholtzTable::FindCell(view.rho_grid, tables.L.nx, density);
      const int j = HelmholtzTable::FindCell(view.e_grid, tables.L.ny, energy);
      if (i < 0 || j < 0)
        {
          return result;
        }
      const auto x = (density - view.rho_grid[i]) / (view.rho_grid[i + 1] - view.rho_grid[i]);
      const auto y = (energy - view.e_grid[j]) / (view.e_grid[j + 1] - view.e_grid[j]);
      const auto *p = view.inv_table + j * tables.L.nx + i;
      // Difference-form interpolation preserves a constant endpoint guess
      // exactly; four weighted terms can round it outside the forward domain.
      const auto lower = p[0] + x*(p[1] - p[0]);
      const auto upper = p[tables.L.nx] + x*(p[tables.L.nx + 1] - p[tables.L.nx]);
      result.temperature = lower + y*(upper - lower);
      for (int iteration = 0; iteration < 100; ++iteration)
        {
          const auto state = At(tables, density, result.temperature);
          if (!state.valid)
            {
              return result;
            }
          const auto cv = state.cv();
          const auto guess = state.energy();
          if (!std::isfinite(cv) || cv <= 0 || !std::isfinite(guess))
            {
              result.status = RecoveryStatus::invalid_thermodynamics;
              return result;
            }
          const auto step = (energy - guess) / cv;
          const auto next = result.temperature + step;
          ++result.iterations;
          const bool inside = std::isfinite(next) && next >= view.T_grid[0]
                              && next <= view.T_grid[tables.L.ny - 1];
          if (std::abs(step) <= 1e-9 + 1e-12 * result.temperature)
            {
              // Only a converged correction may be projected to an endpoint.
              // Otherwise an out-of-domain Newton step is an explicit failure.
              if (next < view.T_grid[0])
                {
                  result.temperature = view.T_grid[0];
                }
              else if (next > view.T_grid[tables.L.ny - 1])
                {
                  result.temperature = view.T_grid[tables.L.ny - 1];
                }
              else
                {
                  result.temperature = next;
                }
              result.status = RecoveryStatus::converged;
              return result;
            }
          result.temperature = next;
          if (!inside)
            {
              return result;
            }
        }
      result.status = RecoveryStatus::iteration_limit;
      return result;
    }

    enum class Quantity { pressure, gibbs };

    template<Quantity Property>
    MFEM_HOST_DEVICE inline mfem::real_t Value(const Evaluation &state)
    {
      if constexpr (Property == Quantity::pressure)
        {
          return state.pressure();
        }
      else
        {
          return state.gibbs();
        }
    }

    struct Gradient
    {
      mfem::real_t density;
      mfem::real_t beta;
    };

    // Stable secant of p or g along density, with T held fixed in one cell.
    template<Quantity Property>
    MFEM_HOST_DEVICE inline mfem::real_t DensitySecant(const Evaluation &first,
        const Evaluation &second, mfem::real_t eta)
    {
      using namespace HelmholtzInterpolation;
      const auto &cell = first.cell;
      const auto r1 = first.density;
      const auto r2 = second.density;
      const auto mean_hr = 0.5*(Evaluate<1, 0>(cell, first.xi, eta)
                                + Evaluate<1, 0>(cell, second.xi, eta));
      const auto divided_hr = DividedDifference<0, 1, 0>(cell, first.xi, eta, second.xi, eta);
      if constexpr (Property == Quantity::pressure)
        {
          return 0.5*(r1*r1 + r2*r2)*divided_hr + (r1 + r2)*mean_hr;
        }
      else
        {
          const auto divided_h = DividedDifference<0>(cell, first.xi, eta, second.xi, eta);
          return divided_h + 0.5*(r1 + r2)*divided_hr + mean_hr;
        }
    }

    // D_beta(f/T)=mean(f)-mean(T)*D_T(f), exactly for beta=1/T.
    template<Quantity Property>
    MFEM_HOST_DEVICE inline mfem::real_t BetaSecant(const Evaluation &first,
        const Evaluation &second, mfem::real_t xi, mfem::real_t density)
    {
      using namespace HelmholtzInterpolation;
      const auto &cell = first.cell;
      const auto mean_t = 0.5*(first.temperature + second.temperature);
      const auto mean_hr = 0.5*(Evaluate<1, 0>(cell, xi, first.eta)
                                + Evaluate<1, 0>(cell, xi, second.eta));
      const auto divided_hr = DividedDifference<1, 1, 0>(cell, xi, first.eta, xi, second.eta);
      const auto hr_beta = mean_hr - mean_t*divided_hr;
      if constexpr (Property == Quantity::pressure)
        {
          return density*density*hr_beta;
        }
      else
        {
          const auto mean_h = 0.5*(Evaluate<>(cell, xi, first.eta)
                                   + Evaluate<>(cell, xi, second.eta));
          const auto divided_h = DividedDifference<1>(cell, xi, first.eta, xi, second.eta);
          return mean_h - mean_t*divided_h + density*hr_beta;
        }
    }

    // Symmetric Itoh-Abe gradient of f/T in (rho,beta=1/T). Endpoints must be
    // valid and refer to the same table. The mixed states then lie in its domain.
    template<Quantity Property>
    MFEM_HOST_DEVICE inline Gradient DiscreteGradient(const HelmholtzTable::Tables &tables,
                                                       const Evaluation &left,
                                                       const Evaluation &right)
    {
      // The mathematical gradient is symmetric. Canonical endpoint order also
      // gives both face orientations the same floating-point subtraction order.
      const bool reverse = left.density > right.density ||
                           (left.density == right.density && left.temperature > right.temperature);
      const auto &first = reverse ? right : left;
      const auto &second = reverse ? left : right;
      const auto beta1 = 1 / first.temperature;
      const auto beta2 = 1 / second.temperature;
      const auto dr = second.density - first.density;
      const auto db = beta2 - beta1;
      const auto rmax = first.density > second.density ? first.density : second.density;
      const auto bmax = beta1 > beta2 ? beta1 : beta2;
      const bool density_difference = std::abs(dr) >= 2.22e-15 + 1.49e-8 * rmax;
      const bool beta_difference = std::abs(db) >= 2.22e-15 + 1.49e-8 * bmax;
      const auto f11 = Value<Property>(first);
      const auto f22 = Value<Property>(second);
      Gradient result;
      const bool same_cell = first.cell.potential == second.cell.potential;
      if (same_cell)
        {
          if (density_difference)
            {
              result.density = 0.5*(DensitySecant<Property>(first, second, first.eta)*beta1
                                    + DensitySecant<Property>(first, second, second.eta)*beta2);
            }
          if (beta_difference)
            {
              result.beta = 0.5*(BetaSecant<Property>(first, second, first.xi, first.density)
                                 + BetaSecant<Property>(first, second, second.xi, second.density));
            }
        }
      else
        {
          if (density_difference || beta_difference)
            {
              const auto f12 = Value<Property>(At(tables, first.density, second.temperature));
              const auto f21 = Value<Property>(At(tables, second.density, first.temperature));
              if (density_difference)
                {
                  result.density = 0.5 * ((f21 * beta1 - f11 * beta1)
                                          + (f22 * beta2 - f12 * beta2)) / dr;
                }
              if (beta_difference)
                {
                  result.beta = 0.5 * ((f12 * beta2 - f11 * beta1)
                                      + (f22 * beta2 - f21 * beta1)) / db;
                }
            }
        }
      if (!density_difference)
        {
          if constexpr (Property == Quantity::pressure)
            {
              result.density = 0.5 * (first.pressure_density_derivative() * beta1
                                      + second.pressure_density_derivative() * beta2);
            }
          else
            {
              result.density = 0.5 * (first.gibbs_density_derivative() * beta1
                                      + second.gibbs_density_derivative() * beta2);
            }
        }
      if (!beta_difference)
        {
          mfem::real_t derivative1;
          mfem::real_t derivative2;
          if constexpr (Property == Quantity::pressure)
            {
              derivative1 = first.pressure_temperature_derivative();
              derivative2 = second.pressure_temperature_derivative();
            }
          else
            {
              derivative1 = first.gibbs_temperature_derivative();
              derivative2 = second.gibbs_temperature_derivative();
            }
          result.beta = 0.5 * ((f11 - first.temperature * derivative1)
                               + (f22 - second.temperature * derivative2));
        }
      return result;
    }

    struct Means
    {
      mfem::real_t density;
      mfem::real_t energy;
    };

    MFEM_HOST_DEVICE inline Means ThermodynamicMeans(const HelmholtzTable::Tables &tables,
                                                      const Evaluation &left,
                                                      const Evaluation &right)
    {
      const auto pressure = DiscreteGradient<Quantity::pressure>(tables, left, right);
      const auto gibbs = DiscreteGradient<Quantity::gibbs>(tables, left, right);
      const auto density = pressure.density / gibbs.density;
      return {density, gibbs.beta - pressure.beta / density};
    }
  }
}
