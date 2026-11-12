// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "LTEHelmholtzEOS.hpp"
#include <limits>

namespace Theseus
{
  namespace HelmholtzTable
  {
#if defined(USE_PLATO) && defined(THESEUS_PLATO_HAS_ENTROPY)
    inline void fill_table(const Layout &L, const mfem::real_t* rho_grid, const mfem::real_t* T_grid,
                           mfem::real_t* lte_table, mfem::real_t &e_min, mfem::real_t &e_max)
    {
      mfem::real_t UKB = 1.380649e-23;
      int nb_comp = plato_get_nb_comp();
      int nb_spec = plato_get_nb_species();
      int nb_temp = plato_get_nb_temp();
      double X_tol = 1e-12;

      mfem::Vector yc(nb_comp), Xc(nb_comp);
      mfem::Vector Xi(nb_spec), Xitol(nb_spec), Xip(nb_spec), Xim(nb_spec), yi(nb_spec),
        Ri(nb_spec), ei(nb_spec), hi(nb_spec), Ji(nb_spec), si(nb_spec), Hi(nb_spec),
        Di(nb_spec),Dij(nb_spec*(nb_spec + 1)/2), di(nb_spec);
      mfem::Vector temp(nb_temp), lambda_int(nb_temp);

      mfem::real_t R, P, nb, e, betaT, alpha, s, H,
        mu, sigma, lambda_trh, lambda_tre, lambda_reactive;

      plato_get_Ri(Ri.GetData());

      int flag = 0;
      for (int i = 0; i < L.nx; ++i)
        {
          const mfem::real_t rho = rho_grid[i];
          for (int j = 0; j < L.ny; ++j)
            {
              yc = 1.0;
              Xc = 1.0;

              const mfem::real_t T = T_grid[j];
              temp = T;

              // Computing LTE composition
              plato_get_eq_composition_mass(&rho, &T, yc.GetData(), yi.GetData(), &flag);
              plato_mass_to_mole_fractions(yi.GetData(), Xi.GetData());

              // Number density and pressure
              R  = Theseus::Kernels::Dot(nb_spec, Ri.GetData(), yi.GetData());
              P  = rho * R * T;
              nb = P / (UKB*T);

              // Energies and entropies per unit-mass
              plato_get_species_energy(temp.GetData(), ei.GetData());
              plato_get_species_entropy(temp.GetData(), &P, si.GetData());
              for (int sp = 0; sp < nb_spec; ++sp)
                {
                  hi[sp] = ei[sp] + Ri[sp]*T;
                }
              for (int sp = 0; sp < nb_spec; ++sp)
                {
                  Hi[sp] = ei[sp] - T*si[sp];
                }
              e = Theseus::Kernels::Dot(nb_spec, yi.GetData(), ei.GetData());
              s = Theseus::Kernels::Dot(nb_spec, yi.GetData(), si.GetData());
              H = Theseus::Kernels::Dot(nb_spec, yi.GetData(), Hi.GetData());

              // Derived thermodynamic properties
              // isothermal compressibility, coeff of thermal expansion
              betaT = plato_get_eq_isoth_comp(&P, &T, Xi.GetData());
              alpha = plato_get_eq_coeff_th_exp(&P, &T, Xi.GetData());

              // Transport properties
              // dynamic viscosity, thermal conductivity, components of thermal conductivity
              plato_get_transp_coeff_comp(&nb, Xi.GetData(), temp.GetData(), &mu, &sigma,
                                          &lambda_trh, &lambda_tre, lambda_int.GetData(), Di.GetData());

              double eps = 1e-5;
              double Tepsp1 = T*(1.0 + eps);
              double Tepsm1 = T*(1.0 - eps);
              // reactive thermal conductivity
              // (NOTE: pass mole fractions with tolerance to procedure solving Stefan-Maxwell's equations)
              plato_get_eq_composition_mole(&P, &Tepsp1, Xc.GetData(), Xip.GetData(), &flag);
              plato_get_eq_composition_mole(&P, &Tepsm1, Xc.GetData(), Xim.GetData(), &flag);
              plato_get_bin_diff_coeff(&nb, &T, &T, Xi.GetData(), Dij.GetData());
              for (int sp = 0; sp < nb_spec; ++sp)
                {
                  di[sp] = -0.5 * (Xip[sp] - Xim[sp])/(eps*T);
                }

              double sum_Xitol = 0.0, X;
              for (int sp = 0; sp < nb_spec; ++sp)
                {
                  X = Xi[sp] + X_tol;
                  Xitol[sp] = X;
                  sum_Xitol += X;
                }

              sum_Xitol = 1.0/sum_Xitol;
              for (int sp = 0; sp < nb_spec; ++sp)
                {
                  Xitol[sp] *= sum_Xitol;
                }

              plato_get_species_diff_flux(&T, &T, &nb, Xitol.GetData(), Dij.GetData(), di.GetData(), Ji.GetData());
              lambda_reactive = Theseus::Kernels::Dot(nb_spec, Ji.GetData(), hi.GetData());

              lte_table[L.property_index(L.H_idx, i, j)]      = H;
              lte_table[L.property_index(L.Hr_idx, i, j)]    = P/(rho*rho);
              lte_table[L.property_index(L.HT_idx, i, j)]    = -s;
              lte_table[L.property_index(L.HrT_idx, i, j)]  = alpha/(rho*rho*betaT);
              lte_table[L.property_index(L.mu_idx, i, j)]     = mu;
              lte_table[L.property_index(L.lambda_idx, i, j)] = lambda_trh + lambda_tre + lambda_int[0] + lambda_reactive;

              if (i == 0 && j == 0)
                {
                  e_min = e;
                  e_max = e;
                }
              else
                {
                  e_min = std::min(e_min, e);
                  e_max = std::max(e_max, e);
                }
            }
        }
    }

    // PLATO must already be initialized for the selected mixture/solver.
    // Grid ownership is retained in data; construction is host-only.
    inline Tables Build(Data &data)
    {
      const Layout layout{data.rho_grid.Size(), data.T_grid.Size()};
      if (layout.nx < 2 || layout.ny < 2)
        {
          throw std::invalid_argument("Helmholtz construction requires at least two samples per axis");
        }
      data.lte_table.SetSize(Layout::num_properties * layout.nx * layout.ny);
      data.inv_table.SetSize(layout.nx * layout.ny);
      data.e_grid.SetSize(layout.ny);
      // Validate dimensions and grids before PLATO/table access.
      LTETable::uniform_grid(layout.ny, 0, 1, data.e_grid);
      Validate(data, layout);
      mfem::real_t minimum_energy;
      mfem::real_t maximum_energy;
      fill_table(layout, data.rho_grid.HostRead(), data.T_grid.HostRead(),
                 data.lte_table.HostWrite(), minimum_energy, maximum_energy);
      // The inverse coordinates must cover the represented forward energy,
      // including cancellation in H-T*H_T and conservative packing. This padding
      // affects guesses only; forward rho/T bounds remain unchanged.
      const auto *forward = data.lte_table.HostRead();
      const int plane_size = layout.nx * layout.ny;
      mfem::real_t rounding_scale = 1;
      for (int j = 0; j < layout.ny; ++j)
        {
          for (int i = 0; i < layout.nx; ++i)
            {
              const int index = layout.property_index(0, i, j);
              const auto h = forward[index];
              const auto tht = data.T_grid[j]*forward[2*plane_size + index];
              const auto energy = h - tht;
              minimum_energy = std::min(minimum_energy, energy);
              maximum_energy = std::max(maximum_energy, energy);
              rounding_scale = std::max(rounding_scale, std::abs(h) + std::abs(tht));
            }
        }
      const auto padding = 64*std::numeric_limits<mfem::real_t>::epsilon()*rounding_scale;
      minimum_energy -= padding;
      maximum_energy += padding;
      LTETable::uniform_grid(layout.ny, minimum_energy, maximum_energy, data.e_grid);
      const auto tables = HostView(data, layout);
      auto *inverse = data.inv_table.HostWrite();
      for (int i = 0; i < layout.nx; ++i)
        {
          const auto density = data.rho_grid[i];
          const auto low_temperature = data.T_grid[0];
          const auto high_temperature = data.T_grid[layout.ny - 1];
          const auto low_energy = HelmholtzEOS::At(tables, density, low_temperature).energy();
          const auto high_energy = HelmholtzEOS::At(tables, density, high_temperature).energy();
          if (!std::isfinite(low_energy) || !std::isfinite(high_energy) || high_energy <= low_energy)
            {
              throw std::invalid_argument("Helmholtz inverse construction needs increasing endpoint energy");
            }
          for (int j = 0; j < layout.ny; ++j)
            {
              const auto energy = data.e_grid[j];
              auto low = low_temperature;
              auto high = high_temperature;
              if (energy <= low_energy)
                {
                  high = low;
                }
              else if (energy >= high_energy)
                {
                  low = high;
                }
              else
                {
                  for (int iteration = 0; iteration < 60; ++iteration)
                    {
                      const auto middle = low + (high - low) / 2;
                      if (HelmholtzEOS::At(tables, density, middle).energy() < energy)
                        {
                          low = middle;
                        }
                      else
                        {
                          high = middle;
                        }
                    }
                }
              // Outside a density column's forward energy range, store the
              // nearest endpoint as a bounded guess, not an extrapolated root.
              inverse[layout.property_index(0, i, j)] = low + (high - low) / 2;
            }
        }
      return HostView(data, layout);
    }
#endif
  }
}
