// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "Physics.hpp"
#include "GasState.hpp"
#include "theseus_kernels.hpp"
#ifdef USE_PLATO
#include "plato_Cpp_library_interface.h"
#include "TheseusConfig.hpp"
#include <filesystem>
#endif

namespace Theseus
{
  namespace LTETable {

    struct Layout
    {
      // LTE table specific quantities
      int nx, ny;                // dimensions of LTE table
      int num_properties = 9;    // CL NOTE : change the num_properties if more are being stored

      // Property indices in the LTE table
      int P_idx        = 0;  // pressure
      int e_idx        = 1;  // internal energy
      int cv_idx       = 2;  // specific heat at constant volume
      int cp_idx       = 3;  // specific heat at constant pressure
      int R_eq_idx     = 4;  // Mixture gas constant
      int gamma_eq_idx = 5;  // gamma-eq
      int c_idx        = 6;  // equilibrium sound-speed
      int mu_idx       = 7;  // shear viscosity
      int lambda_idx   = 8;  // thermal conductivity

      void setup(int nx_, int ny_)
      {
        nx = nx_;
        ny = ny_;
      }

      // Flat index of properties in the flattened 3D LTE Table
      MFEM_HOST_DEVICE inline int property_index(int property, int ind_x, int ind_y) const
      {
        int index;
        // TODO: I don't think assert is allowed in device code
        assert(property < num_properties);
        index = property*ny*nx + ind_y*nx + ind_x;
        return index;
      }

    };

    struct Data
    {
      mfem::Vector lte_table;
      mfem::Vector inv_table;
      mfem::Vector rho_grid;
      mfem::Vector T_grid;
      mfem::Vector e_grid;
    };

    struct View
    {
      const mfem::real_t *lte_table;
      const mfem::real_t *inv_table;
      const mfem::real_t *rho_grid;
      const mfem::real_t *T_grid;
      const mfem::real_t *e_grid;
    };


    struct LTETables {
      MFEM_HOST_DEVICE LTETables() = default;

      LTETables(int nx_, int ny_ = 0)
      { L.setup(nx_, ny_); }

      LTETable::Layout L;
      LTETable::View tables;
    };

    inline void uniform_grid(int N, mfem::real_t min, mfem::real_t max, mfem::Vector &grid)
    {
      grid.SetSize(N);
      double step = (max - min) / (N - 1);
      for (int i = 0; i < N; ++i)
        {
          grid[i] = min + i * step;
        }
    }

    inline void log_grid(int N, mfem::real_t min, mfem::real_t max, mfem::Vector &grid)
    {
      grid.SetSize(N);
      double step = std::log10(max/min) / (N-1);
      for (int i = 0;i < N; ++i)
        {
          grid[i] = min * std::pow(10, i * step);
        }
    }

    inline void fill_table(const LTETable::Layout &L, const mfem::real_t* rho_grid, const mfem::real_t* T_grid,
                           mfem::real_t* lte_table, mfem::real_t &e_min, mfem::real_t &e_max)
    {

      mfem::real_t UKB = 1.380649e-23;
#ifdef USE_PLATO
      int nb_comp = plato_get_nb_comp();
      int nb_spec = plato_get_nb_species();
      int nb_temp = plato_get_nb_temp();
      double X_tol = 1e-12;

      mfem::Vector yc(nb_comp), Xc(nb_comp);
      mfem::Vector Xi(nb_spec), Xitol(nb_spec), Xip(nb_spec), Xim(nb_spec), yi(nb_spec),
        Ri(nb_spec), ei(nb_spec), hi(nb_spec), Ji(nb_spec),
        Di(nb_spec),Dij(nb_spec*(nb_spec + 1)/2), di(nb_spec);
      mfem::Vector temp(nb_temp), lambda_int(nb_temp);

      mfem::real_t R, P, nb, e, betaT, alpha, cv, gam, cp, c,
        mu, sigma, lambda_trh, lambda_tre, lambda_reactive;

      plato_get_Ri(Ri.GetData());

      int flag = 0;
      for(int i=0; i < L.nx; i++)
        {
          const mfem::real_t rho = rho_grid[i];
          for(int j=0; j < L.ny; j++)
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

              // Energies and enthalpy per unit-mass
              plato_get_species_energy(temp.GetData(), ei.GetData());
              for(int sp=0; sp < nb_spec; sp++) hi[sp] = ei[sp] + Ri[sp]*T;
              e = Theseus::Kernels::Dot(nb_spec, yi.GetData(), ei.GetData());

              // Derived thermodynamic properties
              // isothermal compressibility, coeff of thermal expansion, cv, cp, gamma, sound_speed at equilibrium
              betaT = plato_get_eq_isoth_comp(&P, &T, Xi.GetData());
              alpha = plato_get_eq_coeff_th_exp(&P, &T, Xi.GetData());
              cv    = plato_get_eq_cv(&rho, &T, yi.GetData());
              gam   = 1.0 + alpha*alpha*T/(rho*betaT*cv);
              cp    = gam*cv;
              c     = std::sqrt(gam*P/rho);

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
              for(int sp=0; sp < nb_spec; sp++) di[sp] = -0.5 * (Xip[sp] - Xim[sp])/(eps*T);

              double sum_Xitol = 0.0, X;
              for(int sp=0; sp < nb_spec; sp++)
                {
                  X = Xi[sp] + X_tol;
                  Xitol[sp] = X;
                  sum_Xitol += X;
                }

              sum_Xitol = 1.0/sum_Xitol;
              for(int sp=0; sp < nb_spec; sp++) Xitol[sp] *= sum_Xitol;

              plato_get_species_diff_flux(&T, &T, &nb, Xitol.GetData(), Dij.GetData(), di.GetData(), Ji.GetData());
              lambda_reactive = Theseus::Kernels::Dot(nb_spec, Ji.GetData(), hi.GetData());

              lte_table[L.property_index(L.P_idx, i, j)] = P;
              lte_table[L.property_index(L.e_idx, i, j)] = e;
              lte_table[L.property_index(L.cv_idx, i, j)] = cv;
              lte_table[L.property_index(L.cp_idx, i, j)] = cp;
              lte_table[L.property_index(L.R_eq_idx, i, j)] = R;
              lte_table[L.property_index(L.gamma_eq_idx, i, j)] = gam;
              lte_table[L.property_index(L.c_idx, i, j)] = c;
              lte_table[L.property_index(L.mu_idx, i, j)] = mu;
              lte_table[L.property_index(L.lambda_idx, i, j)] = lambda_trh + lambda_tre + lambda_int[0] + lambda_reactive;

              if(i+j == 0)
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
#endif
    }

    inline void fill_inv_table(const LTETable::Layout &L, const mfem::real_t* rho_grid, const mfem::real_t* e_grid,
                               const mfem::real_t* T_grid, mfem::real_t* inv_table)
    {
#ifdef USE_PLATO
      int nb_comp = plato_get_nb_comp();
      int nb_spec = plato_get_nb_species();

      mfem::Vector yc(nb_comp);
      mfem::Vector yi(nb_spec), ei(nb_spec);
      mfem::Vector temp(1);

      mfem::real_t rho, e0, e, res, cv;
      mfem::real_t tol = 1e-8;

      int flag = 0;
      mfem::real_t T = T_grid[0];
      for(int i = 0; i < L.nx; i++)
        {
          rho = rho_grid[i];
          T = T_grid[0];
          for(int j = 0; j < L.ny; j++)
            {
              e0  = e_grid[j];
              res = 1.0;
              yc = 1.0;
              int it = 0;
              // Newton iteration to find T such that internal energy at (rho, T) matches e0
              while(res > tol)
                {
                  temp = T;
                  plato_get_eq_composition_mass(&rho, &T, yc.GetData(), yi.GetData(), &flag);
                  plato_get_species_energy(temp.GetData(), ei.GetData());
                  e = Theseus::Kernels::Dot(nb_spec, yi.GetData(), ei.GetData());
                  cv = plato_get_eq_cv(&rho, &T, yi.GetData());

                  res = -(e - e0)/cv;
                  T += res;
                  it++;
                  res = std::abs(res)/T;

                  if(it > 100)
                    {
                      MFEM_ABORT("Maximum number of iterations reached in fill_inv_table");
                    }
                  if(T<0.0)
                    {
                      MFEM_ABORT("Negative temperature encountered in fill_inv_table");
                    }
                }
              inv_table[L.property_index(0 , i, j)] = T;
            }
        }
#endif
    }

    MFEM_HOST_DEVICE inline
    int hunt(const mfem::real_t *arr, int n, mfem::real_t x, int ind_lo)
    {
      int ind_hi, ind_mid;
      int incr = 1;
      bool ascend = (arr[n-1] >= arr[0]);

      if (ind_lo < 0 || ind_lo >= n)
        {
          ind_lo = -1;
          ind_hi =  n;
        }
      else
        {
          // Right or Left Hunt
          if ( (x >= arr[ind_lo]) == ascend)
            {
              // Hunt right
              if (ind_lo == n-1) return ind_lo;
              ind_hi = ind_lo + incr;

              while (ind_hi < n && ((x >= arr[ind_hi]) == ascend))
                {
                  ind_lo = ind_hi;
                  incr *= 2;
                  ind_hi = ind_lo + incr;
                  if (ind_hi > n-1)
                    {
                      ind_hi = n;
                      break;
                    }
                }
            }
          // Hunt left
          else
            {
              if (ind_lo == 0)
                {
                  ind_lo = -1;
                  return ind_lo;
                }
              ind_hi = ind_lo;
              ind_lo = ind_lo-1;
              while (ind_lo >= 0 && ((x < arr[ind_lo]) == ascend))
                {
                  ind_hi = ind_lo;
                  incr *= 2;
                  if (incr >= ind_hi)
                    {
                      ind_lo = -1;
                      break;
                    }
                  else ind_lo = ind_hi - incr;
                }
            }
        }

      // Binary Search in the estimated bracket
      while(ind_hi - ind_lo != 1)
        {
          ind_mid = ind_lo + (ind_hi - ind_lo)/2;
          if( (x >= arr[ind_mid]) == ascend)
            {
              ind_lo = ind_mid;
            }
          else
            {
              ind_hi = ind_mid;
            }
        }

      if(x == arr[n-1]) ind_lo = n-2;
      if(x == arr[0]) ind_lo = 0;
      return ind_lo;
    }

#ifdef USE_PLATO
    inline int check_plato_database_path(const std::string &path)
    {
      std::filesystem::path thermo_db_path(path);

      if (!std::filesystem::exists(thermo_db_path))
        {
          std::cerr << "Plato database path does not exist: " << path << std::endl;
          return 1;
        }

      if (!std::filesystem::is_directory(thermo_db_path))
        {
          std::cerr << "Plato database path is not a directory: " << path << std::endl;
          return 1;
        }
      std::filesystem::path mix_path(path+"/mixture");
      std::filesystem::path thermo_path(path+"/thermo");
      if(!std::filesystem::is_directory(mix_path) || !std::filesystem::is_directory(thermo_path))
        {
          std::cerr << "Plato database missing mixture or thermo database." << std::endl;
          return 1;
        }

      return 0;
    }
#endif
  }
}
