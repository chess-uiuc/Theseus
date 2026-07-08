// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include "Physics.hpp"
#include "GasState.hpp"
#include "LTETable.hpp"

using namespace Theseus::LTETable;

namespace Theseus
{

  // ============================================================================
  // LTEGasEOS: Gas-mixture in local thermodynamic equilibrium
  // ============================================================================
  struct LTEGasEOS
  {

    // ---- helpers on conservative state --------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t R_gas(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S, const LTETables &lteTables) const
    {
      return property_lookup(lteTables.L.R_eq_idx, phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t density(const PhysicsConstants &phys, const StateLayout &L,
                                const StateView &S, const LTETables &lteTables) const
    {
      return S.mass(L); // this is "rho" (mass density)
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t rhoE(const PhysicsConstants &phys, const StateLayout &L,
                             const StateView &S) const
    {
      return S.energy(L);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t momentum_sq(const PhysicsConstants &phys, const StateLayout &L,
                                    const StateView &S) const
    {
      const int dim = L.dim;   // uses state layout
      mfem::real_t m2 = 0;
      for (int d = 0; d < dim; ++d)
        {
          const mfem::real_t m = S.momentum(L,d);
          m2 += m * m;
        }
      return m2;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t kinetic_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                               const StateView &S, const LTETables &lteTables) const
    {
      // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
      const mfem::real_t rho  = density(phys, L, S, lteTables);
      const mfem::real_t m2   = momentum_sq(phys, L, S);
      return 0.5 * m2 / rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                                const StateView &S, const LTETables &lteTables) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return rhoE(phys, L, S) - kinetic_energy_density(phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_from_pressure(const PhysicsConstants &phys, const StateLayout &L,
                                                      const StateView &S, mfem::real_t pressure_target,
                                                      const LTETables &lteTables) const
    {
      mfem::real_t U[Theseus::MAXEQ];
      PointStateViewRW S_dummy(U);
      S_dummy.set_mass(L, S.mass(L));

      for(int idim = 0; idim < L.dim; idim++)
        {
          S_dummy.set_momentum(L, idim, S.momentum(L, idim));
        }
      S_dummy.set_energy(L, S.energy(L));

      // Secant Method to find internal energy that matches a target pressure
      mfem::real_t tol   = 1e-12;
      mfem::real_t denom = 0.0;
      mfem::real_t ie_old = pressure_target * 3; // initial guess for internal energy (CL NOTE : make sure this is inside the table range)
      mfem::real_t ie_new = ie_old*1.01 + tol;

      mfem::real_t ke = kinetic_energy_density(phys, L, S, lteTables);

      mfem::real_t f_old, f_new, ie_update;

      for(int iter = 0; iter < 100; iter++)
        {
          S_dummy.set_energy(L, ie_old+ke);
          f_old = property_lookup(lteTables.L.P_idx, phys, L, S_dummy, lteTables) - pressure_target;

          S_dummy.set_energy(L, ie_new+ke);
          f_new = property_lookup(lteTables.L.P_idx, phys, L, S_dummy, lteTables) - pressure_target;

          denom = f_new - f_old;

          if( Theseus::Kernels::rabs(f_new) < tol || Theseus::Kernels::rabs(denom) < 1e-12)
            {
              break;
            }

          ie_update = ie_new - f_new * (ie_new - ie_old) / denom;

          ie_old = ie_new;
          ie_new = ie_update;
          if(iter == 99){
            MFEM_ABORT("Secant method did not converge in internal_energy_from_pressure");
          }
        }
      return ie_new;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t specific_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                                 const StateView &S, const LTETables &lteTables) const
    {
      // e = (rho*e) / rho
      const mfem::real_t rho  = density(phys, L, S, lteTables);
      const mfem::real_t rhoe = internal_energy_density(phys, L, S, lteTables);
      return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t pressure(const PhysicsConstants &phys, const StateLayout &L,
                                 const StateView &S, const LTETables &lteTables) const
    {
      return property_lookup(lteTables.L.P_idx, phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t gamma(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S, const LTETables &lteTables) const
    {
      return property_lookup(lteTables.L.gamma_eq_idx, phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t temperature(const PhysicsConstants &phys, const StateLayout &L,
                                    const StateView &S, const LTETables &lteTables) const
    {
      return temp_from_internal_energy(phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const PhysicsConstants &phys, const StateLayout  &L,
                                 const StateView &S, const mfem::real_t *grad_rho,
                                 const mfem::real_t *grad_p, mfem::real_t *grad_t,
                                 const LTETables &lteTables) const
    {
      for(int i = 0; i < L.dim; i++){
        grad_t[i] = grad_p[i]; // CL NOTE : we store T_xi in contiguous gradient array for LTE (W=[rho, u, v, w , T])
      }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t sound_speed(const PhysicsConstants &phys, const StateLayout &L,
                                    const StateView &S, const LTETables &lteTables) const
    {
      return property_lookup(lteTables.L.c_idx, phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t cv(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S, const LTETables &lteTables) const
    {
      return property_lookup(lteTables.L.cv_idx, phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t cp(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S, const LTETables &lteTables) const
    {
      return property_lookup(lteTables.L.cp_idx, phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t entropy(const PhysicsConstants &phys, const StateLayout &L,
                                const StateView &S, const LTETables &lteTables) const
    {
      MFEM_ABORT("CL ALERT : Dont have it in plato tables and also not needed");
      return 0.0;
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const PhysicsConstants &phys, const StateLayout &L,
                              const InStateView &S, OutStateView &E,
                              const LTETables &lteTables) const
    {
      const mfem::real_t T    = temperature(phys, L, S, lteTables);
      const mfem::real_t beta = 1/T;
      const mfem::real_t ent_1 = 0.0;

      E.set_mass(L, ent_1);
      int dim = L.dim;
      int num_scalars = L.num_scalars;
      for(int idim = 0;idim < dim;idim++){
        E.set_momentum(L, idim, beta * S.velocity(L, idim));
      }
      E.set_energy(L, -beta);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const PhysicsConstants &phys, const StateLayout &L,
                                          const InStateView &S, const InStateView &dE,
                                          OutStateView &dPrim, const LTETables &lteTables) const
    {

      const mfem::real_t T    = temperature(phys, L, S, lteTables);

      dPrim.set_mass(L, 0.0); // CL NOTE : won't be using density gradient (if W = [rho, u, v, w , T])
      int dim = L.dim;
      for(int i=0; i < dim; i++)
        {
          dPrim.set_momentum(L, i, dE.momentum(L, i)*T + T*S.velocity(L, i)*dE.energy(L));
        }
      dPrim.set_energy(L, T*T*dE.energy(L));
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                     const InStateView &Se, OutStateView &Sc,
                                     const LTETables &lteTables) const
    {
      std::cerr<< " CL ALERT : Not functional in LTE yet "<<std::endl;
    }

    template<typename InStateView, typename OutStateView>
    inline void primitive_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                       const InStateView &prim, OutStateView &cons,
                                       const LTETables &lteTables) const
    {
      const mfem::real_t rho = prim.mass(L);
      const int dim    = L.dim;
      mfem::real_t v2        = 0.0;

      cons.set_mass(L, rho);
      for(int d = 0; d < dim; d++)
        {
          cons.set_momentum(L, d, rho*prim.velocity(L, d));
          v2 += prim.velocity(L,d)*prim.velocity(L,d);
        }
      mfem::real_t rhoe = internal_energy_from_pressure(phys, L, cons, prim.pressure(L), lteTables);
      cons.set_energy(L, rhoe + 0.5 * rho * v2);
    }

    // TODO: Consider whether this is needed/convenient
    // It *can be* nice to have here, but kind of out-of-place
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void velocity(const PhysicsConstants &phys, const StateLayout &L,
                         const StateView &S, mfem::real_t u[3], const LTETables &lteTables) const
    {
      const int dim = L.dim;
      for (int d = 0; d < dim; ++d)
        {
          u[d] = S.velocity(L, d);
        }
      for (int d = dim; d < 3; ++d)
        {
          u[d] = mfem::real_t(0);
        }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t property_lookup(int property_idx, const PhysicsConstants &phys,
                                        const StateLayout &L, const StateView &S,
                                        const LTETables &lteTables) const
    {
      mfem::real_t T = temp_from_internal_energy(phys, L, S, lteTables);
      return biinterp_lte_table(property_idx, phys, L, S, T, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t temp_from_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                                  const StateView &S, const LTETables &lteTables) const
    {
      mfem::real_t rho = density(phys, L, S, lteTables);
      mfem::real_t e = specific_internal_energy(phys, L, S, lteTables);
      mfem::real_t T = biinterp_inverse_table(phys, L, S, lteTables);

      mfem::real_t tol = 1e-12;
      mfem::real_t res = 1;

      int iter = 0;

      while(res > tol)
        {
          mfem::real_t e_guess  = biinterp_lte_table(lteTables.L.e_idx, phys, L, S, T, lteTables);
          mfem::real_t cv = biinterp_lte_table(lteTables.L.cv_idx, phys, L, S, T, lteTables);

          res = (e - e_guess)/cv;

          T = T + res;
          res = Theseus::Kernels::rabs(res)/T;

          iter++;
          if(iter > 100)
            {
#ifdef __CUDA_ARCH__
              printf("Newton method did not converge in temp_from_internal_energy");
              asm("trap;");
#else
              MFEM_ABORT("Newton method did not converge in temp_from_internal_energy");
#endif
            }
        }

      return T;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t biinterp_inverse_table(const PhysicsConstants &phys, const StateLayout &L,
                                               const StateView &S, const LTETables &lteTables) const
    {
      /*
       * Q01--------Q11
       *  |          |
       *  |          |
       *  |          |
       * Q00--------Q10
       */

      // Point rho and e values
      mfem::real_t rho  = density(phys, L, S, lteTables);
      mfem::real_t e = specific_internal_energy(phys, L, S, lteTables);

      // Get the lower and upper x and y indices of the cell
      int l_x = hunt(lteTables.tables.rho_grid, lteTables.L.nx, rho, 0);
      int l_y = hunt(lteTables.tables.e_grid, lteTables.L.ny, e, 0);
      int u_x = l_x + 1  , u_y = l_y + 1;

      if(l_x < 0 || u_x >= lteTables.L.nx || l_y < 0 || u_y >= lteTables.L.ny)
        {
#ifdef __CUDA_ARCH__
          printf(" CL ALERT : Out of bounds in LTE table lookup! \n");
          printf("l_x : %d l_y : %d \n", l_x, l_y);
          printf("rho : %e < %e < %e \n", lteTables.tables.rho_grid[0],
                 rho, lteTables.tables.rho_grid[lteTables.L.nx-1]);
          printf("e : %e < %e < %e \n", lteTables.tables.e_grid[0],
                 e, lteTables.tables.e_grid[lteTables.L.ny-1]);
          asm("trap;");
#else
          std::cout << " CL ALERT : Out of bounds in LTE table lookup! "<<std::endl;
          std::cout << "l_x : " << l_x << "l_y : " << l_y << std::endl;
          std::cout << "rho : " << lteTables.tables.rho_grid[0] << " < " << rho
                    << " < " << lteTables.tables.rho_grid[lteTables.L.nx-1] << std::endl;
          std::cout << "e : " << lteTables.tables.e_grid[0] << " < "
                    << e << " < " << lteTables.tables.e_grid[lteTables.L.ny-1] << std::endl;
          std::exit(1);
#endif
        }

      // Get the lower and upper x and y coordinates of the cell
      mfem::real_t rho_l  = lteTables.tables.rho_grid[l_x] , rho_u = lteTables.tables.rho_grid[u_x];
      mfem::real_t e_l    = lteTables.tables.e_grid[l_y]   , e_u   = lteTables.tables.e_grid[u_y];

      // Get the corner property values
      mfem::real_t Q00 = lteTables.tables.inv_table[lteTables.L.property_index(0, l_x, l_y)];
      mfem::real_t Q01 = lteTables.tables.inv_table[lteTables.L.property_index(0, l_x, u_y)];
      mfem::real_t Q10 = lteTables.tables.inv_table[lteTables.L.property_index(0, u_x, l_y)];
      mfem::real_t Q11 = lteTables.tables.inv_table[lteTables.L.property_index(0, u_x, u_y)];


      mfem::real_t wx = (rho  - rho_l)  / (rho_u - rho_l);
      mfem::real_t wy = (e - e_l) / (e_u - e_l);


      // Clamp to [0, 1]
      wx = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wx));
      wy = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wy));


      return Q00 * ((1 - wx) * (1 - wy)) + Q01 * ((1 - wx) * wy) +
        Q10 * (wx * (1 - wy)) + Q11 * (wx * wy);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t biinterp_lte_table(int property_idx, const PhysicsConstants &phys,
                                           const StateLayout &L, const StateView &S,
                                           const mfem::real_t T,
                                           const LTETables &lteTables) const
    {
      /*
       * Q01--------Q11
       *  |          |
       *  |          |
       *  |          |
       * Q00--------Q10
       */

      mfem::real_t rho  = density(phys, L, S, lteTables);

      // Get the lower and upper x and y indices of the cell
      int l_x = hunt(lteTables.tables.rho_grid, lteTables.L.nx, rho, 0);
      int l_y = hunt(lteTables.tables.T_grid, lteTables.L.ny, T, 0);
      int u_x = l_x + 1  , u_y = l_y + 1;

      // Get the lower and upper x and y coordinates of the cell
      mfem::real_t rho_l  = lteTables.tables.rho_grid[l_x] , rho_u = lteTables.tables.rho_grid[u_x];
      mfem::real_t T_l    = lteTables.tables.T_grid[l_y]   , T_u   = lteTables.tables.T_grid[u_y];

      // Get the corner property values
      mfem::real_t Q00 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, l_x, l_y)];
      mfem::real_t Q01 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, l_x, u_y)];
      mfem::real_t Q10 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, u_x, l_y)];
      mfem::real_t Q11 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, u_x, u_y)];

      mfem::real_t wx = (rho  - rho_l)  / (rho_u - rho_l);
      mfem::real_t wy = (T - T_l) / (T_u - T_l);

      // Clamp to [0, 1]
      wx = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wx));
      wy = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wy));


      return Q00 * ((1 - wx) * (1 - wy)) + Q01 * ((1 - wx) * wy) +
        Q10 * (wx * (1 - wy)) + Q11 * (wx * wy);
    }
  };
}
