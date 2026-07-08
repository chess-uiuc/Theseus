// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include "Physics.hpp"
#include "GasState.hpp"

namespace Theseus
{

  // ============================================================================
  // EOS: Ideal single-species gas using PhysicsConstants
  // ============================================================================
  struct IdealSingleGasEOS
  {
    // ---- helpers on conservative state --------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t R_gas(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return phys.R_gas;
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t density(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
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
                                         const StateView &S) const
    {
      // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
      const mfem::real_t rho  = density(phys, L, S);
      const mfem::real_t m2   = momentum_sq(phys, L, S);
      return 0.5 * m2 / rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                          const StateView &S) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return rhoE(phys, L, S) - kinetic_energy_density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_from_pressure(const PhysicsConstants &phys, const StateLayout &L,
                                                const StateView &S, mfem::real_t pressure) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return pressure / (phys.gamma - 1.0);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t specific_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                           const StateView &S) const
    {
      // e = (rho*e) / rho
      const mfem::real_t rho  = density(phys, L, S);
      const mfem::real_t rhoe = internal_energy_density(phys, L, S);
      return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t pressure(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S) const
    {
      // p = (gamma - 1) * (rho*E - 0.5*|rho*u|^2 / rho)
      const mfem::real_t rhoe = internal_energy_density(phys, L, S);
      return phys.gammaM1 * rhoe;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t gamma(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return phys.gamma;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t temperature(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      // p = rho*R*T  =>  T = p / (rho*R)
      const mfem::real_t rho = density(phys, L, S);
      const mfem::real_t p   = pressure(phys, L, S);
      return p / (rho * phys.R_gas);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const PhysicsConstants &phys, const StateLayout  &L,
                                 const StateView &S, const mfem::real_t *grad_rho,
                                 const mfem::real_t *grad_p, mfem::real_t *grad_t) const
    {
      const int dim = L.dim;
      const mfem::real_t rho = density(phys, L, S);
      const mfem::real_t pressor = pressure(phys, L, S)/rho;
      const mfem::real_t cv = cp(phys, L, S)/phys.gamma;
      const mfem::real_t fac = phys.gammaM1Inverse/(cv*rho);
      for(int i = 0; i < dim; i++){
        grad_t[i] = fac*(grad_p[i] - pressor*grad_rho[i]);
      }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t sound_speed(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      // a^2 = gamma * p / rho
      const mfem::real_t rho = density(phys, L, S);
      const mfem::real_t p   = pressure(phys, L, S);
      return std::sqrt(phys.gamma * p / rho);
    }
    
    // cp is constant for ideal gas
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t cp(const PhysicsConstants &phys, const StateLayout &L,
                     const StateView & /*S*/) const
    {
      return phys.cp;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t entropy(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
    {
      const mfem::real_t p = pressure(phys, L, S);
      const mfem::real_t gamma = phys.gamma;
      // TODO: Augment for correct treatment of passive scalars
      return std::log(p) - gamma * std::log(S.mass(L));
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const PhysicsConstants &phys, const StateLayout &L,
                              const InStateView &S, OutStateView &E) const
    {
      const mfem::real_t p = pressure(phys, L, S);
      const mfem::real_t gamma = phys.gamma;
      const mfem::real_t rho = S.mass(L);
      const mfem::real_t s = std::log(p) - gamma*std::log(rho);
      const mfem::real_t beta = rho / p;
      const mfem::real_t v2o2 = kinetic_energy_density(phys, L, S) / rho;
      const mfem::real_t s_rho = (gamma - s)/(gamma - 1) - beta*v2o2;

      E.set_mass(L, s_rho);
      int dim = L.dim;
      int num_scalars = L.num_scalars;
      for(int idim = 0;idim < dim;idim++){
        E.set_momentum(L, idim, beta * S.velocity(L, idim));
      }
      E.set_energy(L, -beta);
      // TODO: Update for correct treatment of passive scalars (depends on ES approach)
      // - Here we should probably set the entropy state to scalar_state / density
      // - If we do that, we need to modify the mass component of the entropy state
      // - Making this fix will make the sensor function sensitive to the scalars
      // - If we need to recover CV from this, lax scalar treatment is a nogo
      for(int iscalar = 0;iscalar < num_scalars;iscalar++){
        E.set_scalar(L, iscalar, 0.0);
      }
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const PhysicsConstants &phys, const StateLayout &L,
                                          const InStateView &S, const InStateView &dE,
                                          OutStateView &dPrim) const
    {

      const mfem::real_t ke = kinetic_energy_density(phys, L, S);
      const mfem::real_t p = pressure(phys, L, S);
      const mfem::real_t rho = S.mass(L);
      const mfem::real_t rhoE = S.energy(L);
      const mfem::real_t ie = internal_energy_density(phys, L, S);

      int dim = L.dim;
      int num_scalars = L.num_scalars;

      mfem::real_t drho = 0.0;
      for(int idim = 0; idim < dim; idim++){
        dPrim.set_momentum(L, idim, p/rho * (dE.momentum(L, idim) + S.velocity(L, idim)*dE.energy(L)));
        drho += S.momentum(L, idim)*dPrim.momentum(L, idim);
      }
      drho = rho*dE.mass(L) - dE.energy(L)*(ke - ie) + rho*drho/p;
      dPrim.set_mass(L, drho);
      dPrim.set_energy(L, p/rho * (dPrim.mass(L) + p*dE.energy(L)));
      for(int isp = 0; isp < num_scalars; isp++){
        dPrim.set_scalar(L, isp, 0.0); // just a placeholder for now
      }
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                     const InStateView &Se, OutStateView &Sc) const
    {
      int dim = L.dim;
      const mfem::real_t beta = -Se.energy(L);
      mfem::real_t k = 0.0;
      mfem::real_t vel[3];
      for(int idim = 0;idim < dim;idim++){
        vel[idim] = Se.momentum(L, idim)/beta;
        k += vel[idim]*vel[idim];
      }
      const mfem::real_t gamma = phys.gamma;
      const mfem::real_t s = gamma - (Se.mass(L) + 0.5*k*beta)*(gamma - 1.);
      const mfem::real_t rho = std::pow(std::exp(-s)/beta, 1.0/(gamma - 1));
      Sc.set_mass(L, rho);
      Sc.set_energy(L, rho*(1.0/(beta*(gamma-1.)) + 0.5*k));
      for(int idim = 0;idim < dim;idim++){
        Sc.set_momentum(L, idim, rho*vel[idim]);
      }
    }

    template<typename InStateView, typename OutStateView>
    inline void primitive_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                       const InStateView &prim, OutStateView &cons) const
    {
      const mfem::real_t rho   = prim.mass(L);
      const int dim    = L.dim;
      mfem::real_t v2        = 0.0;

      cons.set_mass(L, rho);
      for(int d = 0; d < dim; d++)
      {
        cons.set_momentum(L, d, rho*prim.velocity(L, d));
        v2 += prim.velocity(L,d)*prim.velocity(L,d);
      }
      mfem::real_t rhoe = prim.pressure(L) / (phys.gamma-1.);
      cons.set_energy(L, rhoe + 0.5 * rho * v2);
    }

    // TODO: Consider whether this is needed/convenient
    // It *can be* nice to have here, but kind of out-of-place
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void velocity(const PhysicsConstants &phys, const StateLayout &L,
                         const StateView &S, mfem::real_t u[3]) const
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
  };
}
