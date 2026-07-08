// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cassert>
#include <cmath>
#include "GasModel.hpp"

// This file contains helper routines that implement flow physics models
// and methods. Helpers in this module need and use results from the gas
// EOS and Transport models, but do not belong themselves in the EOS or
// Transport constructs.
//
// NOTE: The main reason for existence of these helpers is to centralize
// the routines which may need to be updated when adding new gas models
// like LTE/NLTE. Some routines in this module may be invalid for other
// gas types, but are central to how fluxes are computed.
namespace Theseus {
  namespace Flow {

    MFEM_HOST_DEVICE
    inline void RotateState(const StateLayout layout, const mfem::real_t *nor, Theseus::PointStateViewRW &S)
    {
      int dim = layout.dim;
      if(dim == 1){
        S.set_momentum(layout, 0, S.momentum(layout, 0)*nor[0]);
        return;
      }
  
      mfem::real_t tan1[3];
      mfem::real_t rhoV[3];
      for(int idim = 0;idim < dim;idim++)
        rhoV[idim] = S.momentum(layout, idim);
      Theseus::Kernels::Normal(dim, nor, tan1);
      mfem::real_t rho_u = Theseus::Kernels::Dot(dim, rhoV, nor);
      mfem::real_t rho_v = Theseus::Kernels::Dot(dim, rhoV, tan1);
      
      if (dim == 3)
        {
          mfem::real_t tan2[3];
          Theseus::Kernels::Cross(dim, nor, tan1, tan2);
          rhoV[2] = Theseus::Kernels::Dot(dim, rhoV, tan2);
        }
      rhoV[0] = rho_u;
      rhoV[1] = rho_v;
      for(int idim = 0;idim < dim;idim++){
        S.set_momentum(layout, idim, rhoV[idim]);
      }
    }

    // This routine takes as input a  *face-oriented* state as input and returns
    // p* for the 1D Riemann problem at the face. 
    template<typename StateView, typename GasModelT>
    MFEM_HOST_DEVICE
    inline mfem::real_t slipwall_pstar(const StateView &S, const GasModelT &gasModel)
    {
      const mfem::real_t rho = gasModel.density(S);
      const mfem::real_t c = gasModel.sound_speed(S);
      const mfem::real_t p = gasModel.pressure(S);
      const mfem::real_t v = gasModel.velocity(S, 0);
      const mfem::real_t gamma = gasModel.gamma(S);
      const mfem::real_t gammaP1 = gamma + 1.;
      const mfem::real_t gammaM1 = gamma - 1.;
      const mfem::real_t gammaP1Inverse = 1.0/gammaP1;
      const mfem::real_t gammaM1Inverse = 1.0/gammaM1;
      if (v > 0.0){
        return (p + 0.25*v*gammaP1*rho*
                (v + std::sqrt(v*v + 8.0*gammaP1Inverse*p*(gammaM1*gammaP1Inverse+1.0)/rho)));
      } else {
        return p * std::pow(std::max(1.0 + 0.5*gammaM1*v/c, 0.0001), 2.0*gamma*gammaM1Inverse);
      }
      return 0.0;
    }

    // This interface uses the internal entropy state (Se), and the wall temp (Tw) to get a "wall beta"
    // It *requires specialization* for any gas other than ideal single component gas
    // Ideal Gas: beta = 1/(RTwall)
    // Ideal Mixtures / LTE beta = 1/(Rmix*Twall), where Rmix is depending on the mixture Rmix(Y)
    // NLTE: Potentially this will be OK, but EOS-dependent
    template<typename StateView, typename GasModelT>
    MFEM_HOST_DEVICE
    inline mfem::real_t isothermal_wall_beta(const StateView &Se, mfem::real_t Tw, const GasModelT &gasModel)
    {
      // In gas models where R_gas is not constant (e.g. a mixture), we need to pass the *conserved*
      // state to the gasModel.R_gas function. Since we only have ideal atm with fixed R_gas, I am
      // skipping the unnecessary Entropy2Conservative conversion.  
      return (1.0 / (gasModel.R_gas(Se)*Tw));

      // LTE Gas Model wants this one (because it uses dimensional entropy)
      // return (1.0 / Tw);
    }

    struct TotalConditions {
      mfem::real_t p0;
      mfem::real_t T0;
    };

    struct StaticKinematics {
      mfem::real_t rho;
      mfem::real_t v2;
      mfem::real_t energy;
    };

    template<typename ConservedStateView, typename GasModelT>
    MFEM_HOST_DEVICE
    inline StaticKinematics isentropic_total_to_static(const ConservedStateView &S, const TotalConditions &Ct,
                                                       const GasModelT &gasModel)
    {
      StaticKinematics out{};
      const mfem::real_t p = gasModel.pressure(S);
      const mfem::real_t gamma = gasModel.gamma(S);
      const mfem::real_t cp = gasModel.cp(S);
      const mfem::real_t gm1 = gamma - 1.0;
      const mfem::real_t expo = gm1 / gamma;

      mfem::real_t v2 = 2.0*cp*Ct.T0*(1.0 - std::pow(p/Ct.p0, expo));
      v2 = std::max(0.0, v2);
      const mfem::real_t cpT = cp * Ct.T0 - 0.5*v2;
      out.rho = (gamma / gm1)*p/cpT;
      out.energy = p/gm1 + 0.5*v2*out.rho;
      return out;
    }

    template<typename StateView, typename StateViewRW, typename GasModelT>
    MFEM_HOST_DEVICE
    inline void riemann_invariant_outer_state(const StateView &Si, const StateView &So, StateViewRW &S2, const mfem::real_t *n,
                                              const GasModelT &gasModel)
    {
      const int dim = gasModel.dim();
      const mfem::real_t rho_i = gasModel.density(Si);
      const mfem::real_t rho_o = gasModel.density(So);
      mfem::real_t Vn_i = gasModel.momentum(Si, 0)*n[0];
      mfem::real_t Vn_o = gasModel.momentum(So, 0)*n[0];
      if (dim > 1){
        Vn_i += gasModel.momentum(Si, 1)*n[1];
        Vn_o += gasModel.momentum(So, 1)*n[1];
      }
      if (dim > 2){
        Vn_i += gasModel.momentum(Si, 2)*n[2];
        Vn_o += gasModel.momentum(So, 2)*n[2];
      }

      Vn_i /= rho_i;
      Vn_o /= rho_o;

      const mfem::real_t p_i = gasModel.pressure(Si);
      const mfem::real_t a_i = gasModel.sound_speed(Si);
      const mfem::real_t p_o = gasModel.pressure(So);
      const mfem::real_t a_o = gasModel.sound_speed(So);

      const mfem::real_t gamma = gasModel.gamma(Si);
      const mfem::real_t gm1 = gamma - 1.0;
      const mfem::real_t gm1i = 1.0 / gm1;
      const mfem::real_t gi = 1.0/gamma;

      const bool ext_supersonic_n = (std::abs(Vn_o) >= a_o);

      const mfem::real_t Rm =
        (ext_supersonic_n && Vn_i >= 0.0)
        ? (Vn_i - 2.0 * a_i * gm1i)
        : (Vn_o - 2.0 * a_o * gm1i);      
      const mfem::real_t Rp =
        (ext_supersonic_n && Vn_i < 0.0)
        ? (Vn_o + 2.0 * a_o * gm1i)
        : (Vn_i + 2.0 * a_i * gm1i);

      const mfem::real_t Vn_b = 0.5 * (Rm + Rp);
      const mfem::real_t a_b = 0.25*gm1*(Rp - Rm);

      const mfem::real_t dVn_i = Vn_b - Vn_i;
      const mfem::real_t dVn_o = Vn_b - Vn_o;

      // Density from isentrope anchored on inflow/outflow side
      const auto rho_from_ref = [&](mfem::real_t rho_ref, mfem::real_t p_ref) {
        // rho_b = rho_ref * ( (a_b^2 * rho_ref)/(gamma*p_ref) )^(1/(gamma-1))
        const mfem::real_t factor = (a_b*a_b) * rho_ref / (gamma * p_ref);
        return rho_ref * std::pow(factor, gm1i);
      };

      const bool inflow = (Vn_i < 0.0);
      const mfem::real_t rho_b = inflow ? rho_from_ref(rho_o, p_o) : rho_from_ref(rho_i, p_i);
      S2.set_mass(gasModel.L, rho_b);
      const mfem::real_t p_b = (a_b * a_b) * rho_b * gi;
      mfem::real_t vb[3] = {0., 0., 0.};
      mfem::real_t vb2 = 0.0;
      const mfem::real_t dVn = inflow ? dVn_o : dVn_i;
      for(int idim = 0;idim < dim;idim++){
        const mfem::real_t base = inflow ? gasModel.velocity(So, idim) : gasModel.velocity(Si, idim);
        vb[idim] = base + dVn*n[idim];
        vb2 += (vb[idim]*vb[idim]);
        S2.set_momentum(gasModel.L, idim, rho_b*vb[idim]);
      }
      S2.set_energy(gasModel.L, p_b * gm1i + 0.5 * rho_b * vb2);
    }

    template<typename PrimStateView, typename ConsStateView, typename GasModelT>
    MFEM_HOST_DEVICE inline void PrimitiveToConserved(const PrimStateView &prim, ConsStateView &cons, const GasModelT &gasModel){
      const mfem::real_t rho = prim.mass(gasModel.L);
      const int dim = gasModel.dim();
      // NOTE: This call *should* fail for gas models other than ideal single component
      const mfem::real_t gamma = gasModel.gamma(prim);
      mfem::real_t v2 = prim.velocity(gasModel.L, 0)*prim.velocity(gasModel.L, 0);
      cons.set_mass(gasModel.L, rho);
      cons.set_momentum(gasModel.L, 0, rho*prim.velocity(gasModel.L, 0));
      if (dim > 1){
        cons.set_momentum(gasModel.L, 1, rho*prim.velocity(gasModel.L, 1));
        v2 += prim.velocity(gasModel.L, 1)*prim.velocity(gasModel.L, 1);
      }
      if (dim > 2){
        cons.set_momentum(gasModel.L, 2, rho*prim.velocity(gasModel.L, 2));
        v2 += prim.velocity(gasModel.L, 2)*prim.velocity(gasModel.L,2);
      }
      cons.set_energy(gasModel.L, prim.pressure(gasModel.L) / (gamma-1.) + 0.5 * rho * v2); 
    }
  }
}
