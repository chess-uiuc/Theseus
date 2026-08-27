// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"

namespace Theseus
{
    template<typename GasModelT>
    MFEM_HOST_DEVICE
    inline void Roe_dissipation(const GasModelT &gasModel,
                              const PointStateView &S1,
                              const PointStateView &S2,
                              const mfem::real_t *nor,
                              mfem::real_t *diss)
    {
      const int dim = gasModel.dim();
      const int neq = gasModel.num_equations();
      const int mass_eq = gasModel.L.eq_mass;
      const int mom0_eq = gasModel.L.eq_mom0;
      const int ener_eq = gasModel.L.eq_energy;

      mfem::real_t vn1 = 0.0;
      mfem::real_t vn2 = 0.0;
      mfem::real_t vn_roe = 0.0;
      mfem::real_t nor_mag = 0.0;

      const mfem::real_t rho1 = gasModel.density(S1);
      const mfem::real_t rho2 = gasModel.density(S2);
      const mfem::real_t H1 = (gasModel.energy(S1) + gasModel.pressure(S1))/rho1;
      const mfem::real_t H2 = (gasModel.energy(S2) + gasModel.pressure(S2))/rho2;

      // Compute Roe-averaged state
      mfem::real_t roe_state[Theseus::MAXEQ] = {0.,0.,0.,0.,0.};
      const mfem::real_t ratio = Theseus::Kernels::rsqrt(rho2/rho1);
      const mfem::real_t a = 1.0/(ratio + 1.0);
      const mfem::real_t b = a*ratio;

      roe_state[mass_eq] = ratio*rho1;
      roe_state[ener_eq] = a*H1 + b*H2;

      mfem::real_t del_mom[3] = {0,0,0};
      mfem::real_t ke = 0.0;
      for (int d = 0; d < dim; d++)
      {
        nor_mag += nor[d]*nor[d];
        const mfem::real_t v1 = gasModel.velocity(S1, d);
        const mfem::real_t v2 = gasModel.velocity(S2, d);
        del_mom[d] = v2 - v1;
        roe_state[mom0_eq + d] = a*v1 + b*v2;
        ke += roe_state[mom0_eq + d]*roe_state[mom0_eq + d];
        vn1 += v1 * nor[d];
        vn2 += v2 * nor[d];
        vn_roe  += roe_state[mom0_eq + d] * nor[d];
      }
      nor_mag = Theseus::Kernels::rsqrt(nor_mag);
      mfem::real_t inv_nor_mag = 1.0/nor_mag;
      vn1 *= inv_nor_mag;
      vn2 *= inv_nor_mag;
      vn_roe *= inv_nor_mag;
      mfem::real_t c_roe = Theseus::Kernels::rsqrt((gasModel.gamma(S1) - 1)*(roe_state[ener_eq] - 0.5*ke));

      // Harten-Hyman entropy fix for the Roe flux
      mfem::real_t eig[3] = {0.,Kernels::rabs(vn_roe),0.}; // [u-c, u, u+c]

      mfem::real_t c1 = gasModel.sound_speed(S1);
      mfem::real_t c2 = gasModel.sound_speed(S2);
      // Left Acoustic Wave vn - c
      {
        mfem::real_t eig_roe = vn_roe - c_roe;
        mfem::real_t eps = Kernels::rmax(0.0, Kernels::rmax(eig_roe - (vn1 - c1), (vn2 - c2) - eig_roe));
        if(Kernels::rabs(eig_roe) < eps)
        {
          eig[0] = 0.5*(eps + (eig_roe*eig_roe)/eps);
        }
        else
        {
          eig[0] = Kernels::rabs(eig_roe);
        }
      }
      // Right Acoustic Wave vn + c
      {
        mfem::real_t eig_roe = vn_roe + c_roe;
        mfem::real_t eps = Kernels::rmax(0.0, Kernels::rmax(eig_roe - (vn1 + c1), (vn2 + c2) - eig_roe));
        if(Kernels::rabs(eig_roe) < eps)
        {
          eig[2] = 0.5*(eps + (eig_roe*eig_roe)/eps);
        }
        else
        {
          eig[2] = Kernels::rabs(eig_roe);
        }
      }

      mfem::real_t del_rho = rho2 - rho1;
      mfem::real_t del_vn  = vn2 - vn1;
      mfem::real_t del_p   = gasModel.pressure(S2) - gasModel.pressure(S1);
      mfem::real_t ov_csq = 1.0/(c_roe*c_roe);
      mfem::real_t pov_csq = del_p * ov_csq;
      mfem::real_t rhoc_del_vn = roe_state[mass_eq] * del_vn * c_roe * ov_csq;

      // Wave strengths multiplied by eigenvalues
      mfem::real_t alpha_Vn     = eig[1]*(del_rho - pov_csq);
      mfem::real_t alpha_Vn_rho = eig[1]*roe_state[mass_eq];
      mfem::real_t alpha_left   = 0.5*(eig[0]*(pov_csq - rhoc_del_vn));
      mfem::real_t alpha_right  = 0.5*(eig[2]*(pov_csq + rhoc_del_vn));

      // Dissipation terms
      diss[mass_eq] = 0.5*(alpha_Vn + alpha_left + alpha_right);
      diss[ener_eq] = 0.0;
      for (int d = 0; d < dim; d++)
      {
        diss[mom0_eq + d] = 0.5*(alpha_Vn*roe_state[mom0_eq + d] + alpha_Vn_rho*(del_mom[d] - nor[d]*inv_nor_mag*del_vn));
        diss[mom0_eq + d] += 0.5*(alpha_left*(roe_state[mom0_eq + d] - nor[d]*inv_nor_mag*c_roe) + alpha_right*(roe_state[mom0_eq + d] + nor[d]*inv_nor_mag*c_roe));
        diss[ener_eq] += 0.5*alpha_Vn_rho*(roe_state[mom0_eq + d]*del_mom[d]);
      }
      diss[ener_eq] -= 0.5*alpha_Vn_rho*vn_roe*del_vn;
      diss[ener_eq] += 0.5*alpha_Vn*ke + 0.5*alpha_left*(roe_state[ener_eq]-vn_roe*c_roe) + 0.5*alpha_right*(roe_state[ener_eq]+vn_roe*c_roe);

      for(int eq = 0; eq < neq; eq++)
      {
        diss[eq] *= nor_mag;
      }
    };
}
