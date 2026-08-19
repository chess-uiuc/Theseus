// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "NavierStokesFlux.hpp"
#include "theseus_kernels.hpp"

namespace Theseus
{
  namespace RoeFlux
  {
    // The split-form volume operator only needs a consistent two-point flux and
    // a characteristic-speed estimate.  Roe upwinding is applied at faces.
    template<typename GasT>
    MFEM_HOST_DEVICE inline static mfem::real_t
    ComputeVolumeFluxKernel(const GasT &gas,
                            const mfem::real_t *q1,
                            const mfem::real_t *q2,
                            const mfem::real_t *met1,
                            const mfem::real_t *met2,
                            mfem::real_t *flux)
    {
      const int dim = gas.dim();
      const int neq = gas.num_equations();
      mfem::real_t met[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t f1[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t f2[Theseus::MAXEQ][Theseus::MAXDIM];
      Kernels::ComputeMeanVec(met1, met2, met, dim);
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, q1, f1);
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, q2, f2);

      for (int eq = 0; eq < neq; ++eq)
        {
          flux[eq] = 0.0;
          for (int d = 0; d < dim; ++d)
            {
              flux[eq] += 0.5 * (f1[eq][d] + f2[eq][d]) * met[d];
            }
        }

      PointStateView S1{q1};
      PointStateView S2{q2};
      mfem::real_t un1 = 0.0;
      mfem::real_t un2 = 0.0;
      mfem::real_t met_mag2 = 0.0;
      for (int d = 0; d < dim; ++d)
        {
          un1 += gas.velocity(S1, d) * met[d];
          un2 += gas.velocity(S2, d) * met[d];
          met_mag2 += met[d] * met[d];
        }
      const mfem::real_t met_mag = Kernels::rsqrt(met_mag2);
      return Kernels::rmax(Kernels::rabs(un1) + gas.sound_speed(S1) * met_mag,
                           Kernels::rabs(un2) + gas.sound_speed(S2) * met_mag);
    }

    // Classical Roe flux for a calorically perfect gas.  The decomposition is
    // written in normal/tangential-vector form so the same kernel works in 1-D,
    // 2-D, and 3-D and for non-unit MFEM face normals.
    template<typename GasModelT>
    MFEM_HOST_DEVICE inline static mfem::real_t
    ComputeFaceFluxKernel(const GasModelT &gas,
                          const mfem::real_t *qL,
                          const mfem::real_t *qR,
                          const mfem::real_t *nor,
                          mfem::real_t *flux)
    {
      const int dim = gas.dim();
      const int neq = gas.num_equations();
      const int eq_mass = gas.L.eq_mass;
      const int eq_mom0 = gas.L.eq_mom0;
      const int eq_energy = gas.L.eq_energy;
      PointStateView SL{qL};
      PointStateView SR{qR};

      mfem::real_t fL[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t fR[Theseus::MAXEQ][Theseus::MAXDIM];
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, qL, fL);
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, qR, fR);

      mfem::real_t nor_mag2 = 0.0;
      for (int d = 0; d < dim; ++d) { nor_mag2 += nor[d] * nor[d]; }
      const mfem::real_t nor_mag = Kernels::rsqrt(nor_mag2);
      const mfem::real_t inv_nor_mag = 1.0 / nor_mag;

      const mfem::real_t rhoL = gas.density(SL);
      const mfem::real_t rhoR = gas.density(SR);
      const mfem::real_t pL = gas.pressure(SL);
      const mfem::real_t pR = gas.pressure(SR);
      const mfem::real_t rootL = std::sqrt(rhoL);
      const mfem::real_t rootR = std::sqrt(rhoR);
      const mfem::real_t root_sum_inv = 1.0 / (rootL + rootR);
      const mfem::real_t rho_roe = rootL * rootR;
      const mfem::real_t HL = (gas.energy(SL) + pL) / rhoL;
      const mfem::real_t HR = (gas.energy(SR) + pR) / rhoR;
      const mfem::real_t H = (rootL * HL + rootR * HR) * root_sum_inv;

      mfem::real_t u[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t du[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t du_t[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t nhat[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t u2 = 0.0;
      mfem::real_t un = 0.0;
      mfem::real_t dun = 0.0;
      for (int d = 0; d < dim; ++d)
        {
          const mfem::real_t uL = gas.velocity(SL, d);
          const mfem::real_t uR = gas.velocity(SR, d);
          nhat[d] = nor[d] * inv_nor_mag;
          u[d] = (rootL * uL + rootR * uR) * root_sum_inv;
          du[d] = uR - uL;
          u2 += u[d] * u[d];
          un += u[d] * nhat[d];
          dun += du[d] * nhat[d];
        }

      const mfem::real_t gamma = gas.gamma(SL);
      const mfem::real_t a2 = (gamma - 1.0) * (H - 0.5 * u2);
      const mfem::real_t a = std::sqrt(a2);
      const mfem::real_t dp = pR - pL;
      const mfem::real_t drho = rhoR - rhoL;
      const mfem::real_t alpha_minus = 0.5 * (dp - rho_roe * a * dun) / a2;
      const mfem::real_t alpha_plus  = 0.5 * (dp + rho_roe * a * dun) / a2;
      const mfem::real_t alpha_zero  = drho - dp / a2;
      const mfem::real_t lambda_minus = Kernels::rabs(un - a) * nor_mag;
      const mfem::real_t lambda_zero  = Kernels::rabs(un) * nor_mag;
      const mfem::real_t lambda_plus  = Kernels::rabs(un + a) * nor_mag;

      for (int d = 0; d < dim; ++d) { du_t[d] = du[d] - dun * nhat[d]; }

      const mfem::real_t dmass = lambda_minus * alpha_minus
                                + lambda_zero * alpha_zero
                                + lambda_plus * alpha_plus;
      mfem::real_t diss[Theseus::MAXEQ] = {0.0};
      diss[eq_mass] = dmass;
      for (int d = 0; d < dim; ++d)
        {
          diss[eq_mom0 + d] =
            lambda_minus * alpha_minus * (u[d] - a * nhat[d])
            + lambda_zero * (alpha_zero * u[d] + rho_roe * du_t[d])
            + lambda_plus * alpha_plus * (u[d] + a * nhat[d]);
        }
      mfem::real_t u_dot_du_t = 0.0;
      for (int d = 0; d < dim; ++d) { u_dot_du_t += u[d] * du_t[d]; }
      diss[eq_energy] =
        lambda_minus * alpha_minus * (H - a * un)
        + lambda_zero * (0.5 * alpha_zero * u2 + rho_roe * u_dot_du_t)
        + lambda_plus * alpha_plus * (H + a * un);

      // Passive scalars share the contact eigenvalue.  Their acoustic share is
      // carried with the Roe-averaged mass fraction.
      for (int s = 0; s < gas.L.num_scalars; ++s)
        {
          const int eq = gas.L.eq_scalar0 + s;
          const mfem::real_t y_roe =
            (rootL * qL[eq] / rhoL + rootR * qR[eq] / rhoR) * root_sum_inv;
          diss[eq] = y_roe * dmass
            + lambda_zero * ((qR[eq] - qL[eq]) - y_roe * drho);
        }

      for (int eq = 0; eq < neq; ++eq)
        {
          mfem::real_t central = 0.0;
          for (int d = 0; d < dim; ++d)
            {
              central += 0.5 * (fL[eq][d] + fR[eq][d]) * nor[d];
            }
          flux[eq] = central - 0.5 * diss[eq];
        }

      return Kernels::rmax(lambda_zero,
                           Kernels::rmax(lambda_minus, lambda_plus));
    }

    struct InviscidFlux
    {
      template<typename GasModelT>
      MFEM_HOST_DEVICE inline mfem::real_t
      ComputeVolumeFlux(const GasModelT &gas,
                        const mfem::real_t *q1,
                        const mfem::real_t *q2,
                        const mfem::real_t *met1,
                        const mfem::real_t *met2,
                        mfem::real_t *flux) const
      {
        return ComputeVolumeFluxKernel(gas, q1, q2, met1, met2, flux);
      }

      template<typename GasModelT>
      MFEM_HOST_DEVICE inline mfem::real_t
      ComputeFaceFlux(const GasModelT &gas,
                      const mfem::real_t *qminus,
                      const mfem::real_t *qplus,
                      const mfem::real_t *nor,
                      mfem::real_t *flux) const
      {
        return ComputeFaceFluxKernel(gas, qminus, qplus, nor, flux);
      }
    };
  }
}
