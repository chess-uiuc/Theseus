// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "dgsem_cache_utilities.hpp"
#include "bc_kernels.hpp"

namespace Theseus
{

  namespace DGSEMIntegrator
  {
    
    template<typename ContextType>
    MFEM_HOST_DEVICE inline
    static mfem::real_t AssembleElementVolumeKernel(const ContextType &ctx,
                                              const mfem::real_t *el_u, const mfem::real_t *elJac_d,
                                              const mfem::real_t *elMetric_d, mfem::real_t *el_dudt)
    {

      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int Np_z = ctx.Np_z;
      const int dim = ctx.dim;
      const int neq = ctx.num_equations;
      const int dof = Np_x * Np_y * Np_z;
      const mfem::real_t *Dhat2_d = ctx.Dhat2_d;

      mfem::real_t f[Theseus::MAXEQ] = {0.,0.,0.,0.,0.};
      mfem::real_t state1[Theseus::MAXEQ];
      mfem::real_t state2[Theseus::MAXEQ];
      mfem::real_t J = 0.0;
      mfem::real_t max_char_speed = 0.0;

      { // X-direction (metric row 0)
        // Zero'ing probably unnecessary: Chandrashekar flux overwrites it every time
        // for(int q = 0;q < neq;q++) f[q] = 0.0;
        for (int k = 0; k < Np_z; k++)
          for (int j = 0; j < Np_y; j++)
            for (int i = 0; i < Np_x; i++)
              {
                int id1 = k * Np_y * Np_x + j * Np_x + i;
                Kernels::el_gather_state(el_u, dof, neq, id1, state1);
                J = elJac_d[id1];
                const mfem::real_t *met1 = elMetric_d+id1*dim*dim;
                for (int m = i + 1; m < Np_x; m++)
                  {
                    int id2 = k * Np_y * Np_x + j * Np_x + m;
                    Kernels::el_gather_state(el_u, dof, neq, id2, state2);
                    const mfem::real_t *met2 = elMetric_d + id2*dim*dim;

                    const mfem::real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                    max_char_speed = Kernels::rmax(cs, max_char_speed);

                    const mfem::real_t c1 = Dhat2_d[m + Np_x*i];
                    const mfem::real_t c2 = Dhat2_d[i + Np_x*m];
                    Kernels::el_scatter_add(f, dof, neq, id1, c1, el_dudt);
                    Kernels::el_scatter_add(f, dof, neq, id2, c2, el_dudt);

                  }
              }
      } // X-direction block

      // Y-direction (metric row 1)
      if(dim > 1) {
        for (int k = 0; k < Np_z; ++k)
          for (int j = 0; j < Np_y; ++j)
            for (int i = 0; i < Np_x; ++i)
              {
                const int id1 = k*Np_y*Np_x + j*Np_x + i;
                Kernels::el_gather_state(el_u, dof, neq, id1, state1);
                const mfem::real_t *met1 = elMetric_d + id1*dim*dim + 1*dim;

                for (int m = j+1; m < Np_y; ++m)
                  {
                    const int id2 = k*Np_y*Np_x + m*Np_x + i;
                    Kernels::el_gather_state(el_u, dof, neq, id2, state2);
                    const mfem::real_t *met2 = elMetric_d + id2*dim*dim + dim;
                    // ComputeVolumeFlux *overwrites* f, so don't worry about reuse
                    const mfem::real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                    max_char_speed = Kernels::rmax(max_char_speed, cs);

                    const mfem::real_t c1 = Dhat2_d[m + Np_y*j]; // column j, entry m
                    const mfem::real_t c2 = Dhat2_d[j + Np_y*m]; // column m, entry j
                    Kernels::el_scatter_add(f, dof, neq, id1, c1, el_dudt);
                    Kernels::el_scatter_add(f, dof, neq, id2, c2, el_dudt);

                  }
              }
      } // Y-direction block

      if (dim > 2) { // Z-direction (metric row 2)
        for (int k = 0; k < Np_z; ++k)
          for (int j = 0; j < Np_y; ++j)
            for (int i = 0; i < Np_x; ++i)
              {
                const int id1 = k*Np_y*Np_x + j*Np_x + i;
                Kernels::el_gather_state(el_u, dof, neq, id1, state1);
                const mfem::real_t *met1 = elMetric_d + id1*dim*dim + 2*dim;

                for (int m = k+1; m < Np_z; ++m)
                  {
                    const int id2 = m*Np_y*Np_x + j*Np_x + i;
                    Kernels::el_gather_state(el_u, dof, neq, id2, state2);
                    const mfem::real_t *met2 = elMetric_d + id2*dim*dim + 2*dim;

                    const mfem::real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                    max_char_speed = Kernels::rmax(max_char_speed, cs);

                    const mfem::real_t c1 = Dhat2_d[m + Np_z*k];
                    const mfem::real_t c2 = Dhat2_d[k + Np_z*m];
                    Kernels::el_scatter_add(f, dof, neq, id1, c1, el_dudt);
                    Kernels::el_scatter_add(f, dof, neq, id2, c2, el_dudt);
                  }
              }
      } // Z-direction block
      // const int NPtot = Np_x * Np_y * Np_z; // = Np_x * Np_x * Np_x (!)
      Kernels::el_scale(elJac_d, -1.0, dof, neq, el_dudt);

      return max_char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE static mfem::real_t AssembleElementFaceKernel(const ContextT &ctx, const mfem::real_t *u_face,
                                                             const mfem::real_t *nor_face,const mfem::real_t *w_minus,
                                                             const mfem::real_t *w_plus, mfem::real_t *rhs_face)
    { // TODO: Fix hard-coded sizes (5)
      mfem::real_t max_char_speed = 0.0;
      mfem::real_t point_flux[5];
      mfem::real_t qMinus[5];
      mfem::real_t qPlus[5];
      const int nfp = ctx.num_face_points;
      const int neq = ctx.num_equations;
      const int dim = ctx.dim;
      // auto idx = [=](int side, int fp, int eq) -> int
      // {
      //   return (((side)*neq + eq)*nfp + fp);
      // };
      for (int i = 0; i < nfp; i++)
        {
          const mfem::real_t *nor_d = nor_face + i*dim;
          const mfem::real_t wminus = -w_minus[i];
          const mfem::real_t wplus = w_plus[i];
          // Could avoid these copy-in,out 
          for(int j = 0;j < neq;j++){
            qMinus[j] = u_face[ctx.iface_idx(0,i,j)];
            qPlus[j] = u_face[ctx.iface_idx(1,i,j)];
          }
          max_char_speed = \
            Kernels::rmax(max_char_speed, ctx.iflux.ComputeFaceFlux(ctx.gas, qMinus, qPlus,
                                                                    nor_d, point_flux));
          for(int j = 0;j < neq;j++){
            rhs_face[ctx.iface_idx(0, i, j)] = wminus * point_flux[j];
            rhs_face[ctx.iface_idx(1, i, j)] = wplus * point_flux[j];
          }
        }
      // #ifdef AXISYMMETRIC
      //        mfem::Vector phys(dim);
      //        Tr.Transform(ip, phys);
      //        mfem::real_t r = phys[1]; 
      //        flux_num *= r;
      // #endif
      return max_char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE static mfem::real_t AssembleViscousElementFaceKernel(const ContextT &ctx, const mfem::real_t *u_face,
                                                                    const mfem::real_t *nor_face,const mfem::real_t *w_minus,
                                                                    const mfem::real_t *w_plus, const mfem::real_t *dprim_face_x,
                                                                    const mfem::real_t *dprim_face_y, const mfem::real_t*dprim_face_z,
                                                                    mfem::real_t *rhs_face)
    { // TODO: Fix hard-coded sizes (5)
      mfem::real_t max_char_speed = 0.0;
      mfem::real_t point_flux[Theseus::MAXEQ];
      mfem::real_t vflux_minus[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t vflux_plus[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t qMinus[Theseus::MAXEQ];
      mfem::real_t qPlus[Theseus::MAXEQ];
      mfem::real_t gradPrim_plus[Theseus::MAXDIM][Theseus::MAXEQ];
      mfem::real_t gradPrim_minus[Theseus::MAXDIM][Theseus::MAXEQ];
      const mfem::real_t *dprim_face[Theseus::MAXDIM] = {dprim_face_x, dprim_face_y, dprim_face_z};
      const int nfp = ctx.num_face_points;
      const int neq = ctx.num_equations;
      const int dim = ctx.dim;
      // auto idx = [=](int side, int fp, int eq) -> int
      // {
      //   return (((side)*neq + eq)*nfp + fp);
      // };
      for (int i = 0; i < nfp; i++)
        {
          const mfem::real_t *nor_d = nor_face + i*dim;
          const mfem::real_t wminus = -w_minus[i];
          const mfem::real_t wplus = w_plus[i];
          // Could avoid these copy-in,out 
          for(int j = 0;j < neq;j++){
            int minus_index = ctx.iface_idx(0, i, j);
            int plus_index = ctx.iface_idx(1, i, j);
            qMinus[j] = u_face[minus_index];
            qPlus[j] = u_face[plus_index];
            for(int idim = 0;idim < dim;idim++){
              gradPrim_minus[idim][j] = dprim_face[idim][minus_index];
              gradPrim_plus[idim][j] = dprim_face[idim][plus_index];
            }
          }
          max_char_speed = \
            Kernels::rmax(max_char_speed, ctx.iflux.ComputeFaceFlux(ctx.gas, qMinus, qPlus,
                                                                    nor_d, point_flux));

          // Here, point_flux is +(F_inv * Normal)

          // Grab the viscous flux
          NavierStokesFlux::ComputeViscousFluxKernel(ctx.gas, qMinus,
                                                     gradPrim_minus[0],
                                                     gradPrim_minus[1],
                                                     gradPrim_minus[2], vflux_minus);
          NavierStokesFlux::ComputeViscousFluxKernel(ctx.gas, qPlus,
                                                     gradPrim_plus[0],
                                                     gradPrim_plus[1],
                                                     gradPrim_plus[2], vflux_plus);

          // Now we have vflux(+) and vflux(-)
          // In this loop:
          //  - average vflux
          //  - dot avg vflux with nor
          //  - accumulate dotted (avg*n) into point_flux
          for(int j = 0;j < neq;j++){
            for(int idim = 0;idim < dim;idim++){
              mfem::real_t avg = 0.5*(vflux_minus[j][idim] + vflux_plus[j][idim]);
              point_flux[j] -= nor_d[idim]*avg;
            }
          }
          // So now: point_flux = +(F_inv * Normal) -(F^bar_visc * Normal)
          // in this loop:
          // - SET/Overwrite rhs_face
          // - NEGATE the (-) face point_flux to properly orient
          for(int j = 0;j < neq;j++){
            rhs_face[ctx.iface_idx(0, i, j)] = wminus * point_flux[j];
            rhs_face[ctx.iface_idx(1, i, j)] = wplus * point_flux[j];
          }
        }
      // #ifdef AXISYMMETRIC
      //        mfem::Vector phys(dim);
      //        Tr.Transform(ip, phys);
      //        mfem::real_t r = phys[1]; 
      //        flux_num *= r;
      // #endif
      return max_char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE inline static mfem::real_t ComputeFVFluxesKernel(const ContextT &ctx,
                                                                const mfem::real_t *el_u,
                                                                const mfem::real_t *elJac,
                                                                const mfem::real_t *el_metric_xi,
                                                                const mfem::real_t *el_metric_eta,
                                                                const mfem::real_t *el_metric_zeta,
                                                                mfem::real_t *el_dudt)
    {
      const int dim = ctx.dim;
      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int Np_z = ctx.Np_z;
      const int neq = ctx.num_equations;
      const int npe = Np_x * Np_y * Np_z;
      const int ndofe = npe * neq;
      const mfem::real_t *qWgt = ctx.subcell_weights_d;

      mfem::real_t max_char_speed = 0.0;
      mfem::real_t flux_num[5];
      mfem::real_t du_subcell[5];
      mfem::real_t state1_local[5];
      mfem::real_t state2_local[5];

      for (int k = 0; k < Np_z; k++)
        {
          for (int j = 0; j < Np_y; j++)
            {
              for(int q = 0; q < neq;q++){
                du_subcell[q] = 0.0;
              }
              int id1 = k * Np_y * Np_x + j * Np_x;
              Kernels::el_gather_state(el_u, npe, neq, id1, state1_local);
              for (int i = 0; i < Np_x - 1; i++)
                {
                  int id2 = id1 + 1;
                  Kernels::el_gather_state(el_u, npe, neq, id2, state2_local);
                  const mfem::real_t *nor = el_metric_xi + id2*dim;

                  max_char_speed = \
                    Kernels::rmax(max_char_speed,
                                  ctx.iflux.ComputeFaceFlux(ctx.gas, state1_local,
                                                            state2_local, nor, flux_num));
                  for(int q = 0; q < neq;q++){
                    du_subcell[q] -= flux_num[q];
                  }
                  for(int q = 0; q < neq;q++){
                    du_subcell[q] /= (elJac[id1] * qWgt[i]);
                  }
                  Kernels::el_scatter_assign(du_subcell, npe, neq, id1, 1.0, el_dudt);
                  for(int q = 0; q < neq;q++){
                    du_subcell[q] = flux_num[q];
                  }
                  for(int q = 0;q < neq;q++){
                    state1_local[q] = state2_local[q];
                  }
                  id1 = id2;
                }
              for(int q = 0;q < neq;q++){
                du_subcell[q] /= (elJac[id1] * qWgt[Np_x-1]);
              }
              Kernels::el_scatter_assign(du_subcell, npe, neq, id1, 1.0, el_dudt);
            }
        }

      if (dim > 1)
        {
          for (int k = 0; k < Np_z; k++)
            {
              for (int i = 0; i < Np_x; i++)
                {
                  for(int q = 0; q < neq;q++){
                    du_subcell[q] = 0.0;
                  }
                  int id1 = k * Np_y * Np_x + i;
                  Kernels::el_gather_state(el_u, npe, neq, id1,
                                           state1_local);
                  for (int j = 0; j < Np_y - 1; j++)
                    {
                      int id2 = k * Np_y * Np_x + (j + 1) * Np_x + i;
                      Kernels::el_gather_state(el_u, npe, neq, id2,
                                               state2_local);
                      const mfem::real_t *nor = el_metric_eta + id2*dim;
                      max_char_speed = \
                        Kernels::rmax(max_char_speed,
                                      ctx.iflux.ComputeFaceFlux(ctx.gas,
                                                                state1_local,
                                                                state2_local,
                                                                nor, flux_num));
                      for(int q = 0;q < neq;q++){
                        du_subcell[q] -= flux_num[q];
                      }
                      for(int q = 0;q < neq;q++){
                        du_subcell[q] /= (elJac[id1] * qWgt[j]);
                      }
                      Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
                      for(int q = 0;q < neq;q++){
                        du_subcell[q] = flux_num[q];
                        state1_local[q] = state2_local[q];
                      }
                      id1 = id2;                   
                    }
                  for(int q = 0;q < neq;q++){
                    du_subcell[q] /= (elJac[id1] * qWgt[Np_y - 1]);
                  }
                  Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
                }
            }
          if (dim > 2)
            {
              for (int j = 0; j < Np_y; j++)
                {
                  for (int i = 0; i < Np_x; i++)
                    {
                      for(int q = 0; q < neq;q++){
                        du_subcell[q] = 0.0;
                      }
                      int id1 = j * Np_x + i;
                      Kernels::el_gather_state(el_u, npe, neq, id1,
                                               state1_local);
                      for (int k = 0; k < Np_z - 1; k++)
                        {
                          int id2 = (k + 1) * Np_y * Np_x + j * Np_x + i;
                          Kernels::el_gather_state(el_u, npe, neq, id2,
                                                   state2_local);
                          const mfem::real_t *nor = el_metric_zeta + id2*dim;
                          max_char_speed = \
                            Kernels::rmax(max_char_speed,
                                          ctx.iflux.ComputeFaceFlux(ctx.gas, state1_local,
                                                                    state2_local, nor, flux_num));
                          for(int q = 0;q < neq;q++){
                            du_subcell[q] -= flux_num[q];
                          }
                          for(int q = 0;q < neq;q++){
                            du_subcell[q] /= (elJac[id1] * qWgt[k]);
                          }
                          Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
                      
                          for(int q = 0;q < neq;q++){
                            du_subcell[q] = flux_num[q];
                            state1_local[q] = state2_local[q];
                          }
                          id1 = id2;            
                        }
                      for(int q = 0;q < neq;q++){
                        du_subcell[q] /= (elJac[id1] * qWgt[Np_z - 1]);
                      }
                      Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
                    }
                }
            }
        }
      return max_char_speed;
    }


    template<typename ContextType>
    MFEM_HOST_DEVICE inline
    static void AssembleViscousElementVolumeKernel(const ContextType &ctx,
                                                   const mfem::real_t *el_u,
                                                   const mfem::real_t *elJac_d,
                                                   const mfem::real_t *elMetric_d,
                                                   const mfem::real_t *el_gradprim_x,
                                                   const mfem::real_t *el_gradprim_y,
                                                   const mfem::real_t *el_gradprim_z,
                                                   mfem::real_t *el_dudt)
    {
      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int Np_z = ctx.Np_z;
      const int dim  = ctx.dim;
      const int neq  = ctx.num_equations;
      const int dof  = Np_x * Np_y * Np_z;
      const mfem::real_t *Dhat_d = ctx.Dhat_d;

      // One source-point scratch
      mfem::real_t state[Theseus::MAXEQ] = {0., 0., 0., 0., 0.};
      mfem::real_t dqx  [Theseus::MAXEQ] = {0., 0., 0., 0., 0.};
      mfem::real_t dqy  [Theseus::MAXEQ] = {0., 0., 0., 0., 0.};
      mfem::real_t dqz  [Theseus::MAXEQ] = {0., 0., 0., 0., 0.};

      // flux(eq,dir)
      mfem::real_t flux_eq_dir[Theseus::MAXEQ][Theseus::MAXDIM] = {{0.}};
      // one transformed reference-direction flux vector
      mfem::real_t f_ref[Theseus::MAXEQ] = {0., 0., 0., 0., 0.};

      for (int k = 0; k < Np_z; ++k)
        {
          for (int j = 0; j < Np_y; ++j)
            {
              for (int i = 0; i < Np_x; ++i)
                {
                  const int id1 = k * Np_y * Np_x + j * Np_x + i;
                  const mfem::real_t J = elJac_d[id1];
                  const mfem::real_t jInv = 1.0/J;

                  mfem::real_t dU_viscous[Theseus::MAXEQ] = {0., 0., 0., 0., 0.};

                  // xi contribution
                  for (int l = 0; l < Np_x; ++l)
                    {
                      const int idl = k * Np_y * Np_x + j * Np_x + l;
                      const mfem::real_t c = Dhat_d[l + Np_x * i];

                      Kernels::el_gather_state(el_u, dof, neq, idl, state);
                      Kernels::el_gather_grad_state(el_gradprim_x, el_gradprim_y,
                                                    el_gradprim_z, dim, dof, neq, idl,
                                                    dqx, dqy, dqz);

                      const mfem::real_t *adj_row = elMetric_d + idl * dim * dim + 0 * dim;
                      Theseus::NavierStokesFlux::compute_ref_viscous_flux(ctx.gas, dim, neq, state, dqx, dqy, dqz,
                                                                          adj_row, f_ref);
                      for (int q = 0; q < neq; ++q)
                        {
                          dU_viscous[q] += c * f_ref[q];
                        }
                    }

                  // eta contribution
                  if (dim > 1)
                    {
                      for (int l = 0; l < Np_y; ++l)
                        {
                          const int idl = k * Np_y * Np_x + l * Np_x + i;
                          const mfem::real_t c = Dhat_d[l + Np_y * j];

                          Kernels::el_gather_state(el_u, dof, neq, idl, state);
                          Kernels::el_gather_grad_state(el_gradprim_x, el_gradprim_y,
                                                        el_gradprim_z, dim, dof, neq, idl,
                                                        dqx, dqy, dqz);

                          const mfem::real_t *adj_row = elMetric_d + idl * dim * dim + 1 * dim;
                          Theseus::NavierStokesFlux::compute_ref_viscous_flux(ctx.gas, dim, neq, state,
                                                                              dqx, dqy, dqz,
                                                                              adj_row, f_ref);

                          for (int q = 0; q < neq; ++q)
                            {
                              dU_viscous[q] += c * f_ref[q];
                            }
                        }
                    }

                  // zeta contribution
                  if (dim > 2)
                    {
                      for (int l = 0; l < Np_z; ++l)
                        {
                          const int idl = l * Np_y * Np_x + j * Np_x + i;
                          const mfem::real_t c = Dhat_d[l + Np_z * k];

                          Kernels::el_gather_state(el_u, dof, neq, idl, state);
                          Kernels::el_gather_grad_state(el_gradprim_x, el_gradprim_y,
                                                        el_gradprim_z, dim, dof, neq, idl,
                                                        dqx, dqy, dqz);

                          const mfem::real_t *adj_row = elMetric_d + idl * dim * dim + 2 * dim;
                          Theseus::NavierStokesFlux::compute_ref_viscous_flux(ctx.gas, dim, neq, state,
                                                                              dqx, dqy, dqz,
                                                                              adj_row, f_ref);

                          for (int q = 0; q < neq; ++q)
                            {
                              dU_viscous[q] += c * f_ref[q];
                            }
                        }
                    }
                  Kernels::el_scatter_add(dU_viscous, dof, neq, id1, jInv, el_dudt);
                }
            }
        }
    }

    template <typename ContextType>
    MFEM_HOST_DEVICE inline
    static void AssembleGradElementVolumeKernel(const ContextType &ctx,
                                                const mfem::real_t *el_u,
                                                const mfem::real_t *elJac_d,
                                                const mfem::real_t *elMetric_d,
                                                mfem::real_t *el_grad_u[Theseus::MAXDIM])
    {
      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int Np_z = ctx.Np_z;
      const int neq  = ctx.num_equations;
      const int dim  = ctx.dim;
      const int dof  = Np_x * Np_y * Np_z;
      const mfem::real_t *D_d = ctx.D_d;

      if(dim == 1){

        // Keep MAX_EQ in mind later if neq can exceed 5.
        mfem::real_t dudxi[Theseus::MAXEQ];

        for (int i = 0; i < Np_x; ++i)
          {
            const int id = i;

            for (int q = 0; q < neq; ++q)
              {
                dudxi[q]  = 0.0;
              }
            // Reference-space derivatives.
            for (int l = 0; l < Np_x; ++l)
              {
                const int id_x = l;
                const mfem::real_t c_xi  = D_d[i*Np_x + l]; // legacy D_T(l, i)

                for (int q = 0; q < neq; ++q)
                  {
                    dudxi[q]  += el_u[id_x + q * dof] * c_xi;
                  }
              }

            const mfem::real_t invJ = 1.0 / elJac_d[id];
            const mfem::real_t *adj = elMetric_d + id * dim * dim;

            for (int q = 0; q < neq; ++q)
              {
                el_grad_u[0][id + q * dof] = invJ * (dudxi[q] * adj[0]);
              }
          }
      } else if(dim == 2){
        // Keep MAX_EQ in mind later if neq can exceed 5.
        mfem::real_t dudxi[Theseus::MAXEQ];
        mfem::real_t dudeta[Theseus::MAXEQ];

        for (int j = 0; j < Np_y; ++j)
          {
            for (int i = 0; i < Np_x; ++i)
              {
                const int id = j * Np_x + i;

                for (int q = 0; q < neq; ++q)
                  {
                    dudxi[q]  = 0.0;
                    dudeta[q] = 0.0;
                  }

                // Reference-space derivatives.
                for (int l = 0; l < Np_x; ++l)
                  {
                    const int id_x = j * Np_x + l;
                    const int id_y = l * Np_x + i;

                    const mfem::real_t c_xi  = D_d[l + Np_x * i]; // legacy D_T(l,i)
                    const mfem::real_t c_eta = D_d[l + Np_x * j]; // legacy D_T(l,j)

                    for (int q = 0; q < neq; ++q)
                      {
                        dudxi[q]  += el_u[id_x + q * dof] * c_xi;
                        dudeta[q] += el_u[id_y + q * dof] * c_eta;
                      }
                  }

                const mfem::real_t invJ = 1.0 / elJac_d[id];
                const mfem::real_t *adj = elMetric_d + id * dim * dim;

                // adj stored row-major per point:
                // [ adj[0] adj[1] ]
                // [ adj[2] adj[3] ]
                for (int q = 0; q < neq; ++q)
                  {
                    el_grad_u[0][id + q * dof] = invJ * (dudxi[q] * adj[0] + dudeta[q] * adj[2]);
                    el_grad_u[1][id + q * dof] = invJ * (dudxi[q] * adj[1] + dudeta[q] * adj[3]);
                  }
              }
          }
      } else if (dim == 3) {

        mfem::real_t dudxi[Theseus::MAXEQ];
        mfem::real_t dudeta[Theseus::MAXEQ];
        mfem::real_t dudzeta[Theseus::MAXEQ];

        for (int k = 0; k < Np_z; ++k)
          {
            for (int j = 0; j < Np_y; ++j)
              {
                for (int i = 0; i < Np_x; ++i)
                  {
                    const int id = k * Np_x * Np_y + j * Np_x + i;

                    for (int q = 0; q < neq; ++q)
                      {
                        dudxi[q]  = 0.0;
                        dudeta[q] = 0.0;
                        dudzeta[q] = 0.0;
                      }

                    // Reference-space derivatives.
                    for (int l = 0; l < Np_x; ++l)
                      {
                        const int id_x = k * Np_x * Np_y + j * Np_x + l;
                        const int id_y = k * Np_x * Np_y + l * Np_x + i;
                        const int id_z = l * Np_x * Np_y + j * Np_x + i;
                        const mfem::real_t c_xi  = D_d[l + Np_x * i]; // legacy D_T(l,i)
                        const mfem::real_t c_eta = D_d[l + Np_x * j]; // legacy D_T(l,j)
                        const mfem::real_t c_zeta = D_d[l + Np_x * k]; // legacy D_T(l,k)
                        for (int q = 0; q < neq; ++q)
                          {
                            dudxi[q]  += el_u[id_x + q * dof] * c_xi;
                            dudeta[q] += el_u[id_y + q * dof] * c_eta;
                            dudzeta[q] += el_u[id_z + q * dof] * c_zeta;
                          }
                      }

                    const mfem::real_t invJ = 1.0 / elJac_d[id];
                    const mfem::real_t *adj = elMetric_d + id * dim * dim;

                    // adj stored row-major per point:
                    // [ adj[0] adj[1] adj[2] ]
                    // [ adj[3] adj[4] adj[5] ]
                    // [ adj[6] adj[7] adj[8] ]
                    for (int q = 0; q < neq; ++q)
                      {
                        el_grad_u[0][id + q * dof] = invJ * (dudxi[q] * adj[0] +
                                                             dudeta[q] * adj[3] +
                                                             dudzeta[q] * adj[6]);

                        el_grad_u[1][id + q * dof] = invJ * (dudxi[q] * adj[1] +
                                                             dudeta[q] * adj[4] +
                                                             dudzeta[q] * adj[7]);

                        el_grad_u[2][id + q * dof] = invJ * (dudxi[q] * adj[2] +
                                                             dudeta[q] * adj[5] +
                                                             dudzeta[q] * adj[8]);
                      }
                  }
              }
          }
      }
    }

    template <typename ContextT>
    MFEM_HOST_DEVICE inline
    static void AssembleGradInteriorFaceKernel(const ContextT &ctx,
                                               const mfem::real_t *u_face,
                                               const mfem::real_t *nor_face,
                                               const mfem::real_t *w_minus,
                                               const mfem::real_t *w_plus,
                                               mfem::real_t *rhs_face[Theseus::MAXDIM])
    {
      const int nfp = ctx.num_face_points;
      const int neq = ctx.num_equations;
      const int dim = ctx.dim;

      mfem::real_t qMinus[Theseus::MAXEQ];
      mfem::real_t qPlus[Theseus::MAXEQ];
      mfem::real_t jump[Theseus::MAXEQ];

      for (int i = 0; i < nfp; ++i)
        {
          const mfem::real_t *nor_d = nor_face + i * dim;

          const mfem::real_t wminus = w_minus[i];
          const mfem::real_t wplus  = w_plus[i];

          for (int q = 0; q < neq; ++q)
            {
              qMinus[q] = u_face[ctx.iface_idx(0, i, q)];
              qPlus[q]  = u_face[ctx.iface_idx(1, i, q)];
              jump[q]   = mfem::real_t(0.5) * (qPlus[q] - qMinus[q]);
            }

          for ( int idim = 0;idim < dim;idim++){
            mfem::real_t *rhs_d = rhs_face[idim];
            const mfem::real_t n_d = nor_d[idim];
            for (int q = 0; q < neq; ++q)
              {
                const mfem::real_t f_d = jump[q]*n_d;
                rhs_d[ctx.iface_idx(0, i, q)] = wminus * f_d;
                rhs_d[ctx.iface_idx(1, i, q)] = wplus * f_d;
              }
          }
        }
    }

    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE inline
    static void AssembleGradBoundaryPointKernel(const DeviceCacheT &dc,
                                                const Theseus::BCDescriptor &bc,
                                                const mfem::real_t *u_face,
                                                const mfem::real_t *nor_point,
                                                const mfem::real_t scale,
                                                const int fp,
                                                mfem::real_t *rhs_face[Theseus::MAXDIM])
    {
      const int dim = dc.dim;
      const int nfp = dc.num_face_points;
      const int neq = dc.num_equations;

      mfem::real_t state1[Theseus::MAXEQ];
      mfem::real_t fluxN[Theseus::MAXEQ];
      mfem::real_t flux_dir[Theseus::MAXEQ];

      Theseus::Kernels::el_gather_state(u_face, nfp, neq, fp, state1);

      Theseus::BC::ComputeBdrFaceGradFlux(dc, bc, state1, fluxN);

      for(int idim = 0;idim < dim;idim++){
        for(int q = 0;q < neq;q++){
          flux_dir[q] = fluxN[q]*nor_point[idim];
        }
        Theseus::Kernels::el_scatter_add(flux_dir, nfp, neq, fp, scale, rhs_face[idim]);
      }
    }

  };
}
