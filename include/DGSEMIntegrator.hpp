// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "AxisymmetricSource.hpp"
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

    template<typename ContextType>
    MFEM_HOST_DEVICE inline
    static mfem::real_t AssembleVolumePointKernel(
                                                  const ContextType &ctx, const mfem::real_t *el_u,
                                                  const mfem::real_t *elJac_d, const mfem::real_t *elMetric_d,
                                                  const int point, mfem::real_t *el_dudt)
    {
      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int Np_z = ctx.Np_z;
      const int dim = ctx.dim;
      const int neq = ctx.num_equations;
      const int dof = ctx.ndof_scalar_el;
      const mfem::real_t *Dhat2_d = ctx.Dhat2_d;
      const int i = point % Np_x;
      const int j = (point / Np_x) % Np_y;
      const int k = point / (Np_x*Np_y);

      mfem::real_t state_lower[Theseus::MAXEQ];
      mfem::real_t state_upper[Theseus::MAXEQ];
      mfem::real_t flux[Theseus::MAXEQ];
      mfem::real_t point_rate[Theseus::MAXEQ] = {0.0};
      mfem::real_t max_char_speed = 0.0;

      for (int m = 0; m < Np_x; ++m)
        {
          if (m == i) { continue; }
          const int lower = m < i ? m : i;
          const int upper = m < i ? i : m;
          const int lower_point = k*Np_y*Np_x + j*Np_x + lower;
          const int upper_point = k*Np_y*Np_x + j*Np_x + upper;
          Kernels::el_gather_state(el_u, dof, neq, lower_point, state_lower);
          Kernels::el_gather_state(el_u, dof, neq, upper_point, state_upper);
          const mfem::real_t char_speed = ctx.iflux.ComputeVolumeFlux(
                                                                      ctx.gas, state_lower, state_upper,
                                                                      elMetric_d + lower_point*dim*dim,
                                                                      elMetric_d + upper_point*dim*dim, flux);
          max_char_speed = Kernels::rmax(max_char_speed, char_speed);
          const mfem::real_t coefficient = Dhat2_d[m + Np_x*i];
          for (int q = 0; q < neq; ++q)
            {
              point_rate[q] += coefficient*flux[q];
            }
        }

      if (dim > 1)
        {
          for (int m = 0; m < Np_y; ++m)
            {
              if (m == j) { continue; }
              const int lower = m < j ? m : j;
              const int upper = m < j ? j : m;
              const int lower_point = k*Np_y*Np_x + lower*Np_x + i;
              const int upper_point = k*Np_y*Np_x + upper*Np_x + i;
              Kernels::el_gather_state(
                                       el_u, dof, neq, lower_point, state_lower);
              Kernels::el_gather_state(
                                       el_u, dof, neq, upper_point, state_upper);
              const mfem::real_t char_speed = ctx.iflux.ComputeVolumeFlux(
                                                                          ctx.gas, state_lower, state_upper,
                                                                          elMetric_d + lower_point*dim*dim + dim,
                                                                          elMetric_d + upper_point*dim*dim + dim, flux);
              max_char_speed = Kernels::rmax(max_char_speed, char_speed);
              const mfem::real_t coefficient = Dhat2_d[m + Np_y*j];
              for (int q = 0; q < neq; ++q)
                {
                  point_rate[q] += coefficient*flux[q];
                }
            }
        }

      if (dim > 2)
        {
          for (int m = 0; m < Np_z; ++m)
            {
              if (m == k) { continue; }
              const int lower = m < k ? m : k;
              const int upper = m < k ? k : m;
              const int lower_point = lower*Np_y*Np_x + j*Np_x + i;
              const int upper_point = upper*Np_y*Np_x + j*Np_x + i;
              Kernels::el_gather_state(
                                       el_u, dof, neq, lower_point, state_lower);
              Kernels::el_gather_state(
                                       el_u, dof, neq, upper_point, state_upper);
              const mfem::real_t char_speed = ctx.iflux.ComputeVolumeFlux(
                                                                          ctx.gas, state_lower, state_upper,
                                                                          elMetric_d + lower_point*dim*dim + 2*dim,
                                                                          elMetric_d + upper_point*dim*dim + 2*dim, flux);
              max_char_speed = Kernels::rmax(max_char_speed, char_speed);
              const mfem::real_t coefficient = Dhat2_d[m + Np_z*k];
              for (int q = 0; q < neq; ++q)
                {
                  point_rate[q] += coefficient*flux[q];
                }
            }
        }

      Kernels::el_scatter_assign(
                                 point_rate, dof, neq, point, -1.0/elJac_d[point], el_dudt);
      return max_char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE static mfem::real_t AssembleFacePointKernel(const ContextT &ctx,
                                                                 const mfem::real_t *u_face,
                                                                 const mfem::real_t *nor_point,
                                                                 const mfem::real_t w_minus,
                                                                 const mfem::real_t w_plus,
                                                                 const int fp,
                                                                 mfem::real_t *rhs_face)
    {
      mfem::real_t point_flux[Theseus::MAXEQ];
      mfem::real_t qMinus[Theseus::MAXEQ];
      mfem::real_t qPlus[Theseus::MAXEQ];
      const int neq = ctx.num_equations;
      for(int q = 0; q < neq; ++q){
        qMinus[q] = u_face[ctx.iface_idx(0, fp, q)];
        qPlus[q] = u_face[ctx.iface_idx(1, fp, q)];
      }

      const mfem::real_t char_speed =
        ctx.iflux.ComputeFaceFlux(ctx.gas, qMinus, qPlus, nor_point, point_flux);

      for(int q = 0; q < neq; ++q){
        rhs_face[ctx.iface_idx(0, fp, q)] = -w_minus * point_flux[q];
        rhs_face[ctx.iface_idx(1, fp, q)] =  w_plus * point_flux[q];
      }

      return char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE static mfem::real_t AssembleElementFaceKernel(const ContextT &ctx, const mfem::real_t *u_face,
                                                                   const mfem::real_t *nor_face,const mfem::real_t *w_minus,
                                                                   const mfem::real_t *w_plus, mfem::real_t *rhs_face)
    {
      mfem::real_t max_char_speed = 0.0;
      const int nfp = ctx.num_face_points;
      const int dim = ctx.dim;
      for (int fp = 0; fp < nfp; ++fp)
        {
          const mfem::real_t char_speed =
            AssembleFacePointKernel(ctx, u_face, nor_face + fp*dim,
                                    w_minus[fp], w_plus[fp], fp, rhs_face);
          max_char_speed = Kernels::rmax(max_char_speed, char_speed);
        }
      return max_char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE static mfem::real_t AssembleViscousFacePointKernel(
                                                                        const ContextT &ctx, const mfem::real_t *u_face,
                                                                        const mfem::real_t *nor_point, const mfem::real_t w_minus,
                                                                        const mfem::real_t w_plus, const mfem::real_t *dprim_face_x,
                                                                        const mfem::real_t *dprim_face_y, const mfem::real_t *dprim_face_z,
                                                                        const mfem::real_t radius, const int fp, mfem::real_t *rhs_face)
    {
      mfem::real_t point_flux[Theseus::MAXEQ];
      mfem::real_t vflux_minus[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t vflux_plus[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t qMinus[Theseus::MAXEQ];
      mfem::real_t qPlus[Theseus::MAXEQ];
      mfem::real_t gradPrim_plus[Theseus::MAXDIM][Theseus::MAXEQ];
      mfem::real_t gradPrim_minus[Theseus::MAXDIM][Theseus::MAXEQ];
      const mfem::real_t *dprim_face[Theseus::MAXDIM] = {
        dprim_face_x, dprim_face_y, dprim_face_z};
      const int neq = ctx.num_equations;
      const int dim = ctx.dim;

      for(int q = 0; q < neq; ++q){
        const int minus_index = ctx.iface_idx(0, fp, q);
        const int plus_index = ctx.iface_idx(1, fp, q);
        qMinus[q] = u_face[minus_index];
        qPlus[q] = u_face[plus_index];
        for(int idim = 0; idim < dim; ++idim){
          gradPrim_minus[idim][q] = dprim_face[idim][minus_index];
          gradPrim_plus[idim][q] = dprim_face[idim][plus_index];
        }
      }

      const mfem::real_t char_speed =
        ctx.iflux.ComputeFaceFlux(ctx.gas, qMinus, qPlus, nor_point, point_flux);

      NavierStokesFlux::ComputeViscousFluxKernel(
                                                 ctx.gas, qMinus, gradPrim_minus[0], gradPrim_minus[1],
                                                 gradPrim_minus[2], vflux_minus, ctx.axisymmetric,
                                                 ctx.axisymmetric ? radius : 0.0);
      NavierStokesFlux::ComputeViscousFluxKernel(
                                                 ctx.gas, qPlus, gradPrim_plus[0], gradPrim_plus[1],
                                                 gradPrim_plus[2], vflux_plus, ctx.axisymmetric,
                                                 ctx.axisymmetric ? radius : 0.0);

      for(int q = 0; q < neq; ++q){
        for(int idim = 0; idim < dim; ++idim){
          const mfem::real_t avg =
            0.5*(vflux_minus[q][idim] + vflux_plus[q][idim]);
          point_flux[q] -= nor_point[idim]*avg;
        }
      }

      for(int q = 0; q < neq; ++q){
        rhs_face[ctx.iface_idx(0, fp, q)] = -w_minus * point_flux[q];
        rhs_face[ctx.iface_idx(1, fp, q)] =  w_plus * point_flux[q];
      }

      return char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE static mfem::real_t AssembleViscousElementFaceKernel(const ContextT &ctx, const mfem::real_t *u_face,
                                                                          const mfem::real_t *nor_face,const mfem::real_t *w_minus,
                                                                          const mfem::real_t *w_plus, const mfem::real_t *dprim_face_x,
                                                                          const mfem::real_t *dprim_face_y, const mfem::real_t*dprim_face_z,
                                                                          const mfem::real_t *face_radius,
                                                                          mfem::real_t *rhs_face)
    {
      mfem::real_t max_char_speed = 0.0;
      const int nfp = ctx.num_face_points;
      const int dim = ctx.dim;
      for (int fp = 0; fp < nfp; ++fp)
        {
          const mfem::real_t radius =
            ctx.axisymmetric ? face_radius[fp] : 0.0;
          const mfem::real_t char_speed = AssembleViscousFacePointKernel(
                                                                         ctx, u_face, nor_face + fp*dim, w_minus[fp], w_plus[fp],
                                                                         dprim_face_x, dprim_face_y, dprim_face_z, radius, fp, rhs_face);
          max_char_speed = Kernels::rmax(max_char_speed, char_speed);
        }
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
      const mfem::real_t *qWgt = ctx.subcell_weights_d;

      mfem::real_t max_char_speed = 0.0;
      mfem::real_t flux_num[Theseus::MAXEQ];
      mfem::real_t du_subcell[Theseus::MAXEQ];
      mfem::real_t state1_local[Theseus::MAXEQ];
      mfem::real_t state2_local[Theseus::MAXEQ];

      for(int i = 0;i < npe*neq;i++)
        el_dudt[i] = 0.0;

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
    static void AssembleViscousVolumePointKernel(
                                                 const ContextType &ctx, const mfem::real_t *el_u,
                                                 const mfem::real_t *elJac_d, const mfem::real_t *elMetric_d,
                                                 const mfem::real_t *elRadius_d,
                                                 const mfem::real_t *el_gradprim_x,
                                                 const mfem::real_t *el_gradprim_y,
                                                 const mfem::real_t *el_gradprim_z,
                                                 const int point, mfem::real_t *el_dudt)
    {
      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int Np_z = ctx.Np_z;
      const int dim = ctx.dim;
      const int neq = ctx.num_equations;
      const int dof = ctx.ndof_scalar_el;
      const mfem::real_t *Dhat_d = ctx.Dhat_d;
      const int i = point % Np_x;
      const int j = (point / Np_x) % Np_y;
      const int k = point / (Np_x*Np_y);

      mfem::real_t state[Theseus::MAXEQ] = {0.0};
      mfem::real_t dqx[Theseus::MAXEQ] = {0.0};
      mfem::real_t dqy[Theseus::MAXEQ] = {0.0};
      mfem::real_t dqz[Theseus::MAXEQ] = {0.0};
      mfem::real_t f_ref[Theseus::MAXEQ] = {0.0};
      mfem::real_t dU_viscous[Theseus::MAXEQ] = {0.0};

      for (int l = 0; l < Np_x; ++l)
        {
          const int sample = k*Np_y*Np_x + j*Np_x + l;
          const mfem::real_t coefficient = Dhat_d[l + Np_x*i];
          Kernels::el_gather_state(el_u, dof, neq, sample, state);
          Kernels::el_gather_grad_state(
                                        el_gradprim_x, el_gradprim_y, el_gradprim_z, dim, dof, neq,
                                        sample, dqx, dqy, dqz);
          Theseus::NavierStokesFlux::compute_ref_viscous_flux(
                                                              ctx.gas, dim, neq, state, dqx, dqy, dqz,
                                                              elMetric_d + sample*dim*dim, f_ref, ctx.axisymmetric,
                                                              ctx.axisymmetric ? elRadius_d[sample] : 0.0);
          for (int q = 0; q < neq; ++q)
            {
              dU_viscous[q] += coefficient*f_ref[q];
            }
        }

      if (dim > 1)
        {
          for (int l = 0; l < Np_y; ++l)
            {
              const int sample = k*Np_y*Np_x + l*Np_x + i;
              const mfem::real_t coefficient = Dhat_d[l + Np_y*j];
              Kernels::el_gather_state(el_u, dof, neq, sample, state);
              Kernels::el_gather_grad_state(
                                            el_gradprim_x, el_gradprim_y, el_gradprim_z, dim, dof, neq,
                                            sample, dqx, dqy, dqz);
              Theseus::NavierStokesFlux::compute_ref_viscous_flux(
                                                                  ctx.gas, dim, neq, state, dqx, dqy, dqz,
                                                                  elMetric_d + sample*dim*dim + dim, f_ref,
                                                                  ctx.axisymmetric,
                                                                  ctx.axisymmetric ? elRadius_d[sample] : 0.0);
              for (int q = 0; q < neq; ++q)
                {
                  dU_viscous[q] += coefficient*f_ref[q];
                }
            }
        }

      if (dim > 2)
        {
          for (int l = 0; l < Np_z; ++l)
            {
              const int sample = l*Np_y*Np_x + j*Np_x + i;
              const mfem::real_t coefficient = Dhat_d[l + Np_z*k];
              Kernels::el_gather_state(el_u, dof, neq, sample, state);
              Kernels::el_gather_grad_state(
                                            el_gradprim_x, el_gradprim_y, el_gradprim_z, dim, dof, neq,
                                            sample, dqx, dqy, dqz);
              Theseus::NavierStokesFlux::compute_ref_viscous_flux(
                                                                  ctx.gas, dim, neq, state, dqx, dqy, dqz,
                                                                  elMetric_d + sample*dim*dim + 2*dim, f_ref,
                                                                  ctx.axisymmetric,
                                                                  ctx.axisymmetric ? elRadius_d[sample] : 0.0);
              for (int q = 0; q < neq; ++q)
                {
                  dU_viscous[q] += coefficient*f_ref[q];
                }
            }
        }

      Kernels::el_scatter_add(
                              dU_viscous, dof, neq, point, 1.0/elJac_d[point], el_dudt);
      if (ctx.axisymmetric)
        {
          Kernels::el_gather_state(el_u, dof, neq, point, state);
          Kernels::el_gather_grad_state(
                                        el_gradprim_x, el_gradprim_y, el_gradprim_z, dim, dof, neq,
                                        point, dqx, dqy, dqz);
          mfem::real_t source[Theseus::MAXEQ] = {0.0};
          if (!AddAxisymmetricViscousSourceAwayFromAxis(
                                                        ctx.gas, state, dqx, dqy, dqz, elRadius_d[point], source))
            {
              AddAxisymmetricViscousSourceAtAxis(
                                                 ctx, el_u, el_gradprim_x, el_gradprim_y, el_gradprim_z,
                                                 elRadius_d, elJac_d, elMetric_d, point, source);
            }
          Kernels::el_scatter_add(source, dof, neq, point, 1.0, el_dudt);
        }
    }

    template<typename ContextType>
    MFEM_HOST_DEVICE inline
    static void AssembleViscousElementVolumeKernel(const ContextType &ctx,
                                                   const mfem::real_t *el_u,
                                                   const mfem::real_t *elJac_d,
                                                   const mfem::real_t *elMetric_d,
                                                   const mfem::real_t *elRadius_d,
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
                      Theseus::NavierStokesFlux::compute_ref_viscous_flux(
                                                                          ctx.gas, dim, neq, state, dqx, dqy, dqz, adj_row,
                                                                          f_ref, ctx.axisymmetric,
                                                                          ctx.axisymmetric ? elRadius_d[idl] : 0.0);
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
                          Theseus::NavierStokesFlux::compute_ref_viscous_flux(
                                                                              ctx.gas, dim, neq, state, dqx, dqy, dqz, adj_row,
                                                                              f_ref, ctx.axisymmetric,
                                                                              ctx.axisymmetric ? elRadius_d[idl] : 0.0);

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
                          Theseus::NavierStokesFlux::compute_ref_viscous_flux(
                                                                              ctx.gas, dim, neq, state, dqx, dqy, dqz, adj_row,
                                                                              f_ref, ctx.axisymmetric,
                                                                              ctx.axisymmetric ? elRadius_d[idl] : 0.0);

                          for (int q = 0; q < neq; ++q)
                            {
                              dU_viscous[q] += c * f_ref[q];
                            }
                        }
                    }
                  Kernels::el_scatter_add(dU_viscous, dof, neq, id1, jInv, el_dudt);
                  if (ctx.axisymmetric)
                    {
                      Kernels::el_gather_state(el_u, dof, neq, id1, state);
                      Kernels::el_gather_grad_state(
                                                    el_gradprim_x, el_gradprim_y, el_gradprim_z, dim,
                                                    dof, neq, id1, dqx, dqy, dqz);
                      mfem::real_t source[Theseus::MAXEQ] = {0.0};
                      if (!AddAxisymmetricViscousSourceAwayFromAxis(
                                                                    ctx.gas, state, dqx, dqy, dqz,
                                                                    elRadius_d[id1], source))
                        {
                          AddAxisymmetricViscousSourceAtAxis(
                                                             ctx, el_u, el_gradprim_x, el_gradprim_y,
                                                             el_gradprim_z, elRadius_d, elJac_d, elMetric_d,
                                                             id1, source);
                        }
                      Kernels::el_scatter_add(source, dof, neq, id1, 1.0,
                                              el_dudt);
                    }
                }
            }
        }
    }

    template <typename ContextType>
    MFEM_HOST_DEVICE inline
    static void AssembleGradVolumePointKernel(
                                              const ContextType &ctx, const mfem::real_t *el_u,
                                              const mfem::real_t *elJac_d, const mfem::real_t *elMetric_d,
                                              const int point, mfem::real_t *el_grad_u[Theseus::MAXDIM])
    {
      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int neq = ctx.num_equations;
      const int dim = ctx.dim;
      const int dof = ctx.ndof_scalar_el;
      const mfem::real_t *D_d = ctx.D_d;

      const int i = point % Np_x;
      const int j = (point / Np_x) % Np_y;
      const int k = point / (Np_x * Np_y);

      mfem::real_t dudxi[Theseus::MAXEQ] = {0.0};
      mfem::real_t dudeta[Theseus::MAXEQ] = {0.0};
      mfem::real_t dudzeta[Theseus::MAXEQ] = {0.0};

      for (int l = 0; l < Np_x; ++l)
        {
          const int sample = k*Np_y*Np_x + j*Np_x + l;
          const mfem::real_t coefficient = D_d[l + Np_x*i];
          for (int q = 0; q < neq; ++q)
            {
              dudxi[q] += el_u[sample + q*dof] * coefficient;
            }
        }

      if (dim > 1)
        {
          for (int l = 0; l < Np_y; ++l)
            {
              const int sample = k*Np_y*Np_x + l*Np_x + i;
              const mfem::real_t coefficient = D_d[l + Np_y*j];
              for (int q = 0; q < neq; ++q)
                {
                  dudeta[q] += el_u[sample + q*dof] * coefficient;
                }
            }
        }

      if (dim > 2)
        {
          for (int l = 0; l < ctx.Np_z; ++l)
            {
              const int sample = l*Np_y*Np_x + j*Np_x + i;
              const mfem::real_t coefficient = D_d[l + ctx.Np_z*k];
              for (int q = 0; q < neq; ++q)
                {
                  dudzeta[q] += el_u[sample + q*dof] * coefficient;
                }
            }
        }

      const mfem::real_t invJ = 1.0 / elJac_d[point];
      const mfem::real_t *adj = elMetric_d + point*dim*dim;
      for (int q = 0; q < neq; ++q)
        {
          if (dim == 1)
            {
              el_grad_u[0][point + q*dof] = invJ*dudxi[q]*adj[0];
            }
          else if (dim == 2)
            {
              el_grad_u[0][point + q*dof] =
                invJ*(dudxi[q]*adj[0] + dudeta[q]*adj[2]);
              el_grad_u[1][point + q*dof] =
                invJ*(dudxi[q]*adj[1] + dudeta[q]*adj[3]);
            }
          else
            {
              el_grad_u[0][point + q*dof] =
                invJ*(dudxi[q]*adj[0] + dudeta[q]*adj[3] +
                      dudzeta[q]*adj[6]);
              el_grad_u[1][point + q*dof] =
                invJ*(dudxi[q]*adj[1] + dudeta[q]*adj[4] +
                      dudzeta[q]*adj[7]);
              el_grad_u[2][point + q*dof] =
                invJ*(dudxi[q]*adj[2] + dudeta[q]*adj[5] +
                      dudzeta[q]*adj[8]);
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
    static void AssembleGradInteriorFacePointKernel(
                                                    const ContextT &ctx,
                                                    const mfem::real_t *u_face,
                                                    const mfem::real_t *nor_point,
                                                    const mfem::real_t w_minus,
                                                    const mfem::real_t w_plus,
                                                    const int fp,
                                                    mfem::real_t *rhs_face[Theseus::MAXDIM])
    {
      const int neq = ctx.num_equations;
      const int dim = ctx.dim;

      mfem::real_t jump[Theseus::MAXEQ];

      for (int q = 0; q < neq; ++q)
        {
          jump[q] = mfem::real_t(0.5) *
            (u_face[ctx.iface_idx(1, fp, q)] -
             u_face[ctx.iface_idx(0, fp, q)]);
        }

      for (int idim = 0; idim < dim; ++idim){
        mfem::real_t *rhs_d = rhs_face[idim];
        const mfem::real_t n_d = nor_point[idim];
        for (int q = 0; q < neq; ++q)
          {
            const mfem::real_t f_d = jump[q]*n_d;
            rhs_d[ctx.iface_idx(0, fp, q)] = w_minus * f_d;
            rhs_d[ctx.iface_idx(1, fp, q)] = w_plus * f_d;
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
      const int dim = ctx.dim;

      for (int fp = 0; fp < nfp; ++fp)
        {
          AssembleGradInteriorFacePointKernel(
                                              ctx, u_face, nor_face + fp*dim, w_minus[fp], w_plus[fp], fp,
                                              rhs_face);
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
