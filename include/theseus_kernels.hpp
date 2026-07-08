// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include "mfem.hpp"

namespace Theseus
{
  constexpr const int MAXEQ = 5;
  constexpr const int MAXDIM = 3;

  namespace Kernels {
    MFEM_HOST_DEVICE inline mfem::real_t rmax(mfem::real_t a, mfem::real_t b) { return a > b ? a : b; }
    MFEM_HOST_DEVICE inline mfem::real_t rmin(mfem::real_t a, mfem::real_t b) { return a < b ? a : b; }
    MFEM_HOST_DEVICE inline mfem::real_t rsqrt(mfem::real_t x) { return std::sqrt(x); }  // mfem::sqrt?
    MFEM_HOST_DEVICE inline mfem::real_t rlog(mfem::real_t x)  { return std::log(x); }   // mfem::log?
    MFEM_HOST_DEVICE inline mfem::real_t rpow(mfem::real_t x, mfem::real_t y) { return std::pow(x, y); };
    MFEM_HOST_DEVICE inline mfem::real_t rabs(mfem::real_t x) { return std::abs(x); };

    MFEM_HOST_DEVICE inline void Normalize(const int dim, mfem::real_t *vec){
      mfem::real_t fac = 0.0;
      for(int idim = 0;idim < dim;idim++){
        fac += (vec[idim]*vec[idim]);
      }
      fac = 1.0/rsqrt(fac);
      for(int idim = 0;idim < dim;idim++){
        vec[idim] *= fac;
      }
    }

    MFEM_HOST_DEVICE
    inline void Normal(const int dim, const mfem::real_t *vec, mfem::real_t *nor)
    {
      // MFEM_ASSERT(dim == 2 || dim == 3, "Normal only defined here for 2D/3D");
      
      if (dim == 2)
        {
          nor[0] = -vec[1];
          nor[1] =  vec[0];
          return;
        }
      
      // 3D
      const mfem::real_t x = vec[0];
      const mfem::real_t y = vec[1];
      const mfem::real_t z = vec[2];
      
      const mfem::real_t ax = std::fabs(x);
      const mfem::real_t ay = std::fabs(y);
      const mfem::real_t az = std::fabs(z);
      
      // Reject zero vector
      // MFEM_ASSERT(ax > 0 || ay > 0 || az > 0, "Zero vector has no normal");
      
      // Pick the coordinate axis least aligned with vec.
      // Then nor = vec x e_i.
      if (ax <= ay && ax <= az)
        {
          nor[0] =  0.0;
          nor[1] =  z;
          nor[2] = -y;
        }
      else if (ay <= ax && ay <= az)
        {
          nor[0] = -z;
          nor[1] =  0.0;
          nor[2] =  x;
        }
      else
        {
          nor[0] =  y;
          nor[1] = -x;
          nor[2] =  0.0;
        }
    }

    MFEM_HOST_DEVICE
    inline mfem::real_t Dot(const int dim, const mfem::real_t *vec1, const mfem::real_t *vec2)
    {
      mfem::real_t dp = 0.0;
      for(int idim = 0;idim < dim;idim++)
        dp += vec1[idim]*vec2[idim];
      return dp;
    }

    MFEM_HOST_DEVICE
    inline void Cross(const int dim, const mfem::real_t *vec1, const mfem::real_t *vec2, mfem::real_t *cross)
    {
      // MFEM_ASSERT(dim == 3, "Apply cross product only to 3D vectors");
      
      cross[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
      cross[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
      cross[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
    }

    MFEM_HOST_DEVICE
    inline void ComputeMeanVec(const mfem::real_t* a, const mfem::real_t* b, mfem::real_t* out, int n)
    {
      for (int i=0;i<n;++i) out[i] = mfem::real_t(0.5)*(a[i]+b[i]);
    }
    
    MFEM_HOST_DEVICE
    inline mfem::real_t ComputeLogMean(mfem::real_t x, mfem::real_t y, mfem::real_t eps) // eps defaults to 1e-4 on CPU
    {
      const mfem::real_t xi = y / x;
      const mfem::real_t u  = (xi*(xi - 2.0) + 1.0) / (xi*(xi + 2.0) + 1.0);
      
      // polynomial approximation branch when u is small
      if (u < eps)
        {
          // (x+y)*52.5 / (105 + u*(35 + u*(21 + 15*u)))
          const mfem::real_t denom = 105.0 + u*(35.0 + u*(21.0 + 15.0*u));
          return (x + y) * 52.5 / denom;
        }
      else
        {
          return (y - x) / Kernels::rlog(xi);
        }
    }

    // Element storage: component-major (q blocks), length = dof*num_eq
    // u[q*dof + id]
    MFEM_HOST_DEVICE inline
    mfem::real_t el_get(const mfem::real_t *u, int dof, int num_eq, int id, int q)
    {
      (void)num_eq; // not needed for this layout
      return u[q*dof + id];
    }
    
    MFEM_HOST_DEVICE inline
    void el_gather_state(const mfem::real_t *u, const int dof, const int num_eq, const int id, mfem::real_t *dst)
    {
      // MFEM_ASSERT(id >= 0 && id < dof, "element index out of bounds");
      for (int q = 0; q < num_eq; ++q)
        dst[q] = u[q*dof + id];
    }
    
    MFEM_HOST_DEVICE inline
    void el_gather_grad_state(const mfem::real_t *grad_state_x, const mfem::real_t *grad_state_y, const mfem::real_t *grad_state_z,
                              const int dim, const int dof, const int neq, const int id,
                              mfem::real_t *dqx, mfem::real_t *dqy, mfem::real_t *dqz)
    {
      el_gather_state(grad_state_x, dof, neq, id, dqx);
      if (dim > 1) el_gather_state(grad_state_y, dof, neq, id, dqy);
      if (dim > 2) el_gather_state(grad_state_z, dof, neq, id, dqz);
    }

    MFEM_HOST_DEVICE inline
    void el_scatter_add(const mfem::real_t *f,
                        const int dof,
                        const int num_eq,
                        const int id,
                        const mfem::real_t scale,
                        mfem::real_t *du)
    {
      // MFEM_ASSERT(id >= 0 && id < dof, "element index out of bounds");
      // Element storage is component-major (byVDIM):
      // du[q*dof + id] corresponds to "row id, component q" in DenseMatrix(dof, num_eq)
      for (int q = 0; q < num_eq; ++q)
        {
          du[id + q*dof] += scale * f[q];
        }
    }

    MFEM_HOST_DEVICE inline
    void el_scatter_assign(const mfem::real_t *f,
                           const int dof,
                           const int num_eq,
                           const int id,
                           const mfem::real_t scale,
                           mfem::real_t *du)
    {
      // MFEM_ASSERT(id >= 0 && id < dof, "element index out of bounds");
      // Element storage is component-major (byVDIM):
      // du[q*dof + id] corresponds to "row id, component q" in DenseMatrix(dof, num_eq)
      for (int q = 0; q < num_eq; ++q)
        {
          du[id + q*dof] = scale * f[q];
        }
    }

    MFEM_HOST_DEVICE inline
    void el_scale(const mfem::real_t *scale_d,
                  const mfem::real_t fac,
                  const int dof,      // scalar dofs per element
                  const int neq,
                  mfem::real_t *el_soln)        // num equations
    {
      for (int id = 0; id < dof; ++id)
        {
          const mfem::real_t invJ = fac / scale_d[id];
          for (int q = 0; q < neq; ++q)
            {
              el_soln[id + q*dof] *= invJ;
            }
        }
    }

    MFEM_HOST_DEVICE inline bool is_bad_value(mfem::real_t x)
    {
      return !std::isfinite(x);
    }

  }
}
