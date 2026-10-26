// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdio>
#include <climits>
#include "LTEGasModel.hpp"

namespace Theseus
{
  namespace ViscousInvariants
  {
    // Debugging branch: these checks remain active in Release/NDEBUG builds.
    MFEM_HOST_DEVICE inline void Fail(const char *where, const char *reason,
                                      int dim, int equations, int index,
                                      mfem::real_t value)
    {
      printf("VISCOUS_INVARIANT_FAILURE where=%s reason=%s dim=%d "
             "equations=%d index=%d value=%.17e\n",
             where, reason, dim, equations, index, double(value));
#if defined(__CUDA_ARCH__)
      asm("trap;");
#elif defined(__HIP_DEVICE_COMPILE__)
      __builtin_trap();
#else
      fflush(stdout);
      MFEM_ABORT("Viscous invariant failed; see VISCOUS_INVARIANT_FAILURE");
#endif
    }

    template<typename Gas>
    MFEM_HOST_DEVICE inline void Layout(const Gas &gas, const char *where)
    {
      const auto &layout = gas.L;
      const int dim = layout.dim;
      // Validate individual fields before arithmetic or any array indexing.
      if (dim < 1 || dim > MAXDIM || layout.num_scalars < 0 ||
          layout.num_scalars > MAXEQ - dim - 2)
        {
          Fail(where, "layout_capacity", dim, -1, layout.num_scalars, 0.0);
        }
      const int equations = layout.nequations();
      if (gas.dim() != dim || gas.num_equations() != equations ||
          layout.eq_mass != 0 || layout.eq_mom0 != 1 ||
          layout.eq_energy != dim + 1 || layout.num_dofs_scalar <= 0 ||
          layout.eq_scalar0 != (layout.num_scalars ? dim + 2 : -1))
        {
          Fail(where, "layout_fields", dim, equations, layout.eq_energy,
               layout.eq_mass);
        }
      for (int d = 0; d < MAXDIM; ++d)
        {
          if (layout.eq_mom[d] != (d < dim ? d + 1 : -1))
            {
              Fail(where, "momentum_index", dim, equations, d, layout.eq_mom[d]);
            }
        }
    }

    template<typename Gas>
    MFEM_HOST_DEVICE inline void Tables(const Gas &, const char *)
    {
    }

    template<typename EOS, typename Transport>
    MFEM_HOST_DEVICE inline void Tables(const LTEGasModel<EOS, Transport> &gas,
                                       const char *where)
    {
      const auto &layout = gas.T.L;
      const auto &view = gas.T.tables;
      if (layout.nx < 2 || layout.ny < 2 || layout.num_properties != 9)
        {
          Fail(where, "table_shape", gas.dim(), gas.num_equations(),
               layout.nx, layout.ny);
        }
      // property_index performs its offset arithmetic in int.
      if (layout.nx > INT_MAX / layout.ny / layout.num_properties)
        {
          Fail(where, "table_index_overflow", gas.dim(), gas.num_equations(),
               layout.nx, layout.ny);
        }
      if (layout.P_idx != 0 || layout.e_idx != 1 || layout.cv_idx != 2 ||
          layout.cp_idx != 3 || layout.R_eq_idx != 4 || layout.gamma_eq_idx != 5 ||
          layout.c_idx != 6 || layout.mu_idx != 7 || layout.lambda_idx != 8)
        {
          Fail(where, "table_property_indices", gas.dim(), gas.num_equations(),
               layout.mu_idx, layout.lambda_idx);
        }
      if (!view.lte_table || !view.inv_table || !view.rho_grid ||
          !view.T_grid || !view.e_grid)
        {
          Fail(where, "null_table_pointer", gas.dim(), gas.num_equations(), -1, 0.0);
        }
    }

    template<typename Gas>
    MFEM_HOST_DEVICE inline void Mass(const Gas &gas,
                                      const mfem::real_t flux[MAXEQ][MAXDIM],
                                      const char *where)
    {
      Layout(gas, where);
      for (int d = 0; d < gas.dim(); ++d)
        {
          // Use literal row zero after validating the canonical layout.
          // != also rejects NaN; no tolerance is appropriate for a zeroed row.
          if (flux[0][d] != 0.0)
            {
              Fail(where, "nonzero_mass_tensor", gas.dim(), gas.num_equations(),
                   d, flux[0][d]);
            }
        }
    }

    template<typename Gas>
    MFEM_HOST_DEVICE inline void Projection(const Gas &gas, mfem::real_t mass,
                                            const char *where)
    {
      if (mass != 0.0)
        {
          Fail(where, "nonzero_mass_projection", gas.dim(), gas.num_equations(),
               0, mass);
        }
    }
  }
}
