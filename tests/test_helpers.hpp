// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#pragma once
#include "mfem.hpp"
#include "GasState.hpp"

using mfem::real_t;

// Helper to fill a single-DOF conservative state in `U`.
//
// Layout (equation-blocked):
//   eq_mass   -> rho
//   eq_mom[d] -> rho * u_d
//   eq_energy -> rhoE = e_int_density + 0.5 * rho * |u|^2
static void
fill_single_dof_state(Theseus::StateLayout &layout,
                      std::vector<real_t> &U,
                      int dim,
                      real_t rho,
                      const real_t u[3],
                      real_t e_int_density)
{
    const int ndofs = 1;
    const int nscalars = 0;
    const int num_eq = dim+2+nscalars;

    // Zero everything for safety.
    for (int eq = 0; eq < num_eq; ++eq)
    {
        U[layout.index(eq, 0)] = real_t(0);
    }

    // Mass
    U[layout.index(layout.eq_mass, 0)] = rho;

    // Momentum: rho * u_d
    for (int d = 0; d < dim; ++d)
    {
        U[layout.index(layout.eq_mom[d], 0)] = rho * u[d];
    }

    // Kinetic energy density: 0.5 * rho * |u|^2
    real_t vsq = 0;
    for (int d = 0; d < dim; ++d)
    {
        vsq += u[d] * u[d];
    }
    const real_t kinetic = real_t(0.5) * rho * vsq;

    // Total energy density: rhoE = e_int_density + kinetic
    U[layout.index(layout.eq_energy, 0)] = e_int_density + kinetic;
}
