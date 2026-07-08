// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#pragma once

// State-like objects must provide:
//   int dim() const;
//   int num_dofs() const;
//   double mass(int i) const;
//   double momentum(int d, int i) const;
//   double energy(int i) const;
//   void set_mass(int i, double);
//   void set_momentum(int d, int i, double);
//   void set_energy(int i, double);

template <class StateLike>
void run_basic_mass_momentum_energy_test(StateLike &state)
{
    const int dim   = state.dim();
    const int ndofs = state.num_dofs();

    // Initialize with deterministic but nontrivial values
    for (int i = 0; i < ndofs; ++i) {
        double rho     = 10.0 + i;
        double rhoE    = 1000.0 + i;

        state.set_mass(i, rho);
        state.set_energy(i, rhoE);

        for (int d = 0; d < dim; ++d) {
            double rho_u_d = 100.0 * (d + 1) + i;
            state.set_momentum(d, i, rho_u_d);
        }
    }

    // Read back and verify
    for (int i = 0; i < ndofs; ++i) {
        double expected_rho  = 10.0 + i;
        double expected_rhoE = 1000.0 + i;

        EXPECT_CLOSE(state.mass(i),   expected_rho,  1e-14);
        EXPECT_CLOSE(state.energy(i), expected_rhoE, 1e-14);

        for (int d = 0; d < dim; ++d) {
            double expected_rho_u_d = 100.0 * (d + 1) + i;
            EXPECT_CLOSE(state.momentum(d, i), expected_rho_u_d, 1e-14);
        }
    }
}
