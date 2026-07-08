// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "mfem.hpp"

// Adapter for the legacy/Prandtl conservative state layout:
// U(eq, dof) = U[eq * num_dofs_scalar + dof]
//   eq = 0: mass      (rho)
//   eq = 1..dim: momentum components (rho*u_x, rho*u_y, rho*u_z)
//   eq = dim+1: energy (rhoE)
class LegacyConservativeState
{
public:
    using real_t = double;  // or whatever MFEM uses

    LegacyConservativeState(int dim, int num_dofs_scalar)
        : dim_(dim),
          ndofs_(num_dofs_scalar),
          num_eq_(dim_ + 2),
          U_(num_eq_ * ndofs_)
    { }

    int dim() const      { return dim_; }
    int num_dofs() const { return ndofs_; }

    // Getters
    real_t mass(int i) const {
        return U_[0 * ndofs_ + i];
    }

    real_t momentum(int d, int i) const {
        // d in [0, dim_)
        return U_[(1 + d) * ndofs_ + i];
    }

    real_t energy(int i) const {
        return U_[(dim_ + 1) * ndofs_ + i];
    }

    // Setters
    void set_mass(int i, real_t rho) {
        U_[0 * ndofs_ + i] = rho;
    }

    void set_momentum(int d, int i, real_t rho_u_d) {
        U_[(1 + d) * ndofs_ + i] = rho_u_d;
    }

    void set_energy(int i, real_t rhoE) {
        U_[(dim_ + 1) * ndofs_ + i] = rhoE;
    }

    // (Optional) direct access to underlying storage if you ever need it:
    mfem::Vector & data()       { return U_; }
    const mfem::Vector & data() const { return U_; }

private:
    int dim_;
    int ndofs_;
    int num_eq_;

    // This matches the raw sol_state layout used in Simulation.cpp:
    //   sol_state[eq * num_dofs_scalar + i]
    mfem::Vector U_;
};
