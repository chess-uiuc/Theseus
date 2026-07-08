// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <GasState.hpp>

// Adapter that uses the new StateLayout + FieldStateView as a "state-like"
// object compatible with run_basic_mass_momentum_energy_test.
class GasStateSemanticsAdapter
{
public:
    using real_t = mfem::real_t;

    GasStateSemanticsAdapter(int dim, int num_dofs_scalar)
        : layout_(dim, num_dofs_scalar),
          data_((dim + 2 + layout_.num_scalars) * num_dofs_scalar),
          view_(data_.data())
    { }

    // --- StateSemantics "contract" API ---

    int dim() const      { return layout_.dim; }
    int num_dofs() const { return layout_.num_dofs_scalar; }

    // Getters
    real_t mass(int i) const
    {
      return view_.mass(layout_, i);
    }

    real_t momentum(int d, int i) const
    {
      return view_.momentum(layout_, d, i);
    }

    real_t energy(int i) const
    {
      return view_.energy(layout_, i);
    }

    // Setters
    void set_mass(int i, real_t rho)
    {
      view_.mass(layout_, i) = rho;
    }

    void set_momentum(int d, int i, real_t rho_u_d)
    {
      view_.momentum(layout_, d, i) = rho_u_d;
    }

    void set_energy(int i, real_t rhoE)
    {
      view_.energy(layout_, i) = rhoE;
    }

private:
    Theseus::StateLayout layout_;
    std::vector<real_t> data_;
    Theseus::FieldStateView view_;
};
