// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"

namespace Prandtl
{

  class Indicator
  {
  protected:
    std::shared_ptr<mfem::ParFiniteElementSpace> vfes;
    std::shared_ptr<mfem::ParFiniteElementSpace> fes0;
    std::shared_ptr<mfem::ParGridFunction> eta;

    mfem::Array<int> vdof_indices, ind_indx;
    mfem::Vector el_vdofs, ind_dof;
    int num_equations, ndofs, order, dim;
    mfem::Vector state;
  public:
    Indicator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
              std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
              std::shared_ptr<mfem::ParGridFunction> eta);
    virtual void CheckIndicatorSmoothness(const mfem::Vector &indicator) = 0;
    virtual ~Indicator() = default;
  };
}
