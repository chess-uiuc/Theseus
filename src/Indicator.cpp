// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "Indicator.hpp"

namespace Prandtl
{

  Indicator::Indicator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
                       std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
                       std::shared_ptr<mfem::ParGridFunction> eta)
    : vfes(vfes), fes0(fes0), eta(eta), num_equations(vfes->GetVDim()),
      ndofs(vfes->GetFE(0)->GetDof()), order(vfes->GetElementOrder(0)),
      dim(vfes->GetMesh()->SpaceDimension())
  {
    state.SetSize(num_equations);
    ind_dof.SetSize(1);
  }
 
}
