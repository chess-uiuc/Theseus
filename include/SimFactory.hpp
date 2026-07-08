// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "mfem.hpp"
#include "json.hpp"
#include "PerssonPeraireIndicator.hpp"
#include "RHSOperator.hpp"

namespace Theseus
{

  template <typename GasT, typename InvFluxT>
  struct PhysicsTraits
  {
    using GasModel = GasT;
    using InviscidFlux = InvFluxT;
  };

  // TODO: Update for LTE Tables
  std::unique_ptr<Theseus::RHSOperatorBase>
  MakeRHSOperator(const nlohmann::json& runtime,
                  std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
                  std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
                  std::shared_ptr<mfem::ParMesh> pmesh,
                  std::shared_ptr<mfem::ParGridFunction> eta,
                  std::shared_ptr<mfem::ParGridFunction> alpha,
                  std::vector<std::shared_ptr<mfem::ParGridFunction> > &grad_u,
                  std::shared_ptr<Prandtl::Indicator> indicator,
                  std::shared_ptr<mfem::ParGridFunction> r_gf,
                  mfem::real_t alpha_max);
}

#include "SimFactory_impl.hpp"
