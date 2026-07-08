// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "RHSOperator.hpp"

namespace Theseus
{

  template<typename PhysicsT>
  class EulerOperator : public RHSOperator<PhysicsT>
                       
  {
  public:
    using Physics = PhysicsT;
    using Base = RHSOperator<Physics>;
    using OperatorCache = typename Base::OperatorCache;
    using DeviceCache = typename Base::DeviceCache;
    using Gas = typename Base::Gas;
    using InviscidFlux = typename Base::InviscidFlux;
  protected:
    using Base::operator_cache;
    using Base::device_cache;
  public:

    EulerOperator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes_,
                  std::shared_ptr<mfem::ParFiniteElementSpace> fes0_,
                  std::shared_ptr<mfem::ParMesh> pmesh_,
                  std::shared_ptr<mfem::ParGridFunction> eta_,
                  std::shared_ptr<mfem::ParGridFunction> alpha_,
                  std::shared_ptr<Prandtl::Indicator> indicator_,
                  std::shared_ptr<const Gas> gasModel_,
                  const std::string &gasModelName_,
                  const std::string &numFluxName_,
                  std::shared_ptr<mfem::ParGridFunction> r_gf_ = nullptr,
                  const mfem::real_t alpha_max = 0.5, const mfem::real_t alpha_min = 0.001)
    : RHSOperator<PhysicsT>(vfes_, fes0_, pmesh_, eta_, alpha_, indicator_,
                            gasModel_, gasModelName_, numFluxName_,
                            "Euler", r_gf_, alpha_max, alpha_min)
    {}

    ~EulerOperator() = default;
    mfem::real_t FlowMult(const mfem::Vector &pu, mfem::Vector &pdudt) const override;
    mfem::real_t MultEuler_Volume(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    mfem::real_t MultEuler_InteriorFaces(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    mfem::real_t MultEuler_BoundaryFaces(const mfem::Vector &pu, mfem::Vector &pdudt) const;

  };
  
}

#include "EulerOperator_impl.hpp"
