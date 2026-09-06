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
  class NSOperator : public Theseus::RHSOperator<PhysicsT>
  {
  public:
    using Physics = PhysicsT;
    using Base = Theseus::RHSOperator<Physics>;
    using OperatorCache = typename Base::OperatorCache;
    using DeviceCache = typename Base::DeviceCache;
    using Gas = typename Base::Gas;
    using InviscidFlux = typename Base::InviscidFlux;
  protected:
    using Base::operator_cache;
    using Base::device_cache;
  private:
    std::vector<std::shared_ptr<mfem::ParGridFunction> > grad_u;
  public:
    NSOperator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes_,
               std::shared_ptr<mfem::ParFiniteElementSpace> fes0_,
               std::shared_ptr<mfem::ParMesh> pmesh_,
               std::shared_ptr<mfem::ParGridFunction> eta_,
               std::shared_ptr<mfem::ParGridFunction> alpha_,
               std::vector<std::shared_ptr<mfem::ParGridFunction> > &grad_u_,
               std::shared_ptr<Prandtl::PerssonPeraireIndicator> indicator_,
               std::shared_ptr<const Gas> gasModel_,
               const std::string &gasModelName_,
               const std::string &numFluxName_,
               const mfem::real_t alpha_max=0.5, const mfem::real_t alpha_min=0.001)
    : RHSOperator<Physics>(vfes_, fes0_, pmesh_, eta_, alpha_,
                           indicator_, gasModel_, gasModelName_, numFluxName_,
                           "NavierStokes", alpha_max, alpha_min),
      grad_u(grad_u_)
    {}

    // Top level RHS routine
    void FlowMult(const mfem::Vector &u, mfem::Vector &dudt) const override;

    // Gradient Operator Interface (BR1 aux rhs)
    void GradOperator(const mfem::Vector &u, std::vector<mfem::Vector *> &grad_u) const;
    void GradOperator_Volume(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperator_InteriorFaces(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperator_BoundaryFaces(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;

    void ComputeEntropyState(const mfem::Vector &u, mfem::Vector &e) const;
    void ComputeGradPrimFromGradEntropy(const mfem::Vector &u,
                                        std::vector<mfem::Vector *> &gradEntropy) const;

    // NavierStokes RHS Interface
    void MultCNS(const mfem::Vector &u,
                 const std::vector<mfem::Vector *> &grad_prim,
                 mfem::Vector &pdudt) const;
    void MultCNS_Volume(const mfem::Vector &pu,
                        const std::vector<mfem::Vector *> &p_grad_prim,
                        mfem::Vector &pdudt) const;
    void MultCNS_InteriorFaces(const mfem::Vector &pu,
                               const std::vector<mfem::Vector *> &p_grad_prim,
                               mfem::Vector &pdudt) const;
    void MultCNS_BoundaryFaces(const mfem::Vector &pu,
                               const std::vector<mfem::Vector *> &p_grad_prim,
                               mfem::Vector &pdudt) const;
  };

}

#include "NSOperator_impl.hpp"
