// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "DGSEMIntegrator.hpp"
#include "ModalBasis.hpp"
#include "Indicator.hpp"
#include "GasModel.hpp"
#include "dgsem_cache_utilities.hpp"
#include "bc_cache_utilities.hpp"

namespace Theseus
{
  class RHSOperatorBase : public mfem::TimeDependentOperator,
                          public mfem::ParNonlinearForm
  {
  protected:
    mutable mfem::real_t max_char_speed;
    std::shared_ptr<mfem::ParFiniteElementSpace> vfes;
    std::shared_ptr<mfem::ParFiniteElementSpace> fes0;
    std::shared_ptr<mfem::ParMesh> pmesh;
    std::shared_ptr<mfem::ParGridFunction> eta;
    std::shared_ptr<mfem::ParGridFunction> r_gf;
    std::shared_ptr<Prandtl::Indicator> indicator;

    const int num_equations, dim, order, num_elements;
    const int num_dofs_scalar;
    const int Ndofs;

    const mfem::real_t sharpness_fac = 9.21024;
    const mfem::real_t modalThreshold;
    const mfem::real_t alpha_min;
    const mfem::real_t alpha_max;

    std::vector<mfem::Array<int>> bdr_marker;
    mfem::Array<Theseus::BCDescriptor> bc_descriptors;
    mfem::Vector bc_vector_data;
    mfem::Vector bc_scalar_data;

    mutable mfem::real_t alpha_dof;
    mutable Theseus::IntegralMeasures diag0;
  public:
    RHSOperatorBase(std::shared_ptr<mfem::ParFiniteElementSpace> vfes_,
                    std::shared_ptr<mfem::ParFiniteElementSpace> fes0_,
                    std::shared_ptr<mfem::ParMesh> pmesh_,
                    std::shared_ptr<mfem::ParGridFunction> eta_,
                    std::shared_ptr<mfem::ParGridFunction> alpha_,
                    std::shared_ptr<Prandtl::Indicator> indicator_,
                    std::shared_ptr<mfem::ParGridFunction> r_gf_ = nullptr,
                    const mfem::real_t alpha_max = 0.5, const mfem::real_t alpha_min = 0.001)
    : mfem::TimeDependentOperator(vfes_->GetTrueVSize()),
      mfem::ParNonlinearForm(vfes_.get()),
      vfes(vfes_), fes0(fes0_), pmesh(pmesh_),
      eta(eta_), r_gf(r_gf_), indicator(indicator_),
      num_equations(vfes->GetVDim()),
      dim(pmesh->SpaceDimension()),
      order(vfes->GetElementOrder(0)),
      num_elements(pmesh->GetNE()),
      num_dofs_scalar(vfes_->GetTrueVSize()/vfes_->GetVDim()),
      Ndofs(vfes->GetFE(0)->GetDof()),
      modalThreshold(0.5 * std::pow(10.0, -1.8 * std::pow(order, 0.25))),
      alpha_min(alpha_min), alpha_max(alpha_max)
    {
      diag0.mass = 0.0;
      diag0.ke = 0.0;
      diag0.en = 0.0;
    }
    virtual ~RHSOperatorBase() = default;
    void SetBCDescriptorData(const mfem::Array<Theseus::BCDescriptor> &bc_descr, const mfem::Vector &bc_scalar_dat,
                             const mfem::Vector &bc_vector_dat)
    {
      bc_descriptors = bc_descr;
      bc_scalar_data = bc_scalar_dat;
      bc_vector_data = bc_vector_dat;
    }

    void AddBdrFaceMarker(mfem::Array<int> &bdr_marker_)
    {
      bdr_marker.push_back(bdr_marker_);
    }

    virtual void Finalize(mfem::real_t time=0)
    {
      SetTime(time);
    }

    IntegralMeasures GetIntegralMeasuresBaseline() const { return diag0; }

    inline mfem::real_t GetMaxCharSpeed() const
    {
      return max_char_speed;
    }

    inline mfem::real_t& GetTimeRef()
    {
      return t;
    }
    virtual void ComputeIntegralMeasures(const mfem::Vector &u, Theseus::IntegralMeasures &diag) const
    { std::cout << "RHSOperatorBase::ComputeIntegralMeasures empty." << std::endl; }
    virtual void Mult(const mfem::Vector &u, mfem::Vector &dudt) const override
    { std::cout << "RHSOperatorBase::Mult empty." << std::endl; }
    virtual std::string GasModelName() const {
      return std::string("RHSOperatorBase::GasModel: NONE");
    }
    virtual std::string NumFluxName() const {
      return std::string("RHSOperatorBase::NumFlux: NONE");
    }
    virtual std::string FlowModelName() const {
      return std::string("RHSOperatorBase::FlowModel: NONE");
    }
    virtual const GasModelInterface& GetGasModelInterface() const
    {
      MFEM_ABORT("RHSOperatorBase::GetGasModelInterface() called on base class.");
    }
  };

  template<typename PhysicsT>
  class RHSOperator : public RHSOperatorBase
  {
  public:
    using Physics = PhysicsT;
    using OperatorCache = DGSEMOperatorCacheT<Physics>;
    using DeviceCache = DGSEMDeviceCacheT<Physics>;
    using Gas = typename Physics::GasModel;
    using InviscidFlux = typename Physics::InviscidFlux;
  protected:
    mutable OperatorCache operator_cache;
    mutable DeviceCache device_cache;
    const std::string gasModelName;
    const std::string numFluxName;
    const std::string flowModelName;
    std::shared_ptr<const Gas> gas;
    std::shared_ptr<const GasModelInterface> gas_interface;
  public:
    RHSOperator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes_,
                std::shared_ptr<mfem::ParFiniteElementSpace> fes0_,
                std::shared_ptr<mfem::ParMesh> pmesh_,
                std::shared_ptr<mfem::ParGridFunction> eta_,
                std::shared_ptr<mfem::ParGridFunction> alpha_,
                std::shared_ptr<Prandtl::Indicator> indicator_,
                std::shared_ptr<const Gas> gas_,
                const std::string &gasModelName_,
                const std::string &numFluxName_,
                const std::string &flowModelName_,
                std::shared_ptr<mfem::ParGridFunction> r_gf_ = nullptr,
                const mfem::real_t alpha_max = 0.5, const mfem::real_t alpha_min = 0.001)
    : RHSOperatorBase(vfes_, fes0_, pmesh_, eta_, alpha_,
                      indicator_, r_gf_, alpha_max, alpha_min),
      gasModelName(gasModelName_), numFluxName(numFluxName_),
      flowModelName(flowModelName_), gas(std::move(gas_)), 
      gas_interface(std::make_shared<Theseus::GasModelInterfaceT<Gas>>(gas))
    {
      operator_cache.gas = *gas;
      operator_cache.alpha = alpha_;
    }

    OperatorCache &GetOperatorCacheReference(){ return operator_cache; };

    virtual ~RHSOperator() = default;
    void Finalize(mfem::real_t time = 0) override;

#ifdef SUBCELL_FV_BLENDING
    void ComputeBlendingCoefficient(const mfem::Vector &u) const;
    void ComputeBlendingCoefficientFromIndicator(const mfem::Vector &indicator_field) const;
    void ComputeIndicatorField(const mfem::Vector &u, mfem::Vector &indicator_field) const;
#endif

    void Mult(const mfem::Vector &u, mfem::Vector &dudt) const override;
    void ComputeIntegralMeasures(const mfem::Vector &u, Theseus::IntegralMeasures &diag) const override;
    virtual mfem::real_t FlowMult(const mfem::Vector &pu, mfem::Vector &pdudt) const = 0;

    std::string GasModelName() const override { return gasModelName; }
    std::string NumFluxName() const override { return numFluxName; }
    std::string FlowModelName() const override { return flowModelName; }

    const Gas &GetGasModel() const { return *gas; };
    const GasModelInterface& GetGasModelInterface() const override
    {
      return *gas_interface;
    }
  };

}

#include "RHSOperator_impl.hpp"
