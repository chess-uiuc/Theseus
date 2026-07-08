// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "Physics.hpp"
#include "GasState.hpp"
#include "EOS.hpp"
#include "Transport.hpp"

namespace Theseus
{

  // ============================================================================
  // GasModel: Encapsulate EOS/Transport
  // ============================================================================
  template <typename EOSImpl, typename TransportImpl>
  struct GasModel
  {

    PhysicsConstants phys;
    StateLayout L;
    EOSImpl eos;
    TransportImpl transport;

    MFEM_HOST_DEVICE GasModel() = default;

    MFEM_HOST_DEVICE
    GasModel(const PhysicsConstants &phys_in, const StateLayout &L_in,
             const EOSImpl &eos_in, const TransportImpl &tr_in)
      : phys(phys_in), L(L_in), eos(eos_in), transport(tr_in)
    { };

    MFEM_HOST_DEVICE
    GasModel(const PhysicsConstants &phys_in, const StateLayout &L_in)
      : phys(phys_in), L(L_in)
    { };

    template<typename HostDataT>
    MFEM_HOST_DEVICE GasModel<EOSImpl, TransportImpl> to_device(HostDataT &host_data) {
      GasModel<EOSImpl, TransportImpl> retVal(phys,L,eos,transport);
      return retVal;
    }

    // Utilities and constants etc
    MFEM_HOST_DEVICE
    inline int num_equations() const
    { return L.nequations(); };

    MFEM_HOST_DEVICE
    inline int dim() const
    { return L.dim; };

    // State Access
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t velocity(const StateView &S, int d) const
    { return S.velocity(L,d);};

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t momentum(const StateView &S, int d) const
    { return S.momentum(L,d);};

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t density(const StateView &S) const
    {
      return eos.density(phys, L, S);
    };

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t mass(const StateView &S) const
    {
      return eos.density(phys, L, S);
    };

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t scalar(const StateView &S, int s) const
    {
      return S.scalar(L, s);
    };

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t energy(const StateView &S) const
    {
      return S.energy(L);
    };

    // --- Thermodynamics ------------------------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t pressure(const StateView &S) const
    {
      return eos.pressure(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t gamma(const StateView &S) const
    {
      return eos.gamma(phys, L, S);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t cp(const StateView &S) const
    {
      return eos.cp(phys, L, S);
    }
    
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t R_gas(const StateView &S) const
    {
      return eos.R_gas(phys, L, S);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t temperature(const StateView &S) const
    {
      return eos.temperature(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t sound_speed(const StateView &S) const
    {
      return eos.sound_speed(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t kinetic_energy_density(const StateView &S) const
    {
      return eos.kinetic_energy_density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_from_pressure(const StateView &S, mfem::real_t pressure) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return eos.internal_energy_from_pressure(phys, L, S, pressure);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t specific_internal_energy(const StateView &S) const
    {
      return eos.specific_internal_energy(phys, L, S);
    }
    
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const StateView &S,
                                 const mfem::real_t *grad_r, const mfem::real_t *grad_p,
                                 mfem::real_t *grad_t) const
    {
      return eos.grad_temperature(phys, L, S, grad_r, grad_p, grad_t);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t entropy(const StateView &S)
    {
      return eos.entropy(phys, L, S);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const InStateView &S, OutStateView &E) const
    {
      return eos.entropy_state(phys, L, S, E);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const InStateView &S, const InStateView &dS,
                                          OutStateView &dPrim) const
    {
      return eos.grad_entropy_to_grad_prim(phys, L, S, dS, dPrim);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const InStateView &Se, OutStateView &Sc) const
    {
      return eos.entropy_to_conserved(phys, L, Se, Sc);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void primitive_to_conserved(const InStateView &Sp, OutStateView &Sc) const
    {
      return eos.entropy_to_conserved(phys, L, Sp, Sc);
    }
 
    // --- Transport -----------------------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t viscosity(const StateView &S) const
    {
      return transport.viscosity(phys, L, eos, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t bulk_viscosity(const StateView &S) const
    {
      return transport.bulk_viscosity(phys, L, eos, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t thermal_conductivity(const StateView &S) const
    {
      return transport.thermal_conductivity(phys, L, eos, S);
    }
  };

  using IdealGasModel = GasModel<IdealSingleGasEOS, Transport>;

  // Bridge class interface - helps external callers
  // (e.g. Simulation driver) access essential GasModel
  // functions.
  class GasModelInterface
  {
  public:
    virtual ~GasModelInterface() = default;
    virtual mfem::real_t density(const Theseus::DofStateView &S) const {
      MFEM_ABORT("GasModelInterface::density called on base class.");
    }
    virtual mfem::real_t velocity(const Theseus::DofStateView &S,
                                  int d) const{
      MFEM_ABORT("GasModelInterface::velocity called on base class.");
    }
    virtual mfem::real_t pressure(const Theseus::DofStateView &S) const {
      MFEM_ABORT("GasModelInterface::pressure called on base class.");
    }
    virtual mfem::real_t temperature(const Theseus::DofStateView &S) const {
      MFEM_ABORT("GasModelInterface::temperature called on base class.");
    }
    virtual const Theseus::StateLayout& layout() const {
      MFEM_ABORT("GasModelInterface::layout called on base class.");
    }
    virtual const Theseus::PhysicsConstants& phys() const {
      MFEM_ABORT("GasModelInterface::density called on base class.");
    }
  };

  // Templated wrapper for GasModelInterface
  template <typename GasT>
  class GasModelInterfaceT : public GasModelInterface
  {
  public:
    explicit GasModelInterfaceT(std::shared_ptr<const GasT> gas_)
      : gas(std::move(gas_))
    {}

    mfem::real_t density(const Theseus::DofStateView &S) const override
    {
      return gas->density(S);
    }

    mfem::real_t velocity(const Theseus::DofStateView &S, int d) const override
    {
      return gas->velocity(S, d);
    }

    mfem::real_t pressure(const Theseus::DofStateView &S) const override
    {
      return gas->pressure(S);
    }

    mfem::real_t temperature(const Theseus::DofStateView &S) const override
    {
      return gas->temperature(S);
    }

    const Theseus::StateLayout& layout() const override
    {
      return gas->L;
    }

    const Theseus::PhysicsConstants& phys() const override
    {
      return gas->phys;
    }

  private:
    std::shared_ptr<const GasT> gas;
  };

} // namespace Theseus
