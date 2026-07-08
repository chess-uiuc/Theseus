// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace Theseus
{

  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::Finalize(mfem::real_t time)
  {
    Theseus::ScopedTimer finalize_timer("RHSOperator::Finalize");

    RHSOperatorBase::Finalize(time);
    GetOperatorCache(vfes.get(), &operator_cache);
    AssembleBoundaryFaceGeometryTerms(vfes.get(), bdr_marker, &operator_cache);
#ifdef SUBCELL_FV_BLENDING
    {
      Theseus::ScopedTimer timer("ComputeSubcellMetrics");
      ComputeSubcellMetrics(vfes.get(), &operator_cache);
    }
#endif

    // operator_cache.gas = gasModel; // gasModel *must* be POD
    // operator_cache.alpha = alpha; // Alpha is a driver-owned gridfunc
    operator_cache.bc_descriptors = bc_descriptors;
    operator_cache.bc_scalar_data = bc_scalar_data;
    operator_cache.bc_vector_data = bc_vector_data;

    GetDeviceCache(operator_cache, device_cache);
    //nonlinearForm->SetOperatorCache(&operator_cache);
    //integrator->SetOperatorCache(&operator_cache); 
  }

#ifdef SUBCELL_FV_BLENDING  
  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::ComputeIndicatorField(const mfem::Vector &u,
                                                    mfem::Vector &indicator_field) const
  {
    Theseus::ScopedTimer timer("ComputeIndicator");

    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();
    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int dim = dc.dim;
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int Np_x = dc.Np_x;
    const int Np_y = dc.Np_y;
    const int Np_z = dc.Np_z;

    MFEM_ASSERT(nval_restr == ne*ndof*neq, "Unexpected size for volume restriction in indicator calc.");
    const int nval_ind = nval_restr / neq;

    if(operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }

    if(indicator_field.Size() != nval_ind){
      indicator_field.SetSize(nval_ind);
      indicator_field.UseDevice();
    }

    mfem::Vector &Ue(operator_cache.uVol);
    mfem::real_t *ifield_d = indicator_field.Write();

    operator_cache.restr_v->Mult(u, Ue);
    const mfem::real_t *Ue_d = Ue.Read();
    const int estride = ndof*neq;
    
    // Inside the FORALL below, executed on device
    mfem::forall(nval_ind, [=] MFEM_HOST_DEVICE (int vind)
    {
      const int e = vind / ndof;
      const int evind = vind - e * ndof;
      const mfem::real_t *u_el = Ue_d + e * estride;
      mfem::real_t elstate[Theseus::MAXEQ];
      Theseus::Kernels::el_gather_state(u_el, ndof, neq, evind, elstate);
      Theseus::PointStateView S{elstate};
      ifield_d[vind] = dc.gas.pressure(S) * dc.gas.density(S);
    });

  }

  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::ComputeBlendingCoefficientFromIndicator(const mfem::Vector &indicator_field) const
  {
    {
      Theseus::ScopedTimer timer("CheckIndicatorSmoothness_Host");
      // This is a HOST-only routine.  Make sure the input is host-readable
      indicator->CheckIndicatorSmoothness(indicator_field);
    }
    Theseus::ScopedTimer timer("ComputeAlpha_Host");
    mfem::real_t *alpha_h = operator_cache.alpha->HostWrite();
    const mfem::real_t *eta_h = eta->HostRead();
    for (int el = 0; el < num_elements; el++)
      {
        alpha_dof = 1.0 / (1.0 + std::exp(-sharpness_fac * (eta_h[el] - modalThreshold) / modalThreshold));
        if (alpha_dof < alpha_min)
          {
            alpha_dof = 0.0;
          }
        else if (alpha_dof > (1.0 - alpha_min))
          {
            alpha_dof = 1.0;
          }
        alpha_h[el] = std::min(alpha_dof, alpha_max);
      }
  }
#endif

  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::ComputeIntegralMeasures(const mfem::Vector &u, Theseus::IntegralMeasures &diag) const
  {
    Theseus::ScopedTimer timer("ComputeIntegralMeasures");
    
    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();
    
    // Copy the device cache so that it is not member data
    auto dc = device_cache;
    
    // Device cache parameters
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const mfem::real_t *qWts_d = dc.elQWgts_d;
    auto gas = dc.gas;

    if(operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);

    operator_cache.restr_v->Mult(u, Ue);
    const mfem::real_t *Ue_d = Ue.Read();
    const int estride = ndof*neq;

    mfem::Vector elMass_integral(ne);
    mfem::Vector elKE_integral(ne);
    mfem::Vector elEnergy_integral(ne);
    mfem::Vector elMaxPressure(ne);
    mfem::Vector elMaxTemperature(ne);
    mfem::Vector elMaxDensity(ne);
    mfem::Vector elMinPressure(ne);
    mfem::Vector elMinTemperature(ne);
    mfem::Vector elMinDensity(ne);

    elMass_integral.UseDevice();
    elKE_integral.UseDevice();
    elEnergy_integral.UseDevice();
    elMaxPressure.UseDevice();
    elMaxTemperature.UseDevice();
    elMaxDensity.UseDevice();
    elMinPressure.UseDevice();
    elMinTemperature.UseDevice();
    elMinDensity.UseDevice();

    mfem::real_t *elMass_int_d = elMass_integral.Write();
    mfem::real_t *elKE_int_d = elKE_integral.Write();
    mfem::real_t *elEnergy_int_d = elEnergy_integral.Write();

    mfem::real_t *elPress_max_d = elMaxPressure.Write();
    mfem::real_t *elTemp_max_d = elMaxTemperature.Write();
    mfem::real_t *elDens_max_d = elMaxDensity.Write();
    mfem::real_t *elPress_min_d = elMinPressure.Write();
    mfem::real_t *elTemp_min_d = elMinTemperature.Write();
    mfem::real_t *elDens_min_d = elMinDensity.Write();

    // Inside the FORALL below, executed on device
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
      const mfem::real_t *u_el = Ue_d + e * estride;
      const mfem::real_t *qWgt = qWts_d + e * ndof;
   
      mfem::real_t mass_int = 0.0;
      mfem::real_t ke_int = 0.0;
      mfem::real_t en_int = 0.0;
      mfem::real_t min_dens = 1e32;
      mfem::real_t max_dens = 0.0;
      mfem::real_t min_temp = 1e32;
      mfem::real_t max_temp = 0.0;
      mfem::real_t min_press = 1e32;
      mfem::real_t max_press = 0.0;

      for(int ep = 0;ep < ndof;ep++){
        mfem::real_t elstate[Theseus::MAXEQ];
        Theseus::Kernels::el_gather_state(u_el, ndof, neq, ep, elstate);
        Theseus::PointStateView S{elstate};

        mfem::real_t rho = gas.density(S);
        mfem::real_t ke = gas.kinetic_energy_density(S);
        mfem::real_t rhoE = gas.energy(S); // energy density
        mfem::real_t press = gas.pressure(S);
        mfem::real_t temper = gas.temperature(S);

        mass_int += rho * qWgt[ep];
        ke_int += ke * qWgt[ep];
        en_int += rhoE * qWgt[ep];

        min_temp = Theseus::Kernels::rmin(min_temp, temper);
        max_temp = Theseus::Kernels::rmax(max_temp, temper);
        min_dens = Theseus::Kernels::rmin(min_dens, rho);
        max_dens = Theseus::Kernels::rmax(max_dens, rho);
        min_press = Theseus::Kernels::rmin(min_press, press);
        max_press = Theseus::Kernels::rmax(max_press, press);
      }

      elMass_int_d[e]   = mass_int;
      elKE_int_d[e]     = ke_int;
      elEnergy_int_d[e] = en_int;
      elPress_max_d[e]  = max_press;
      elPress_min_d[e]  = min_press;
      elDens_max_d[e]   = max_dens;
      elDens_min_d[e]   = min_dens;
      elTemp_min_d[e]   = min_temp;
      elTemp_max_d[e]   = max_temp;

    });

    // diag.mass = mfem::Sum(elMass_integral);
    // diag.ke = mfem::Sum(elKE_integral);
    // diag.en = mfem::Sum(elEnergy_integral);
    diag.mass = 0.0;
    diag.ke   = 0.0;
    diag.en   = 0.0;
    diag.min_press = 1e32;
    diag.max_press = 0.0;
    diag.min_dens = 1e32;
    diag.max_dens = 0.0;
    diag.min_temp = 1e32;
    diag.max_temp = 0.0;

    const mfem::real_t *mass_h = elMass_integral.HostRead();
    const mfem::real_t *ke_h   = elKE_integral.HostRead();
    const mfem::real_t *en_h   = elEnergy_integral.HostRead();
    const mfem::real_t *minpress_h = elMinPressure.HostRead();
    const mfem::real_t *maxpress_h = elMaxPressure.HostRead();
    const mfem::real_t *mindens_h = elMinDensity.HostRead();
    const mfem::real_t *maxdens_h = elMaxDensity.HostRead();
    const mfem::real_t *mintemp_h = elMinTemperature.HostRead();
    const mfem::real_t *maxtemp_h = elMaxTemperature.HostRead();

    for (int e = 0; e < ne; ++e) {
      diag.mass += mass_h[e];
      diag.ke   += ke_h[e];
      diag.en   += en_h[e];
      diag.min_press = Theseus::Kernels::rmin(diag.min_press, minpress_h[e]);
      diag.max_press = Theseus::Kernels::rmax(diag.max_press, maxpress_h[e]);
      diag.min_temp = Theseus::Kernels::rmin(diag.min_temp, mintemp_h[e]);
      diag.max_temp = Theseus::Kernels::rmax(diag.max_temp, maxtemp_h[e]);
      diag.min_dens = Theseus::Kernels::rmin(diag.min_dens, mindens_h[e]);
      diag.max_dens = Theseus::Kernels::rmax(diag.max_dens, maxdens_h[e]);
    }

    mfem::real_t sendbuf[3] = {diag.mass, diag.ke, diag.en};
    mfem::real_t recvbuf[3] = {0.0, 0.0, 0.0};

    MPI_Allreduce(sendbuf, recvbuf, 3, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, pmesh->GetComm());

    diag.mass = recvbuf[0];
    diag.ke = recvbuf[1];
    diag.en = recvbuf[2];

    sendbuf[0] = diag.min_press;
    sendbuf[1] = diag.min_temp;
    sendbuf[2] = diag.min_dens;

    MPI_Allreduce(sendbuf, recvbuf, 3, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, pmesh->GetComm());

    diag.min_press = recvbuf[0];
    diag.min_temp = recvbuf[1];
    diag.min_dens = recvbuf[2];

    sendbuf[0] = diag.max_press;
    sendbuf[1] = diag.max_temp;
    sendbuf[2] = diag.max_dens;

    MPI_Allreduce(sendbuf, recvbuf, 3, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, pmesh->GetComm());

    diag.max_press = recvbuf[0];
    diag.max_temp = recvbuf[1];
    diag.max_dens = recvbuf[2];

    if(diag0.mass == 0.0){
      diag0 = diag;
    }

  }
  
  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::Mult(const mfem::Vector &u, mfem::Vector &dudt) const
  {
    Theseus::ScopedTimer timer("RHSMult");
    
#ifdef SUBCELL_FV_BLENDING
    {
      Theseus::ScopedTimer timer("SubcellBlendingStep");
      mfem::Vector indicator_field;
      // Since the CV are on-device, and computing
      // the indicator requires the CV, we compute
      // the indicator on-device and xfer only it and
      // alpha (the blending coeff) from host/device.
      ComputeIndicatorField(u, indicator_field);
      ComputeBlendingCoefficientFromIndicator(indicator_field);
    }
#endif

    // max_char_speed is consumed by external components
    // between steps
    max_char_speed = FlowMult(u, dudt);

  }

}
