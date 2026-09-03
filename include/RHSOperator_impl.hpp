// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace Theseus
{
  template<typename PhysicsT>
  StabilityEstimate RHSOperator<PhysicsT>::EstimateStability(const mfem::Vector &u) const
  {
    const mfem::Vector &pu = this->Prolongate(u);
    const int restricted_size = operator_cache.restr_v->Height();
    if (operator_cache.uVol.Size() != restricted_size)
      {
        operator_cache.uVol.SetSize(restricted_size);
        operator_cache.uVol.UseDevice();
      }
    operator_cache.restr_v->Mult(pu, operator_cache.uVol);

    const mfem::real_t advection_scale = operator_cache.stabilityAdvectionScale;
    const mfem::real_t diffusion_scale = operator_cache.stabilityDiffusionScale;
    const mfem::real_t surface_scale = operator_cache.stabilitySurfaceScale;

    const auto dc = device_cache;
    const auto gas_model = *gas;
    const mfem::real_t *state = operator_cache.uVol.Read();
    const mfem::real_t *jacobian = operator_cache.elJac.Read();
    const mfem::real_t *metric = operator_cache.elMetric.Read();
    const int points = dc.num_elements * dc.ndof_scalar_el;
    const int dofs = dc.ndof_scalar_el;
    const int equations = dc.num_equations;
    const int dimensions = dc.dim;

    mfem::Vector &advective = operator_cache.stabilityAdvectiveRate;
    mfem::Vector &diffusive = operator_cache.stabilityDiffusiveRate;
    if (advective.Size() != points)
      {
        advective.SetSize(points);
        advective.UseDevice();
        diffusive.SetSize(points);
        diffusive.UseDevice();
      }
    mfem::real_t *advective_rate = advective.Write();
    mfem::real_t *diffusive_rate = diffusive.Write();
    mfem::forall(points, [=] MFEM_HOST_DEVICE (int point)
    {
      const int element = point / dofs;
      const int node = point % dofs;
      const mfem::real_t *element_state = state + element * dofs * equations;
      mfem::real_t point_state[MAXEQ];
      Kernels::el_gather_state(element_state, dofs, equations, node, point_state);
      PointStateView S{point_state};
      const mfem::real_t inverse_jacobian = 1.0 / jacobian[point];
      mfem::real_t directional_sum = 0.0;
      mfem::real_t metric_square_sum = 0.0;
      for (int reference_direction = 0; reference_direction < dimensions;
           ++reference_direction)
        {
          mfem::real_t velocity_dot_metric = 0.0;
          mfem::real_t metric_norm_squared = 0.0;
          for (int physical_direction = 0; physical_direction < dimensions;
               ++physical_direction)
            {
              const mfem::real_t value = metric[
                (point * dimensions + reference_direction) * dimensions
                + physical_direction];
              velocity_dot_metric += gas_model.velocity(S, physical_direction) * value;
              metric_norm_squared += value * value;
            }
          directional_sum += (Kernels::rabs(velocity_dot_metric)
                              + gas_model.sound_speed(S)
                                * Kernels::rsqrt(metric_norm_squared))
                             * inverse_jacobian;
          metric_square_sum += metric_norm_squared
                               * inverse_jacobian * inverse_jacobian;
        }
      advective_rate[point] = advection_scale * directional_sum;
#ifdef PARABOLIC
      const mfem::real_t density = gas_model.density(S);
      const mfem::real_t gamma = gas_model.gamma(S);
      const mfem::real_t shear_viscosity = gas_model.viscosity(S);
      const mfem::real_t longitudinal_viscosity =
        mfem::real_t(4.0 / 3.0) * shear_viscosity
        + gas_model.bulk_viscosity(S);
      const mfem::real_t momentum_diffusivity =
        Kernels::rmax(shear_viscosity, longitudinal_viscosity) / density;
      const mfem::real_t thermal_diffusivity =
        gas_model.thermal_conductivity(S) * gamma
        / (density * gas_model.cp(S));
      const mfem::real_t effective_diffusivity =
        Kernels::rmax(momentum_diffusivity, thermal_diffusivity);
      diffusive_rate[point] = diffusion_scale * effective_diffusivity
                              * metric_square_sum;
#else
      diffusive_rate[point] = 0.0;
#endif
    });

    StabilityEstimate estimate;
    const mfem::real_t *advective_host = advective.HostRead();
    const mfem::real_t *diffusive_host = diffusive.HostRead();
    for (int point = 0; point < points; ++point)
      {
        estimate.advective_rate = std::max(estimate.advective_rate,
                                            advective_host[point]);
        estimate.diffusive_rate = std::max(estimate.diffusive_rate,
                                            diffusive_host[point]);
      }

    // Surface corrections carry the endpoint quadrature/Jacobian scaling in
    // fw_minus/fw_plus.  Compute their normal-aligned acoustic rates directly
    // instead of consuming flux-specific wave-speed return values, whose units
    // historically differ between numerical flux implementations.
    const int interior_size = operator_cache.restr_f->Height();
    const int face_points = dc.num_face_points;
    const int interior_points = interior_size / (2 * equations);
    if (interior_points > 0)
      {
        if (operator_cache.uInt.Size() != interior_size)
          {
            operator_cache.uInt.SetSize(interior_size);
            operator_cache.uInt.UseDevice();
          }
        operator_cache.restr_f->Mult(pu, operator_cache.uInt);
        mfem::Vector &surface = operator_cache.stabilitySurfaceRate;
        if (surface.Size() < interior_points)
          {
            surface.SetSize(interior_points);
            surface.UseDevice();
          }
        const mfem::real_t *face_state = operator_cache.uInt.Read();
        const mfem::real_t *normal = dc.nor_d;
        const mfem::real_t *weight_minus = dc.fw_minus_d;
        const mfem::real_t *weight_plus = dc.fw_plus_d;
        mfem::real_t *surface_rate = surface.Write();
        mfem::forall(interior_points, [=] MFEM_HOST_DEVICE (int point)
        {
          const int face = point / face_points;
          const int face_point = point % face_points;
          const int face_size = 2 * face_points * equations;
          const mfem::real_t *states = face_state + face * face_size;
          mfem::real_t minus_state[MAXEQ];
          mfem::real_t plus_state[MAXEQ];
          for (int equation = 0; equation < equations; ++equation)
            {
              minus_state[equation] = states[(0 * equations + equation)
                                             * face_points + face_point];
              plus_state[equation] = states[(1 * equations + equation)
                                            * face_points + face_point];
            }
          PointStateView minus{minus_state};
          PointStateView plus{plus_state};
          mfem::real_t normal_squared = 0.0;
          mfem::real_t minus_normal_velocity = 0.0;
          mfem::real_t plus_normal_velocity = 0.0;
          for (int direction = 0; direction < dimensions; ++direction)
            {
              const mfem::real_t normal_component =
                normal[point * dimensions + direction];
              normal_squared += normal_component * normal_component;
              minus_normal_velocity += gas_model.velocity(minus, direction)
                                       * normal_component;
              plus_normal_velocity += gas_model.velocity(plus, direction)
                                      * normal_component;
            }
          const mfem::real_t normal_magnitude = Kernels::rsqrt(normal_squared);
          const mfem::real_t normal_wave_speed = Kernels::rmax(
            Kernels::rabs(minus_normal_velocity)
              + gas_model.sound_speed(minus) * normal_magnitude,
            Kernels::rabs(plus_normal_velocity)
              + gas_model.sound_speed(plus) * normal_magnitude);
          surface_rate[point] = normal_wave_speed
            * Kernels::rmax(Kernels::rabs(weight_minus[point]),
                            Kernels::rabs(weight_plus[point]))
            * surface_scale;
        });
        const mfem::real_t *surface_host = surface.HostRead();
        for (int point = 0; point < interior_points; ++point)
          estimate.surface_rate = std::max(estimate.surface_rate,
                                            surface_host[point]);
      }

    const int boundary_size = operator_cache.restr_b->Height();
    const int boundary_points = boundary_size / equations;
    if (boundary_points > 0)
      {
        if (operator_cache.uBnd.Size() != boundary_size)
          {
            operator_cache.uBnd.SetSize(boundary_size);
            operator_cache.uBnd.UseDevice();
          }
        operator_cache.restr_b->Mult(pu, operator_cache.uBnd);
        mfem::Vector &surface = operator_cache.stabilitySurfaceRate;
        const int surface_offset = interior_points;
        if (surface.Size() < surface_offset + boundary_points)
          {
            surface.SetSize(surface_offset + boundary_points);
            surface.UseDevice();
          }
        const mfem::real_t *face_state = operator_cache.uBnd.Read();
        const mfem::real_t *normal = dc.bnd_nor_d;
        const mfem::real_t *weight = dc.bnd_wt_d;
        const int *boundary_marker = dc.bnd_marker_index_d;
        const BCDescriptor *boundary_conditions = dc.bc_descr_d;
        const mfem::real_t *boundary_data = dc.bc_vector_d;
        mfem::real_t *surface_rate = surface.Write() + surface_offset;
        mfem::forall(boundary_points, [=] MFEM_HOST_DEVICE (int point)
        {
          const int face = point / face_points;
          const int face_point = point % face_points;
          const int marker = boundary_marker[face];
          if (marker < 0)
            {
              surface_rate[point] = 0.0;
              return;
            }
          const mfem::real_t *states = face_state
            + face * face_points * equations;
          mfem::real_t interior_state[MAXEQ];
          for (int equation = 0; equation < equations; ++equation)
            interior_state[equation] = states[equation * face_points + face_point];
          PointStateView interior{interior_state};
          mfem::real_t normal_squared = 0.0;
          mfem::real_t normal_velocity = 0.0;
          for (int direction = 0; direction < dimensions; ++direction)
            {
              const mfem::real_t normal_component =
                normal[point * dimensions + direction];
              normal_squared += normal_component * normal_component;
              normal_velocity += gas_model.velocity(interior, direction)
                                 * normal_component;
            }
          const mfem::real_t normal_wave_speed =
            Kernels::rabs(normal_velocity)
            + gas_model.sound_speed(interior) * Kernels::rsqrt(normal_squared);
          mfem::real_t boundary_wave_speed = normal_wave_speed;
          const BCDescriptor &condition = boundary_conditions[marker];
          if (condition.type == int(BCType::SupersonicInflow))
            {
              PointStateView exterior{boundary_data + condition.data_index};
              mfem::real_t exterior_normal_velocity = 0.0;
              for (int direction = 0; direction < dimensions; ++direction)
                exterior_normal_velocity += gas_model.velocity(exterior, direction)
                                            * normal[point * dimensions + direction];
              boundary_wave_speed = Kernels::rmax(
                boundary_wave_speed,
                Kernels::rabs(exterior_normal_velocity)
                  + gas_model.sound_speed(exterior) * Kernels::rsqrt(normal_squared));
            }
          surface_rate[point] = boundary_wave_speed * Kernels::rabs(weight[point])
                                * surface_scale;
        });
        const mfem::real_t *surface_host = surface.HostRead() + surface_offset;
        for (int point = 0; point < boundary_points; ++point)
          estimate.surface_rate = std::max(estimate.surface_rate,
                                            surface_host[point]);
      }
    return estimate;
  }


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

    operator_cache.bc_descriptors = bc_descriptors;
    operator_cache.bc_scalar_data = bc_scalar_data;
    operator_cache.bc_vector_data = bc_vector_data;
    ValidateAxisBoundaryGeometry(operator_cache);

#ifdef SUBCELL_FV_BLENDING
    MFEM_VERIFY(indicator, "SUBCELL_FV_BLENDING enabled but indicator is null.");
    BuildPerssonDeviceCache(operator_cache, indicator->ModalBasis());
#endif
    GetDeviceCache(operator_cache, device_cache);
  }

  // pu should be prolongated
  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::FetchRestrictions(const mfem::Vector &pu, mfem::Vector &uVol,
                                                mfem::Vector &uInt, mfem::Vector &uBnd) const
  {
    Theseus::ScopedTimer timer("FetchRestrictions");
    const int psize = operator_cache.restr_v->Height();
    if(uVol.Size() != psize){
      uVol.SetSize(psize);
      uVol.UseDevice();
    }
    {
      Theseus::ScopedTimer vrt("VolumeRestriction");
      operator_cache.restr_v->Mult(pu, uVol);
    }
    const int int_restr_size = operator_cache.restr_f->Height();
    if(uInt.Size() != int_restr_size){
      uInt.SetSize(int_restr_size);
      uInt.UseDevice();
    }
    {
      Theseus::ScopedTimer ifr("InteriorFaceRestriction");
      operator_cache.restr_f->Mult(pu, uInt);
    }
    const int bnd_restr_size = operator_cache.restr_b->Height();
    if(uBnd.Size() != bnd_restr_size){
      uBnd.SetSize(bnd_restr_size);
      uBnd.UseDevice();
    }
    {
      Theseus::ScopedTimer bndr("BoundaryFaceRestriction");
      operator_cache.restr_b->Mult(pu, uBnd);
    }
    operator_cache.u_vol_restr_ready = true;
    operator_cache.u_bnd_restr_ready = true;
    operator_cache.u_int_restr_ready = true;
  }

#ifdef SUBCELL_FV_BLENDING
  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::ComputeIndicatorField(const mfem::Vector &pu) const
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
    mfem::Vector &Ue(operator_cache.uVol);
    if(!operator_cache.u_vol_restr_ready){
      operator_cache.restr_v->Mult(pu, Ue);
      operator_cache.u_vol_restr_ready = true;
    }
    const mfem::real_t *Ue_d = Ue.Read();

    mfem::Vector &indicator_field(operator_cache.indicatorField);
    if(indicator_field.Size() != nval_ind){
      indicator_field.SetSize(nval_ind);
      indicator_field.UseDevice();
    }
    mfem::real_t *ifield_d = indicator_field.Write();

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
  void RHSOperator<PhysicsT>::ComputeBlendingCoefficient() const
  {
    ScopedTimer timer("ComputeBlendingCoeff");
    const mfem::real_t *eta_d = operator_cache.eta.Read();
    mfem::real_t *alpha_d = operator_cache.alpha->Write();
    // operator_cache.alpha_d;
    int ne = operator_cache.num_elements;
    mfem::real_t mthresh = modalThreshold;
    mfem::real_t sharp_fac = sharpness_fac;
    mfem::real_t alpmin = alpha_min;
    mfem::real_t alpmax = alpha_max;
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
      mfem::real_t alpha_dof = \
        1.0 / (1.0 + std::exp(-sharp_fac * (eta_d[e] - mthresh) / mthresh));
      if (alpha_dof < alpmin)
        {
          alpha_dof = 0.0;
        }
      else if (alpha_dof > (1.0 - alpmin))
        {
          alpha_dof = 1.0;
        }
      alpha_d[e] = std::min(alpha_dof, alpmax);
    });
  }

  template<typename PhysicsT>
  void RHSOperator<PhysicsT>::CheckIndicatorSmoothness() const
  {
    Theseus::ScopedTimer timer("CheckIndicatorSmoothness");

    const int ne = operator_cache.num_elements;
    const int ndofs = operator_cache.ndof_scalar_el;
    constexpr int block_size = 256;

    const mfem::real_t *indicator_d = operator_cache.indicatorField.Read();
    const mfem::real_t *modal_d = operator_cache.modal.Read();
    const mfem::real_t *keep_M1_d = operator_cache.keep_M1.Read();
    const mfem::real_t *keep_M2_d = operator_cache.keep_M2.Read();
    mfem::real_t *eta_d = operator_cache.eta.Write();

    // One thread block cooperates on each element.  The previous kernel used
    // one thread per element, leaving the dense ndofs-by-ndofs modal transform
    // entirely serial and severely under-filling accelerators at modest ne.
    mfem::forall_2D(ne, block_size, 1, [=] MFEM_HOST_DEVICE (int e)
    {
      const mfem::real_t *u = indicator_d + e * ndofs;

      MFEM_SHARED mfem::real_t mm_s[block_size];
      MFEM_SHARED mfem::real_t m1m1_s[block_size];
      MFEM_SHARED mfem::real_t m2m2_s[block_size];

      MFEM_FOREACH_THREAD(t, x, block_size)
        {
          mm_s[t] = 0.0;
          m1m1_s[t] = 0.0;
          m2m2_s[t] = 0.0;
        }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(m, x, ndofs)
        {
          mfem::real_t mode = 0.0;
          for (int q = 0; q < ndofs; ++q)
            {
              // Modal data is cached transposed so adjacent threads read adjacent
              // coefficients while each dot product retains its original q order.
              mode += modal_d[q * ndofs + m] * u[q];
            }

          const int t = MFEM_THREAD_ID(x);
          const mfem::real_t mode2 = mode * mode;
          mm_s[t] += mode2;
          m1m1_s[t] += keep_M1_d[m] * mode2;
          m2m2_s[t] += keep_M2_d[m] * mode2;
        }
      MFEM_SYNC_THREAD;

      for (int stride = block_size / 2; stride > 0; stride /= 2)
        {
          MFEM_FOREACH_THREAD(t, x, stride)
            {
              mm_s[t] += mm_s[t + stride];
              m1m1_s[t] += m1m1_s[t + stride];
              m2m2_s[t] += m2m2_s[t + stride];
            }
          MFEM_SYNC_THREAD;
        }

      if (MFEM_THREAD_ID(x) == 0)
        {
          const mfem::real_t mm = mm_s[0];
          const mfem::real_t m1m1 = m1m1_s[0];
          const mfem::real_t m2m2 = m2m2_s[0];
          const mfem::real_t eps = 1.0e-30;
          mfem::real_t val = 0.0;

          if (mm > eps)
            {
              val = 1.0 - m1m1 / mm;
              if (m1m1 > eps)
                {
                  val = Theseus::Kernels::rmax(val, 1.0 - m2m2 / m1m1);
                }
              else
                {
                  val = 1.0;
                }
            }
          eta_d[e] = Theseus::Kernels::rmin(
                                            Theseus::Kernels::rmax(val, 0.0), 1.0);
        }
    });

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
    const mfem::real_t *radius_d = dc.elRadius_d;
    auto gas = dc.gas;

    if(operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);
    if(!operator_cache.u_vol_restr_ready){
      operator_cache.restr_v->Mult(u, Ue);
      operator_cache.u_vol_restr_ready = true;
    }

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
      const mfem::real_t *radius = dc.axisymmetric ?
        radius_d + e * ndof : nullptr;

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

        const mfem::real_t measure = qWgt[ep] *
          AxisymmetricGeometry::MeasureMultiplier(
                                                  dc.axisymmetric, dc.axisymmetric ? radius[ep] : 0.0);
        mass_int += rho * measure;
        ke_int += ke * measure;
        en_int += rhoE * measure;

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
    operator_cache.u_vol_restr_ready = false;
    operator_cache.u_bnd_restr_ready = false;
    operator_cache.u_int_restr_ready = false;
    {
      Theseus::ScopedTimer rhsPrep("RHSRestriction");
      const mfem::Vector &pu = this->Prolongate(u);
      FetchRestrictions(pu, operator_cache.uVol, operator_cache.uInt, operator_cache.uBnd);
      if (this->P)
        {
          if(operator_cache.pdudt.Size() != this->P->Height()){
            operator_cache.pdudt.SetSize(this->P->Height());
          }
        }
    }
    mfem::Vector &pdudt = this->P ? operator_cache.pdudt : dudt;

    // This block is executed by the host
    int nval_restr = operator_cache.restr_v->Height();
    if(operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);
    if(operator_cache.rhsVol.Size() != nval_restr){
      operator_cache.rhsVol.SetSize(nval_restr);
      operator_cache.rhsVol.UseDevice();
    }
    mfem::Vector &dUe(operator_cache.rhsVol);

#ifdef SUBCELL_FV_BLENDING
    {
      Theseus::ScopedTimer timer("SubcellBlendingStep");
      const mfem::Vector &pu = this->Prolongate(u);
      ComputeIndicatorField(pu);
      CheckIndicatorSmoothness();
      ComputeBlendingCoefficient();
    }
#endif

    // Zero on-device
    int psize = pdudt.Size();
    mfem::real_t *pdudt_d = pdudt.Write();
    {
      Theseus::ScopedTimer zerotim("ZeroRHS");
      mfem::forall(psize, [=] MFEM_HOST_DEVICE (int i) { pdudt_d[i] = 0.0; });
    }

    {
      Theseus::ScopedTimer timer("FlowMult");
      FlowMult(u, pdudt);
    }

    if (this->Serial())
      {
        if(this->cP) this->cP->MultTranspose(pdudt, dudt);
      }
    else
      {
        if(this->P) this->P->MultTranspose(pdudt, dudt);
      }

    const int N = this->ess_tdof_list.Size();
    const auto idx = this->ess_tdof_list.Read();
    auto DU_RW = dudt.ReadWrite();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });

    // reset restriction readiness
    operator_cache.u_vol_restr_ready = false;
    operator_cache.u_bnd_restr_ready = false;
    operator_cache.u_int_restr_ready = false;
  }
}
