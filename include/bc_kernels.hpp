// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "bc_cache_utilities.hpp"
#include "Flow.hpp"
#include "NavierStokesFlux.hpp"

namespace Theseus {

  namespace BC {

    template<typename GasModelT>
    MFEM_HOST_DEVICE inline void ReflectAxisState(
      const GasModelT &gas, const mfem::real_t *interior,
      mfem::real_t *exterior)
    {
      const int equations = gas.L.nequations();
      for (int equation = 0; equation < equations; ++equation)
        {
          exterior[equation] = interior[equation];
        }
      const int radial_momentum =
        gas.L.eq_mom[AxisymmetricGeometry::radial_coordinate];
      exterior[radial_momentum] = -interior[radial_momentum];
    }


    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE inline
    void ComputeBdrFaceGradFlux(const DeviceCacheT &dc,
                                const Theseus::BCDescriptor &bc,
                                const mfem::real_t *state1,
                                mfem::real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const int neq = dc.num_equations;
      const int dim = dc.dim;
      
      for (int q = 0; q < neq; ++q)
        {
          fluxN[q] = mfem::real_t(0);
        }

      switch (static_cast<Theseus::BCType>(bc.type))
        {
        case Theseus::BCType::PrescribedState:
          {
            mfem::real_t entropy[Theseus::MAXEQ];
            Theseus::PointStateView S{dc.bc_vector_d + bc.data_index};
            Theseus::PointStateViewRW E{entropy};
            gas.entropy_state(S,E);
            for (int q=0;q<neq;++q) fluxN[q]=entropy[q]-state1[q];
            return;
          }

        case Theseus::BCType::NoSlipAdiab:
          {
            const mfem::real_t *vector_data = dc.bc_vector_d;
            const mfem::real_t *Vwall = vector_data + bc.data_index;
            // unused - but be aware
            // const mfem::real_t *wallHeat = vector_data + bc.data_index + dim;

            // Note this must be entropy state
            Theseus::PointStateView S{state1};
            Theseus::PointStateViewRW F{fluxN};

            const mfem::real_t v = -gas.energy(S);

            // Build the same provisional "boundary" state as legacy, then subtract state1.
            F.set_mass(gas.L, gas.mass(S)); // Is this really correct? v+ != v- so.. hrm.
            for(int idim = 0;idim < dim;idim++){
              F.set_momentum(gas.L, idim, Vwall[idim] * v);
            }
            F.set_energy(gas.L, -v);
            for (int q = 0; q < neq; ++q)
              {
                fluxN[q] -= state1[q];
              }
            return;
          }

	case Theseus::BCType::NoSlipIso:
	  {
	    const mfem::real_t *bc_data = dc.bc_vector_d + bc.data_index;
	    const mfem::real_t *Vwall  = bc_data;
	    const mfem::real_t Twall   = bc_data[dim];

	    // state1 is entropy state
	    Theseus::PointStateView S{state1};
	    Theseus::PointStateViewRW F{fluxN};

            // The EOS defines the normalization of the entropy energy variable.
            const mfem::real_t beta_like = gas.isothermal_wall_beta(Twall);

	    F.set_mass(gas.L, gas.mass(S));
	    for (int idim = 0; idim < dim; ++idim)
	      {
		F.set_momentum(gas.L, idim, Vwall[idim] * beta_like);
	      }
	    F.set_energy(gas.L, -beta_like);

	    for (int q = 0; q < neq; ++q)
	      {
		fluxN[q] -= state1[q];
	      }

	    return;
	  }

        case Theseus::BCType::Axis:
          {
            mfem::real_t exterior[Theseus::MAXEQ];
            ReflectAxisState(gas, state1, exterior);
            for (int equation = 0; equation < neq; ++equation)
              {
                fluxN[equation] = exterior[equation] - state1[equation];
              }
            return;
          }

        default:
          {
            // Conservative placeholder for unsupported BCs.
            for (int q = 0; q < neq; ++q)
              {
                fluxN[q] = mfem::real_t(0);
              }
            return;
          }
        }
    }


    template<typename GasModelT>
    MFEM_HOST_DEVICE
    void SlipWallInviscidFluxKernel(const GasModelT &gas, const mfem::real_t *state,
                                    const mfem::real_t *normal, mfem::real_t *flux)
    {
      const auto pressure = Flow::slipwall_pstar_normal(PointStateView{state}, gas, normal);
      for (int eq = 0; eq < gas.num_equations(); ++eq)
        {
          flux[eq] = 0.0;
        }
      for (int d = 0; d < gas.dim(); ++d)
        {
          flux[gas.L.eq_mom0 + d] = pressure*normal[d];
        }
    }

    template<typename GasModelT>
    MFEM_HOST_DEVICE
    void NoSlipAdiabWallFluxKernel(const GasModelT &gas, const mfem::real_t *state,
      const mfem::real_t *grad_x, const mfem::real_t *grad_y, const mfem::real_t *grad_z,
      const mfem::real_t *normal, const mfem::real_t vWall[Theseus::MAXDIM],
      mfem::real_t qWall, bool axisymmetric, mfem::real_t radius, mfem::real_t *flux)
    {
      // Finish wall thermodynamics before producing any viscous output.
      const auto pressure = Flow::slipwall_pstar_normal(PointStateView{state}, gas, normal);
      NavierStokesFlux::ComputeViscousNormalFluxKernel(
        gas, state, grad_x, grad_y, grad_z, normal, flux, axisymmetric, radius);
      mfem::real_t magnitude_squared = 0.0;
      for (int d = 0; d < gas.dim(); ++d)
        {
          magnitude_squared += normal[d]*normal[d];
        }
      mfem::real_t energy_flux = qWall*Kernels::rsqrt(magnitude_squared);
      for (int d = 0; d < gas.dim(); ++d)
        {
          energy_flux += vWall[d]*flux[gas.L.eq_mom0 + d];
        }
      flux[gas.L.eq_energy] = energy_flux;
      for (int eq = 0; eq < gas.num_equations(); ++eq)
        {
          flux[eq] = -flux[eq];
        }
      for (int d = 0; d < gas.dim(); ++d)
        {
          flux[gas.L.eq_mom0 + d] += pressure*normal[d];
        }
    }

    template<typename GasModelT>
    MFEM_HOST_DEVICE
    void NoSlipIsothWallFluxKernel(const GasModelT &gas, const mfem::real_t *state,
      const mfem::real_t *grad_x, const mfem::real_t *grad_y, const mfem::real_t *grad_z,
      const mfem::real_t *normal, const mfem::real_t vWall[Theseus::MAXDIM],
      mfem::real_t tWall, bool axisymmetric, mfem::real_t radius, mfem::real_t *flux)
    {
      // Finish wall thermodynamics before producing any viscous output.
      const auto pressure = Flow::slipwall_pstar_normal(PointStateView{state}, gas, normal);
      NavierStokesFlux::ComputeViscousNormalFluxKernel(
        gas, state, grad_x, grad_y, grad_z, normal, flux, axisymmetric, radius);
      const PointStateView view{state};
      const int energy = gas.L.eq_energy;
      // Keep the trace-conduction convention, including regularization at the axis.
      for (int d = 0; d < gas.dim(); ++d)
        {
          flux[energy] -= gas.velocity(view, d)*flux[gas.L.eq_mom0 + d];
        }
      for (int d = 0; d < gas.dim(); ++d)
        {
          flux[energy] += vWall[d]*flux[gas.L.eq_mom0 + d];
        }
      for (int eq = 0; eq < gas.num_equations(); ++eq)
        {
          flux[eq] = -flux[eq];
        }
      for (int d = 0; d < gas.dim(); ++d)
        {
          flux[gas.L.eq_mom0 + d] += pressure*normal[d];
        }
    }


    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE
    void ApplyBoundaryConditionInviscid(const DeviceCacheT &dc,
                                          const Theseus::BCDescriptor &bc,
                                          const mfem::real_t *state1,
                                          const mfem::real_t *nor,
                                          mfem::real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const mfem::real_t *scalar_data = dc.bc_scalar_d;
      const mfem::real_t *vector_data = dc.bc_vector_d;
      switch (static_cast<Theseus::BCType>(bc.type))
        {
        case Theseus::BCType::SlipWall:
          SlipWallInviscidFluxKernel(gas, state1, nor, fluxN);
          return;
          
        case Theseus::BCType::SupersonicOutflow:
          dc.iflux.ComputeFaceFlux(gas, state1, state1, nor, fluxN);
          return;
          
        case Theseus::BCType::PrescribedState:
        case Theseus::BCType::SupersonicInflow:
          {
            const mfem::real_t *bc_state = vector_data + bc.data_index;
            dc.iflux.ComputeFaceFlux(gas, state1, bc_state, nor, fluxN);
            return;
          }
          
        case Theseus::BCType::Symmetry:
          {
            const int neq = dc.num_equations;
            Theseus::PointStateView S{state1};
            mfem::real_t bc_state[Theseus::MAXEQ];
            for(int ieq = 0;ieq < neq;ieq++){
              bc_state[ieq] = state1[ieq];
            }
            Theseus::PointStateViewRW S2{bc_state};
            const int dim = dc.dim;
            mfem::real_t unorm[Theseus::MAXDIM];
            mfem::real_t mom[Theseus::MAXDIM];
            for(int idim = 0;idim < dim;idim++){
              unorm[idim] = nor[idim];
              mom[idim] = S.momentum(gas.L, idim);
            }
            Theseus::Kernels::Normalize(dim, unorm);
            mfem::real_t nv = Theseus::Kernels::Dot(dim, mom, unorm);
            for(int idim = 0;idim < dim;idim++){
              mfem::real_t mm = -2.0*nv*unorm[idim] + mom[idim];
              S2.set_momentum(gas.L, idim, mm);
            }
            dc.iflux.ComputeFaceFlux(gas, state1, bc_state, nor, fluxN);
            return;
          }
        case Theseus::BCType::Axis:
          {
            mfem::real_t bc_state[Theseus::MAXEQ];
            ReflectAxisState(gas, state1, bc_state);
            dc.iflux.ComputeFaceFlux(gas, state1, bc_state, nor, fluxN);
            return;
          }
        default:
          {
            const int neq = dc.num_equations;
            for (int eq = 0; eq < neq; ++eq) { fluxN[eq] = 0.0; }
            return;
          }
        }
    }

    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE
    void ApplyViscousBoundaryCondition(const DeviceCacheT &dc,
                                         const Theseus::BCDescriptor &bc,
                                         const mfem::real_t *state1,
                                         const mfem::real_t *gradPrim_x,
                                         const mfem::real_t *gradPrim_y,
                                         const mfem::real_t *gradPrim_z,
                                         const mfem::real_t *nor,
                                         mfem::real_t radius,
                                         mfem::real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const int dim = dc.dim;
      const mfem::real_t *scalar_data = dc.bc_scalar_d;
      const mfem::real_t *vector_data = dc.bc_vector_d;
      switch (static_cast<Theseus::BCType>(bc.type))
        {
        case Theseus::BCType::SlipWall:
        case Theseus::BCType::SupersonicInflow:
        case Theseus::BCType::SupersonicOutflow:
        case Theseus::BCType::Symmetry:
          ApplyBoundaryConditionInviscid(dc, bc, state1, nor, fluxN);
          return;

        case Theseus::BCType::PrescribedState:
          {
            ApplyBoundaryConditionInviscid(dc,bc,state1,nor,fluxN);
            mfem::real_t visc[Theseus::MAXEQ][Theseus::MAXDIM];
            NavierStokesFlux::ComputeViscousFluxKernel(
              gas,state1,gradPrim_x,gradPrim_y,gradPrim_z,visc,dc.axisymmetric,radius);
            for(int q=0;q<dc.num_equations;++q)
              for(int d=0;d<dim;++d) fluxN[q]-=visc[q][d]*nor[d];
            return;
          }

        case Theseus::BCType::NoSlipAdiab:
          {
            const mfem::real_t *bc_vec_data = vector_data + bc.data_index;
            mfem::real_t vWall[Theseus::MAXDIM];
            for(int idim=0;idim < dim;idim++){
              vWall[idim] = bc_vec_data[idim];
            }
            const mfem::real_t qWall = bc_vec_data[dim];
            NoSlipAdiabWallFluxKernel(gas, state1, gradPrim_x, gradPrim_y,
                                      gradPrim_z, nor, vWall, qWall,
                                      dc.axisymmetric, radius, fluxN);
            return;
          }
        case Theseus::BCType::NoSlipIso:
          {
            const mfem::real_t *bc_vec_data = vector_data + bc.data_index;
            mfem::real_t vWall[Theseus::MAXDIM];
            for(int idim=0;idim < dim;idim++){
              vWall[idim] = bc_vec_data[idim];
            }
            const mfem::real_t tWall = bc_vec_data[dim];
            NoSlipIsothWallFluxKernel(gas, state1, gradPrim_x, gradPrim_y,
                                      gradPrim_z, nor, vWall, tWall,
                                      dc.axisymmetric, radius, fluxN);
            return;
          }
        case Theseus::BCType::Axis:
          {
            ApplyBoundaryConditionInviscid(dc, bc, state1, nor, fluxN);
            mfem::real_t viscous_flux[Theseus::MAXEQ][Theseus::MAXDIM];
            NavierStokesFlux::ComputeViscousFluxKernel(
              gas, state1, gradPrim_x, gradPrim_y, gradPrim_z,
              viscous_flux, dc.axisymmetric, radius);
            for (int equation = 0; equation < dc.num_equations; ++equation)
              {
                for (int direction = 0; direction < dim; ++direction)
                  {
                    fluxN[equation] -=
                      nor[direction] * viscous_flux[equation][direction];
                  }
              }
            return;
          }
        default:
          {
            const int neq = dc.num_equations;
            for (int eq = 0; eq < neq; ++eq) { fluxN[eq] = 0.0; }
            return;
          }
        }
    }
  }
}
