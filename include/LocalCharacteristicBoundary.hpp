// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "GasModel.hpp"
#include "bc_cache_utilities.hpp"

// Keep this one small state constructor out of CUDA caller kernels; it is
// shared by flux, gradient and CFL assembly. No host inlining restriction.
#if defined(__CUDA_ARCH__)
#define THESEUS_LOCAL_BC_FUNCTION __noinline__
#else
#define THESEUS_LOCAL_BC_FUNCTION inline
#endif

namespace Theseus
{
  namespace BC
  {
    // Physical face target, not HEGEL's reflected finite-volume ghost state.
    // Payload starts with a complete conserved reference state. Total inlet
    // appends p0,T0 and a unit Cartesian flow direction. All input validation
    // takes place during host configuration; no acoustic-path integration.
    template<class Gas>
    MFEM_HOST_DEVICE THESEUS_LOCAL_BC_FUNCTION bool LocalCharacteristicState(
        const Gas &gas, BCType type, const mfem::real_t *interior,
        const mfem::real_t *normal, const mfem::real_t *data, mfem::real_t *target)
    {
      // Host parsing rejects unsupported backends. Compile out this path so
      // existing Helmholtz configurations do not instantiate legacy table code.
      if constexpr (!Gas::supports_local_characteristic)
        {
          return false;
        }
      else
        {
          const auto &L = gas.L;
          PointStateView I{interior};
          PointStateView A{data};
          const auto rho_i = gas.density(I);
          const auto p_i = gas.pressure(I);
          const auto c_i = gas.sound_speed(I);
          const auto impedance = rho_i * c_i;
          mfem::real_t norm2 = 0;
          mfem::real_t qi = 0;
          mfem::real_t qa = 0;
          for (int d = 0; d < L.dim; ++d)
            {
              norm2 += normal[d] * normal[d];
              qi += gas.velocity(I, d) * normal[d];
              qa += gas.velocity(A, d) * normal[d];
            }
          const auto norm = std::sqrt(norm2);
          qi /= norm;
          qa /= norm;
          if (type != BCType::FrozenTotal && qi >= c_i)
            {
              for (int q = 0; q < L.nequations(); ++q)
                {
                  target[q] = interior[q];
                }
              return true;
            }
          const auto p_a = gas.pressure(A);
          auto pressure = p_a;
          mfem::real_t velocity[MAXDIM] = {};
          ThermodynamicState thermo;
          if (type == BCType::FrozenTotal ||
              (type == BCType::CharacteristicTotal && qi <= 0))
            {
              const auto *total = data + L.nequations();
              const auto gamma = gas.gamma(I);
              const auto gm1 = gamma - 1;
              const auto H = gas.R_gas(I) * gas.temperature(I) * gamma / gm1
                             + gas.kinetic_energy_density(I) / rho_i;
              const auto invariant = -qi - 2 * c_i / gm1;
              const auto a = 1 + 2 / gm1;
              const auto b = 2 * invariant;
              const auto c = gm1 * (invariant * invariant / 2 - H);
              const auto discriminant = std::sqrt(b * b - 4 * a * c);
              const auto cb = Kernels::rmax((-b + discriminant) / (2 * a),
                                            (-b - discriminant) / (2 * a));
              const auto speed = 2 * cb / gm1 + invariant;
              const auto factor = 1 + gm1 * (speed * speed / (cb * cb)) / 2;
              pressure = total[0] * std::pow(factor, -gamma / gm1);
              mfem::real_t qb = 0;
              for (int d = 0; d < L.dim; ++d)
                {
                  velocity[d] = speed * total[d + 2];
                  qb += velocity[d] * normal[d] / norm;
                }
              // The frozen total model keeps EOS p,T material for either sign.
              // The existing characteristic total inlet retains its outgoing adaptation.
              if (type != BCType::FrozenTotal && qb >= 0)
                {
                  thermo = gas.boundary_rho_p(rho_i + (pressure - p_i) / (c_i * c_i), pressure);
                  for (int d = 0; d < L.dim; ++d)
                    {
                      velocity[d] = gas.velocity(I, d) + (qb - qi) * normal[d] / norm;
                    }
                }
              else
                {
                  thermo = gas.boundary_p_T(pressure, total[1] / factor);
                }
            }
          else
            {
              if (type == BCType::CharacteristicReference && qi <= 0)
                {
                  pressure = (p_a + p_i + impedance * (qi - qa)) / 2;
                }
              const auto qb = qi - (pressure - p_i) / impedance;
              const bool outgoing = qb >= 0;
              const auto *material = outgoing ? interior : data;
              PointStateView M{material};
              const auto qm = outgoing ? qi : qa;
              for (int d = 0; d < L.dim; ++d)
                {
                  velocity[d] = gas.velocity(M, d) + (qb - qm) * normal[d] / norm;
                }
              if (outgoing || type == BCType::CharacteristicReference)
                {
                  const auto rho = gas.density(M) + (pressure - gas.pressure(M)) / (c_i * c_i);
                  thermo = gas.boundary_rho_p(rho, pressure);
                }
              else
                {
                  thermo = gas.boundary_p_T(pressure, gas.temperature(A));
                }
            }
          PointStateViewRW B{target};
          B.set_mass(L, thermo.density);
          mfem::real_t speed2 = 0;
          mfem::real_t qb = 0;
          for (int d = 0; d < L.dim; ++d)
            {
              B.set_momentum(L, d, thermo.density * velocity[d]);
              speed2 += velocity[d] * velocity[d];
              qb += velocity[d] * normal[d];
            }
          B.set_energy(L, thermo.density * (thermo.specific_internal_energy + speed2 / 2));
          return qb >= 0;
        }
    }
  }
}
