// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include <memory>
#include "unit_test.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"
#include "ChandrashekarFlux.hpp"
#include "RanochaFlux.hpp"
#include "PirozzoliFlux.hpp"
#include "LaxFriedrichsFlux.hpp"
#include "HLLFlux.hpp"
#include "RoeFlux.hpp"

#include <cmath>

using real_t = mfem::real_t;

static void set_state_2d(mfem::Vector &q,
                         real_t rho,
                         real_t u,
                         real_t v,
                         real_t p,
                         real_t gamma)
{
    q.SetSize(4);

    const real_t ke = 0.5 * rho * (u*u + v*v);
    const real_t E  = p / (gamma - 1.0) + ke;

    q(0) = rho;
    q(1) = rho * u;
    q(2) = rho * v;
    q(3) = E;
}

static void compute_physical_normal_flux_2d(const mfem::Vector &q,
                                            const mfem::Vector &nor,
                                            real_t gamma,
                                            mfem::Vector &flux)
{
    flux.SetSize(4);

    const real_t rho = q(0);
    const real_t u   = q(1) / rho;
    const real_t v   = q(2) / rho;
    const real_t E   = q(3);

    const real_t vel2 = u*u + v*v;
    const real_t p = (gamma - 1.0) * (E - 0.5 * rho * vel2);

    const real_t nx = nor(0);
    const real_t ny = nor(1);
    const real_t un = u*nx + v*ny;

    flux(0) = rho * un;
    flux(1) = rho * u * un + p * nx;
    flux(2) = rho * v * un + p * ny;
    flux(3) = (E + p) * un;
}

template <typename FluxT, typename GasT>
static void run_face_flux_consistency_2d(const FluxT &num_flux,
                                         const GasT &gas)
{
    mfem::Vector q(4), nor(2), flux(4), expected(4);

    set_state_2d(q, 1.2, 31.0, -7.0, 101325.0, gas.phys.gamma);

    nor(0) = 0.6;
    nor(1) = 0.8;

    num_flux.ComputeFaceFlux(gas, q.HostRead(), q.HostRead(), nor.HostRead(),
			     flux.HostWrite());
    compute_physical_normal_flux_2d(q, nor, gas.phys.gamma, expected);

    for (int eq = 0; eq < 4; ++eq)
    {
        EXPECT_CLOSE(flux(eq), expected(eq), 1.0e-9);
    }
}

template <typename FluxT, typename GasT>
static void run_face_flux_normal_reversal_2d(const FluxT &num_flux,
                                             const GasT &gas)
{
    mfem::Vector qL(4), qR(4), n(2), minus_n(2);
    mfem::Vector flux_n(4), flux_minus_n_swapped(4);
    double gamma = gas.phys.gamma;

    set_state_2d(qL, 1.0,  20.0,  3.0, 100000.0, gamma);
    set_state_2d(qR, 0.8, -10.0, -2.0,  90000.0, gamma);

    n(0) = 1.0;
    n(1) = 0.0;

    minus_n(0) = -n(0);
    minus_n(1) = -n(1);

    num_flux.ComputeFaceFlux(gas, qL.HostRead(), qR.HostRead(), n.HostRead(),
			     flux_n.HostWrite());
    num_flux.ComputeFaceFlux(gas, qR.HostRead(), qL.HostRead(), minus_n.HostRead(),
			     flux_minus_n_swapped.HostWrite());

    for (int eq = 0; eq < 4; ++eq)
    {
      const real_t lhs = flux_n(eq);
      const real_t rhs = -flux_minus_n_swapped(eq);
      const real_t scale = std::max<real_t>(1.0, std::max(std::abs(lhs), std::abs(rhs)));
      EXPECT_CLOSE(flux_n(eq), -flux_minus_n_swapped(eq), 1.0e-12*scale);
    }
}

template <typename FluxT, typename GasT>
static void run_zero_normal_velocity_pressure_flux_2d(const FluxT &num_flux,
                                                      const GasT  &gas)
{
    mfem::Vector q(4), n(2), flux(4);
    double gamma = gas.phys.gamma;
    // Velocity tangential to x-normal face.
    set_state_2d(q, 1.0, 0.0, 12.0, 100000.0, gamma);

    n(0) = 1.0;
    n(1) = 0.0;

    num_flux.ComputeFaceFlux(gas, q.HostRead(), q.HostRead(), n.HostRead(),
			     flux.HostWrite());

    EXPECT_CLOSE(flux(0),      0.0, 1.0e-12);
    EXPECT_CLOSE(flux(1), 100000.0, 1.0e-8);
    EXPECT_CLOSE(flux(2),      0.0, 1.0e-12);
    EXPECT_CLOSE(flux(3),      0.0, 1.0e-8);
}

template <typename FluxT, typename GasT>
static void run_face_flux_finite_strong_state_2d(const FluxT &num_flux,
                                                 const GasT &gas)
{
    mfem::Vector qL(4), qR(4), n(2), flux(4);

    set_state_2d(qL, 1.0,  800.0,  50.0, 101325.0, gas.phys.gamma);
    set_state_2d(qR, 0.2, -300.0, -20.0,  20000.0, gas.phys.gamma);

    n(0) = 0.6;
    n(1) = 0.8;

    num_flux.ComputeFaceFlux(gas, qL.HostRead(),
			     qR.HostRead(), n.HostRead(), flux.HostWrite());

    for (int eq = 0; eq < 4; ++eq)
    {
        EXPECT_TRUE(std::isfinite(flux(eq)));
    }
}

template <typename FluxT, typename GasT>
static void run_vol_flux_tadmor_condition_2d(const FluxT &num_flux,
                                             const GasT &gas)
{
    mfem::Vector qL(4), qR(4), met1(3), met2(3), flux(4);
    mfem::Vector wL(4), wR(4);
    double gamma = gas.phys.gamma;

    set_state_2d(qL, 1.0,        20.0, 3.0, 100000.0, gamma);
    set_state_2d(qR, 1.000001,  -10.0,-2.0,  90000.0, gamma);
    Theseus::PointStateView SL(qL.HostRead());
    Theseus::PointStateView SR(qR.HostRead());
    Theseus::PointStateViewRW WL(wL.HostWrite());
    Theseus::PointStateViewRW WR(wR.HostWrite());
    gas.entropy_state(SL, WL);
    gas.entropy_state(SR, WR);

    for(unsigned int d=0; d<2; d++)
    {
      met1 = 0.0;
      met2 = 0.0;
      met1(d) = 1.0;
      met2(d) = 1.0;
      num_flux.ComputeVolumeFlux(gas, qL.HostRead(), qR.HostRead(), met1.HostRead(), met2.HostRead(), flux.HostWrite());

      mfem::real_t psi_L = gas.density(SL)*gas.velocity(SL, d);
      mfem::real_t psi_R = gas.density(SR)*gas.velocity(SR, d);
      mfem::real_t RHS = psi_R - psi_L;
      mfem::real_t LHS = 0.0;
      for(int e=0; e<4; e++)
      {
        LHS += flux(e) * (wR(e) - wL(e));
      }
      EXPECT_CLOSE(LHS, RHS, 1e-13);
    }
}

TEST(RiemannFlux_Consistency_2D)
{
  const int dim = 2;
  const int ndofs = 1;

  Theseus::PhysicsConstants phys(
                                 /* gamma = */ 1.4,
                                 /* Pr    = */ 0.72,
                                 /* R_gas = */ 287.05,
                                 /* mu    = */ 0.02);
  
  Theseus::StateLayout layout(dim, ndofs);
  Theseus::IdealGasModel gasModel(phys, layout);
  
  Theseus::ChandrashekarFlux::InviscidFlux chan;
  Theseus::LaxFriedrichsFlux::InviscidFlux llf;
  Theseus::HLLFlux::InviscidFlux           hll;
  Theseus::RoeFlux::InviscidFlux           roe;
  Theseus::RanochaFlux::InviscidFlux       ran;
  Theseus::PirozzoliFlux::InviscidFlux      pi;

  run_face_flux_consistency_2d(chan, gasModel);
  run_face_flux_consistency_2d(llf,  gasModel);
  run_face_flux_consistency_2d(hll,  gasModel);
  run_face_flux_consistency_2d(roe,  gasModel);
  run_face_flux_consistency_2d(ran,  gasModel);
  run_face_flux_consistency_2d(pi,   gasModel);
  
    return 0;
}

TEST(RiemannFlux_NormalReversal_2D)
{
    const int dim = 2;
    const int ndofs = 1;

    Theseus::PhysicsConstants phys(1.4, 0.72, 287.05, 0.02);
    Theseus::StateLayout layout(dim, ndofs);
    Theseus::IdealGasModel gasModel(phys, layout);

    Theseus::ChandrashekarFlux::InviscidFlux chan;
    Theseus::LaxFriedrichsFlux::InviscidFlux llf;
    Theseus::HLLFlux::InviscidFlux           hll;
    Theseus::RoeFlux::InviscidFlux           roe;
    Theseus::RanochaFlux::InviscidFlux       ran;
    Theseus::PirozzoliFlux::InviscidFlux      pi;

    run_face_flux_normal_reversal_2d(chan, gasModel);
    run_face_flux_normal_reversal_2d(llf,  gasModel);
    run_face_flux_normal_reversal_2d(hll,  gasModel);
    run_face_flux_normal_reversal_2d(roe,  gasModel);
    run_face_flux_normal_reversal_2d(ran,  gasModel);
    run_face_flux_normal_reversal_2d(pi,   gasModel);

    return 0;
}

TEST(RiemannFlux_ZeroNormalVelocityPressureFlux_2D)
{
    const int dim = 2;
    const int ndofs = 1;

    Theseus::PhysicsConstants phys(1.4, 0.72, 287.05, 0.02);
    Theseus::StateLayout layout(dim, ndofs);
    Theseus::IdealGasModel gasModel(phys, layout);

    Theseus::ChandrashekarFlux::InviscidFlux chan;
    Theseus::LaxFriedrichsFlux::InviscidFlux llf;
    Theseus::HLLFlux::InviscidFlux           hll;
    Theseus::RoeFlux::InviscidFlux           roe;
    Theseus::RanochaFlux::InviscidFlux       ran;
    Theseus::PirozzoliFlux::InviscidFlux      pi;

    run_zero_normal_velocity_pressure_flux_2d(chan, gasModel);
    run_zero_normal_velocity_pressure_flux_2d(llf,  gasModel);
    run_zero_normal_velocity_pressure_flux_2d(hll,  gasModel);
    run_zero_normal_velocity_pressure_flux_2d(roe,  gasModel);
    run_zero_normal_velocity_pressure_flux_2d(ran,  gasModel);
    run_zero_normal_velocity_pressure_flux_2d(pi,   gasModel);

    return 0;
}

TEST(RiemannFlux_FiniteStrongState_2D)
{
    const int dim = 2;
    const int ndofs = 1;

    Theseus::PhysicsConstants phys(1.4, 0.72, 287.05, 0.02);
    Theseus::StateLayout layout(dim, ndofs);
    Theseus::IdealGasModel gasModel(phys, layout);

    Theseus::ChandrashekarFlux::InviscidFlux chan;
    Theseus::LaxFriedrichsFlux::InviscidFlux llf;
    Theseus::HLLFlux::InviscidFlux           hll;
    Theseus::RoeFlux::InviscidFlux           roe;
    Theseus::RanochaFlux::InviscidFlux       ran;
    Theseus::PirozzoliFlux::InviscidFlux      pi;

    run_face_flux_finite_strong_state_2d(chan, gasModel);
    run_face_flux_finite_strong_state_2d(llf,  gasModel);
    run_face_flux_finite_strong_state_2d(hll,  gasModel);
    run_face_flux_finite_strong_state_2d(roe,  gasModel);
    run_face_flux_finite_strong_state_2d(ran,  gasModel);
    run_face_flux_finite_strong_state_2d(pi,   gasModel);

    return 0;
}

TEST(ChandrashekarFlux_DissipationIgnoresCommonTangentialVelocity_2D)
{
    Theseus::PhysicsConstants phys(1.4, 0.72, 287.05, 0.02);
    Theseus::StateLayout layout(2, 1);
    Theseus::IdealGasModel gasModel(phys, layout);
    Theseus::ChandrashekarFlux::InviscidFlux flux_operator;
    mfem::Vector qL(4), qR(4), qL_boosted(4), qR_boosted(4), normal(2);
    mfem::Vector flux(4), boosted_flux(4);

    set_state_2d(qL, 1.1, 12.0, 0.0, 95000.0, phys.gamma);
    set_state_2d(qR, 0.9, -7.0, 0.0, 88000.0, phys.gamma);
    set_state_2d(qL_boosted, 1.1, 12.0, 1200.0, 95000.0, phys.gamma);
    set_state_2d(qR_boosted, 0.9, -7.0, 1200.0, 88000.0, phys.gamma);
    normal(0) = 1.0;
    normal(1) = 0.0;

    flux_operator.ComputeFaceFlux(gasModel, qL.HostRead(), qR.HostRead(),
                                  normal.HostRead(), flux.HostWrite());
    flux_operator.ComputeFaceFlux(
      gasModel, qL_boosted.HostRead(), qR_boosted.HostRead(),
      normal.HostRead(), boosted_flux.HostWrite());

    // A common tangential boost changes other conserved flux components but
    // must not change the mass dissipation across this face.
    EXPECT_CLOSE(flux(0), boosted_flux(0), 1.0e-11);

    return 0;
}

TEST(RoeFlux_ResolvesStationaryContact_2D)
{
    Theseus::PhysicsConstants phys(1.4, 0.72, 287.05, 0.02);
    Theseus::StateLayout layout(2, 1);
    Theseus::IdealGasModel gasModel(phys, layout);
    Theseus::RoeFlux::InviscidFlux roe;
    mfem::Vector qL(4), qR(4), n(2), flux(4);

    // Equal pressure and velocity with a density jump is a pure contact wave.
    set_state_2d(qL, 1.0, 0.0, 7.0, 100000.0, phys.gamma);
    set_state_2d(qR, 2.0, 0.0, 7.0, 100000.0, phys.gamma);
    n(0) = 1.0;
    n(1) = 0.0;

    roe.ComputeFaceFlux(gasModel, qL.HostRead(), qR.HostRead(),
                        n.HostRead(), flux.HostWrite());

    EXPECT_CLOSE(flux(0), 0.0, 1.0e-12);
    EXPECT_CLOSE(flux(1), 100000.0, 1.0e-8);
    EXPECT_CLOSE(flux(2), 0.0, 1.0e-12);
    EXPECT_CLOSE(flux(3), 0.0, 1.0e-8);
    return 0;
}

TEST(ECFlux_Tadmor_2D)
{
  const int dim =2;
  const int ndofs = 1;

  Theseus::PhysicsConstants phys(1.4, 0.72, 287.05, 0.02);

  Theseus::StateLayout layout(dim, ndofs);
  Theseus::IdealGasModel gasModel(phys, layout);

  Theseus::ChandrashekarFlux::InviscidFlux chan;
  Theseus::RanochaFlux::InviscidFlux ran;

  run_vol_flux_tadmor_condition_2d(chan, gasModel);
  run_vol_flux_tadmor_condition_2d(ran, gasModel);
  return 0;
}

namespace
{
  template<typename GasT>
  void CheckLLFNormalFluxAgainstTensorReference(const GasT &gas,
                                               const mfem::Vector &left,
                                               const mfem::Vector &right,
                                               const mfem::Vector &normal)
  {
    const int equations = gas.num_equations();
    const int dimension = gas.dim();
    const mfem::Vector saved_left(left);
    const mfem::Vector saved_right(right);
    mfem::real_t physical_left[Theseus::MAXEQ][Theseus::MAXDIM] = {};
    mfem::real_t physical_right[Theseus::MAXEQ][Theseus::MAXDIM] = {};
    mfem::Vector actual(equations);

    Theseus::NavierStokesFlux::ComputeInviscidFluxKernel(
      gas, left.HostRead(), physical_left);
    Theseus::NavierStokesFlux::ComputeInviscidFluxKernel(
      gas, right.HostRead(), physical_right);
    const auto wave_speed = Theseus::NavierStokesFlux::MaximumNormalWaveSpeed(
      gas, left.HostRead(), right.HostRead(), normal.HostRead());
    Theseus::LaxFriedrichsFlux::InviscidFlux{}.ComputeFaceFlux(
      gas, left.HostRead(), right.HostRead(), normal.HostRead(), actual.HostWrite());

    const auto sample_left = Theseus::NavierStokesFlux::MakeInviscidNormalFluxState(
      gas, left.HostRead());
    const auto sample_right = Theseus::NavierStokesFlux::MakeInviscidNormalFluxState(
      gas, right.HostRead());
    mfem::Vector hll_actual(equations);
    Theseus::HLLFlux::InviscidFlux{}.ComputeFaceFlux(
      gas, left.HostRead(), right.HostRead(), normal.HostRead(), hll_actual.HostWrite());
    mfem::real_t un_left = 0.0, un_right = 0.0, normal_squared = 0.0;
    for (int d = 0; d < dimension; ++d)
      {
        un_left += gas.velocity(Theseus::PointStateView{left.HostRead()}, d) * normal[d];
        un_right += gas.velocity(Theseus::PointStateView{right.HostRead()}, d) * normal[d];
        normal_squared += normal[d] * normal[d];
      }
    const auto normal_length = Theseus::Kernels::rsqrt(normal_squared);
    const auto c_left = gas.sound_speed(Theseus::PointStateView{left.HostRead()}) * normal_length;
    const auto c_right = gas.sound_speed(Theseus::PointStateView{right.HostRead()}) * normal_length;
    const auto sl = std::min(un_left - c_left, un_right - c_right);
    const auto sr = std::max(un_left + c_left, un_right + c_right);
    for (int equation = 0; equation < equations; ++equation)
      {
        mfem::real_t projected_left = 0.0;
        mfem::real_t projected_right = 0.0;
        for (int direction = 0; direction < dimension; ++direction)
          {
            projected_left += physical_left[equation][direction] * normal[direction];
            projected_right += physical_right[equation][direction] * normal[direction];
          }
        EXPECT_CLOSE(Theseus::NavierStokesFlux::ComputeInviscidNormalFluxEquation(
                       gas, sample_left, normal.HostRead(), equation), projected_left,
                     1e-12 * std::max<mfem::real_t>(1.0, std::abs(projected_left)));
        EXPECT_CLOSE(Theseus::NavierStokesFlux::ComputeInviscidNormalFluxEquation(
                       gas, sample_right, normal.HostRead(), equation), projected_right,
                     1e-12 * std::max<mfem::real_t>(1.0, std::abs(projected_right)));
        const auto expected = 0.5 * (projected_left + projected_right) -
          0.5 * wave_speed * (right[equation] - left[equation]);
        EXPECT_CLOSE(actual[equation], expected,
                     1e-12 * std::max<mfem::real_t>(1.0, std::abs(expected)));
        const auto hll_expected = sl >= 0.0 ? projected_left :
          (sr <= 0.0 ? projected_right :
           (sr * projected_left - sl * projected_right +
            sl * sr * (right[equation] - left[equation])) / (sr - sl));
        EXPECT_CLOSE(hll_actual[equation], hll_expected,
                     1e-12 * std::max<mfem::real_t>(1.0, std::abs(hll_expected)));
        EXPECT_EQ(left[equation], saved_left[equation]);
        EXPECT_EQ(right[equation], saved_right[equation]);
      }
  }

  void PackLLFReferenceState(const Theseus::StateLayout &layout,
                            mfem::real_t density,
                            mfem::real_t internal_energy,
                            const mfem::real_t velocity[3],
                            mfem::Vector &state)
  {
    state.SetSize(layout.nequations());
    state = 0.0;
    state[layout.eq_mass] = density;
    mfem::real_t kinetic_energy = 0.0;
    for (int direction = 0; direction < layout.dim; ++direction)
      {
        state[layout.eq_mom[direction]] = density * velocity[direction];
        kinetic_energy += 0.5 * density * velocity[direction] * velocity[direction];
      }
    state[layout.eq_energy] = density * internal_energy + kinetic_energy;
    for (int scalar = 0; scalar < layout.num_scalars; ++scalar)
      {
        state[layout.eq_scalar0 + scalar] = density * 0.2;
      }
  }

  template<typename GasT>
  void CheckLLFFaceReferenceCases(const GasT &gas,
                                 mfem::real_t internal_energy_left,
                                 mfem::real_t internal_energy_right)
  {
    const mfem::real_t velocity_left[3] = {13.0, -7.0, 4.0};
    const mfem::real_t velocity_right[3] = {-9.0, 6.0, -2.0};
    const mfem::real_t velocity_rest[3] = {0.0, 0.0, 0.0};
    const mfem::real_t normal_components[3] = {0.3, -0.7, 0.2};
    mfem::Vector left, right, normal(gas.dim());
    PackLLFReferenceState(gas.L, 0.3, internal_energy_left, velocity_left, left);
    PackLLFReferenceState(gas.L, 0.8, internal_energy_right, velocity_right, right);
    for (int direction = 0; direction < gas.dim(); ++direction)
      {
        normal[direction] = normal_components[direction];
      }
    CheckLLFNormalFluxAgainstTensorReference(gas, left, right, normal);
    CheckLLFNormalFluxAgainstTensorReference(gas, left, left, normal);

    // A reflected trace with a small, non-unit coordinate normal.
    right = left;
    const int reflected_direction = gas.dim() - 1;
    right[gas.L.eq_mom[reflected_direction]] =
      -left[gas.L.eq_mom[reflected_direction]];
    normal = 0.0;
    normal[reflected_direction] = -0.0004;
    CheckLLFNormalFluxAgainstTensorReference(gas, left, right, normal);

    PackLLFReferenceState(gas.L, 0.3, internal_energy_left, velocity_rest, left);
    CheckLLFNormalFluxAgainstTensorReference(gas, left, left, normal);
    normal = 0.0;
    CheckLLFNormalFluxAgainstTensorReference(gas, left, left, normal);
  }

  struct LLFAnalyticLTETables
  {
    Theseus::LTETable::Data storage;
    Theseus::LTETable::LTETables tables{3, 3};

    LLFAnalyticLTETables()
    {
      storage.rho_grid.SetSize(3);
      storage.T_grid.SetSize(3);
      storage.e_grid.SetSize(3);
      storage.lte_table.SetSize(81);
      storage.inv_table.SetSize(9);
      const mfem::real_t temperatures[3] = {200.0, 400.0, 800.0};
      for (int index = 0; index < 3; ++index)
        {
          storage.rho_grid[index] = 0.1 + 0.5 * index;
          storage.T_grid[index] = temperatures[index];
          storage.e_grid[index] = 700.0 * temperatures[index];
        }
      for (int temperature_index = 0; temperature_index < 3; ++temperature_index)
        {
          const auto temperature = storage.T_grid[temperature_index];
          for (int density_index = 0; density_index < 3; ++density_index)
            {
              const auto density = storage.rho_grid[density_index];
              const mfem::real_t properties[9] = {
                density * 287.05 * temperature, 700.0 * temperature,
                700.0, 980.0, 287.05, 1.4,
                std::sqrt(1.4 * 287.05 * temperature), 1.8e-5, 0.025};
              for (int property = 0; property < 9; ++property)
                {
                  storage.lte_table[tables.L.property_index(
                    property, density_index, temperature_index)] = properties[property];
                }
              storage.inv_table[temperature_index * 3 + density_index] = temperature;
            }
        }
      tables.tables = {storage.lte_table.HostRead(), storage.inv_table.HostRead(),
                       storage.rho_grid.HostRead(), storage.T_grid.HostRead(),
                       storage.e_grid.HostRead()};
    }
  };
}

TEST(LLFFace_NormalAccumulationMatchesTensorReference_CPG)
{
  for (int dimension = 1; dimension <= 3; ++dimension)
    {
      const Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 0.02);
      const Theseus::StateLayout layout(dimension, 1);
      const Theseus::IdealGasModel gas(physics, layout);
      CheckLLFFaceReferenceCases(gas, 200000.0, 300000.0);
    }
  return 0;
}

TEST(LLFFace_NormalAccumulationMatchesTensorReference_LTE)
{
  LLFAnalyticLTETables fixture;
  for (int dimension = 1; dimension <= 3; ++dimension)
    {
      const Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 0.02);
      const Theseus::StateLayout layout(dimension, 1);
      const Theseus::LTEGasModel<Theseus::LTEGasEOS, Theseus::LTETransport>
        gas(physics, layout, fixture.tables);
      CheckLLFFaceReferenceCases(gas, 210000.0, 315000.0);
    }
  return 0;
}

TEST(LLFFace_NormalAccumulationPreservesPassiveScalarFlux)
{
  const Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 0.02);
  const Theseus::StateLayout layout(2, 1, 1);
  const Theseus::IdealGasModel gas(physics, layout);
  CheckLLFFaceReferenceCases(gas, 200000.0, 300000.0);
  return 0;
}

namespace RoeTensorReference
{
  using namespace Theseus;
    template<typename GasModelT>
    MFEM_HOST_DEVICE inline static void
    ComputeTensorReferenceFaceFlux(const GasModelT &gas,
                          const mfem::real_t *qL,
                          const mfem::real_t *qR,
                          const mfem::real_t *nor,
                          mfem::real_t *flux)
    {
      const int dim = gas.dim();
      const int neq = gas.num_equations();
      const int eq_mass = gas.L.eq_mass;
      const int eq_mom0 = gas.L.eq_mom0;
      const int eq_energy = gas.L.eq_energy;
      PointStateView SL{qL};
      PointStateView SR{qR};

      mfem::real_t fL[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t fR[Theseus::MAXEQ][Theseus::MAXDIM];
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, qL, fL);
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, qR, fR);

      mfem::real_t nor_mag2 = 0.0;
      for (int d = 0; d < dim; ++d) { nor_mag2 += nor[d] * nor[d]; }
      const mfem::real_t nor_mag = Kernels::rsqrt(nor_mag2);
      const mfem::real_t inv_nor_mag = 1.0 / nor_mag;

      const mfem::real_t rhoL = gas.density(SL);
      const mfem::real_t rhoR = gas.density(SR);
      const mfem::real_t pL = gas.pressure(SL);
      const mfem::real_t pR = gas.pressure(SR);
      const mfem::real_t rootL = std::sqrt(rhoL);
      const mfem::real_t rootR = std::sqrt(rhoR);
      const mfem::real_t root_sum_inv = 1.0 / (rootL + rootR);
      const mfem::real_t rho_roe = rootL * rootR;
      const mfem::real_t HL = (gas.energy(SL) + pL) / rhoL;
      const mfem::real_t HR = (gas.energy(SR) + pR) / rhoR;
      const mfem::real_t H = (rootL * HL + rootR * HR) * root_sum_inv;

      mfem::real_t u[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t du[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t du_t[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t nhat[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t u2 = 0.0;
      mfem::real_t un = 0.0;
      mfem::real_t dun = 0.0;
      for (int d = 0; d < dim; ++d)
        {
          const mfem::real_t uL = gas.velocity(SL, d);
          const mfem::real_t uR = gas.velocity(SR, d);
          nhat[d] = nor[d] * inv_nor_mag;
          u[d] = (rootL * uL + rootR * uR) * root_sum_inv;
          du[d] = uR - uL;
          u2 += u[d] * u[d];
          un += u[d] * nhat[d];
          dun += du[d] * nhat[d];
        }

      const mfem::real_t gamma = gas.gamma(SL);
      const mfem::real_t a2 = (gamma - 1.0) * (H - 0.5 * u2);
      const mfem::real_t a = std::sqrt(a2);
      const mfem::real_t dp = pR - pL;
      const mfem::real_t drho = rhoR - rhoL;
      const mfem::real_t alpha_minus = 0.5 * (dp - rho_roe * a * dun) / a2;
      const mfem::real_t alpha_plus  = 0.5 * (dp + rho_roe * a * dun) / a2;
      const mfem::real_t alpha_zero  = drho - dp / a2;
      const mfem::real_t lambda_minus = Kernels::rabs(un - a) * nor_mag;
      const mfem::real_t lambda_zero  = Kernels::rabs(un) * nor_mag;
      const mfem::real_t lambda_plus  = Kernels::rabs(un + a) * nor_mag;

      for (int d = 0; d < dim; ++d) { du_t[d] = du[d] - dun * nhat[d]; }

      const mfem::real_t dmass = lambda_minus * alpha_minus
        + lambda_zero * alpha_zero
        + lambda_plus * alpha_plus;
      mfem::real_t diss[Theseus::MAXEQ] = {0.0};
      diss[eq_mass] = dmass;
      for (int d = 0; d < dim; ++d)
        {
          diss[eq_mom0 + d] =
            lambda_minus * alpha_minus * (u[d] - a * nhat[d])
            + lambda_zero * (alpha_zero * u[d] + rho_roe * du_t[d])
            + lambda_plus * alpha_plus * (u[d] + a * nhat[d]);
        }
      mfem::real_t u_dot_du_t = 0.0;
      for (int d = 0; d < dim; ++d) { u_dot_du_t += u[d] * du_t[d]; }
      diss[eq_energy] =
        lambda_minus * alpha_minus * (H - a * un)
        + lambda_zero * (0.5 * alpha_zero * u2 + rho_roe * u_dot_du_t)
        + lambda_plus * alpha_plus * (H + a * un);

      // Passive scalars share the contact eigenvalue.  Their acoustic share is
      // carried with the Roe-averaged mass fraction.
      for (int s = 0; s < gas.L.num_scalars; ++s)
        {
          const int eq = gas.L.eq_scalar0 + s;
          const mfem::real_t y_roe =
            (rootL * qL[eq] / rhoL + rootR * qR[eq] / rhoR) * root_sum_inv;
          diss[eq] = y_roe * dmass
            + lambda_zero * ((qR[eq] - qL[eq]) - y_roe * drho);
        }

      for (int eq = 0; eq < neq; ++eq)
        {
          mfem::real_t central = 0.0;
          for (int d = 0; d < dim; ++d)
            {
              central += 0.5 * (fL[eq][d] + fR[eq][d]) * nor[d];
            }
          flux[eq] = central - 0.5 * diss[eq];
        }

    }


}

TEST(RoeFace_NormalAccumulationMatchesTensorReference_CPG)
{
  for (int dimension = 1; dimension <= 3; ++dimension)
    {
      const Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 0.02);
      const Theseus::StateLayout layout(dimension, 1);
      const Theseus::IdealGasModel gas(physics, layout);
      const mfem::real_t vl[3] = {13.0, -7.0, 4.0};
      const mfem::real_t vr[3] = {-9.0, 6.0, -2.0};
      const mfem::real_t directions[3] = {0.3, -0.7, 0.2};
      mfem::Vector left, right, normal(dimension), actual(layout.nequations()),
        expected(layout.nequations());
      PackLLFReferenceState(layout, 0.3, 200000.0, vl, left);
      PackLLFReferenceState(layout, 0.8, 300000.0, vr, right);
      for (int d = 0; d < dimension; ++d) { normal[d] = directions[d]; }
      for (int sample = 0; sample < 3; ++sample)
        {
          if (sample == 1) { right = left; }
          if (sample == 2)
            {
              right[layout.eq_mom[dimension - 1]] *= -1.0;
              normal = 0.0;
              normal[dimension - 1] = -0.0004;
            }
          Theseus::RoeFlux::InviscidFlux{}.ComputeFaceFlux(
            gas, left.HostRead(), right.HostRead(), normal.HostRead(), actual.HostWrite());
          RoeTensorReference::ComputeTensorReferenceFaceFlux(
            gas, left.HostRead(), right.HostRead(), normal.HostRead(), expected.HostWrite());
          for (int q = 0; q < layout.nequations(); ++q)
            {
              EXPECT_CLOSE(actual[q], expected[q],
                           1e-12 * std::max<mfem::real_t>(1.0, std::abs(expected[q])));
            }
        }
    }
  return 0;
}

TEST(HLLFace_SupersonicBranchesMatchTensorReference)
{
  LLFAnalyticLTETables fixture;
  for (int dimension = 1; dimension <= 3; ++dimension)
    {
      const Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 0.02);
      const Theseus::StateLayout layout(dimension, 1);
      const Theseus::IdealGasModel cpg(physics, layout);
      const Theseus::LTEGasModel<Theseus::LTEGasEOS, Theseus::LTETransport>
        lte(physics, layout, fixture.tables);
      for (const mfem::real_t sign : {-1.0, 1.0})
        {
          const mfem::real_t vl[3] = {sign * 1800.0, -7.0, 4.0};
          const mfem::real_t vr[3] = {sign * 1500.0, 6.0, -2.0};
          mfem::Vector left, right, normal(dimension);
          normal = 0.0;
          normal[0] = 0.25;
          PackLLFReferenceState(layout, 0.3, 210000.0, vl, left);
          PackLLFReferenceState(layout, 0.8, 315000.0, vr, right);
          for (const auto *state : {left.HostRead(), right.HostRead()})
            {
              const Theseus::PointStateView view{state};
              EXPECT_TRUE(std::abs(cpg.velocity(view, 0)) > cpg.sound_speed(view));
              EXPECT_TRUE(std::abs(lte.velocity(view, 0)) > lte.sound_speed(view));
            }
          CheckLLFNormalFluxAgainstTensorReference(cpg, left, right, normal);
          CheckLLFNormalFluxAgainstTensorReference(lte, left, right, normal);
        }
    }
  return 0;
}

TEST(RoeFace_NormalAccumulationPreservesPassiveScalarFlux)
{
  const Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 0.02);
  const Theseus::StateLayout layout(2, 1, 1);
  const Theseus::IdealGasModel gas(physics, layout);
  const mfem::real_t vl[3] = {13.0, -7.0, 0.0};
  const mfem::real_t vr[3] = {-9.0, 6.0, 0.0};
  mfem::Vector left, right, normal(2), actual(layout.nequations()),
    expected(layout.nequations());
  PackLLFReferenceState(layout, 0.3, 200000.0, vl, left);
  PackLLFReferenceState(layout, 0.8, 300000.0, vr, right);
  right[layout.eq_scalar0] = 0.8 * 0.7;
  const mfem::Vector saved_left(left);
  const mfem::Vector saved_right(right);
  normal[0] = 0.3;
  normal[1] = -0.7;
  Theseus::RoeFlux::InviscidFlux{}.ComputeFaceFlux(
    gas, left.HostRead(), right.HostRead(), normal.HostRead(), actual.HostWrite());
  RoeTensorReference::ComputeTensorReferenceFaceFlux(
    gas, left.HostRead(), right.HostRead(), normal.HostRead(), expected.HostWrite());
  for (int q = 0; q < layout.nequations(); ++q)
    {
      EXPECT_CLOSE(actual[q], expected[q],
                   1e-12 * std::max<mfem::real_t>(1.0, std::abs(expected[q])));
      EXPECT_EQ(left[q], saved_left[q]);
      EXPECT_EQ(right[q], saved_right[q]);
    }
  return 0;
}
