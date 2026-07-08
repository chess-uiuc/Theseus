// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include <memory>
#include "unit_test.hpp"
#include "GasModel.hpp"
#include "ChandrashekarFlux.hpp"
#include "LaxFriedrichsFlux.hpp"
#include "HLLFlux.hpp"

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

  run_face_flux_consistency_2d(chan, gasModel);
  run_face_flux_consistency_2d(llf,  gasModel);
  run_face_flux_consistency_2d(hll,  gasModel);
  
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

    run_face_flux_normal_reversal_2d(chan, gasModel);
    run_face_flux_normal_reversal_2d(llf,  gasModel);
    run_face_flux_normal_reversal_2d(hll,  gasModel);

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

    run_zero_normal_velocity_pressure_flux_2d(chan, gasModel);
    run_zero_normal_velocity_pressure_flux_2d(llf,  gasModel);
    run_zero_normal_velocity_pressure_flux_2d(hll,  gasModel);

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

    run_face_flux_finite_strong_state_2d(chan, gasModel);
    run_face_flux_finite_strong_state_2d(llf,  gasModel);
    run_face_flux_finite_strong_state_2d(hll,  gasModel);

    return 0;
}
