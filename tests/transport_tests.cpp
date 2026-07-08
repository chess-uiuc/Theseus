// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "unit_test.hpp"
#include "mfem.hpp"
#include "Physics.hpp"
#include "Transport.hpp"
#include "EOS.hpp"

#include <cmath>

using mfem::real_t;
using Theseus::PhysicsConstants;
using Theseus::Transport;
using Theseus::IdealSingleGasEOS;
using Theseus::PointStateView;
using Theseus::StateLayout;

TEST(Transport_mu_kappa)
{
    // Reasonably physical values (exact numbers not crucial for the test)
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(gamma, Pr, R_gas, mu);

    IdealSingleGasEOS eos;
    Transport transport;
    StateLayout layout{3,1};
    real_t rho = 1.0;
    real_t rhoVx = 0.0;
    real_t rhoVy = 0.0;
    real_t rhoVz = 0.0;
    real_t rhoE = 1.0/(gamma-1.0);

    // Two distinct temperatures to probe temperature dependence
    const real_t T1 = phys->T0;
    const real_t T2 = 2.0 * phys->T0;

    const real_t cp  = phys->cp;
    const real_t tol = 1.0e-12;

    real_t state_data_T1[5] = {rho, rhoVx, rhoVy, rhoVz, T1*R_gas/(gamma-1.0)};
    real_t state_data_T2[5] = {rho, rhoVx, rhoVy, rhoVz, T2*R_gas/(gamma-1.0)};
    PointStateView S1{state_data_T1};
    PointStateView S2{state_data_T2};

#ifdef SUTHERLAND
    // -------------------------------------------------------------------------
    // Sutherland-law behavior
    // -------------------------------------------------------------------------
    auto sutherland_mu = [&](real_t T) -> real_t {
        const real_t Trel  = T / phys->T0;
        const real_t T0pTs = phys->T0 + phys->Ts;
        // Match the implementation in Transport::viscosity
        return phys->mu0 * T0pTs * Trel * std::sqrt(Trel) / (T + phys->Ts);
    };

    const real_t mu1_expected = sutherland_mu(T1);
    const real_t mu2_expected = sutherland_mu(T2);

    const real_t mu1 = transport.viscosity(*phys, layout, eos, S1);
    const real_t mu2 = transport.viscosity(*phys, layout, eos, S2);

    // Check viscosity matches Sutherland formula
    EXPECT_CLOSE(mu1, mu1_expected, tol);
    EXPECT_CLOSE(mu2, mu2_expected, tol);

    // In Sutherland mode, viscosity should *not* be constant in T
    EXPECT_TRUE(std::abs(mu1 - mu2) > 1.0e-16);

    // Thermal conductivity: k = mu(T) * cp / Pr
    const real_t k1_expected = mu1_expected * cp * phys->PrInverse;
    const real_t k2_expected = mu2_expected * cp * phys->PrInverse;

    const real_t k1 = transport.thermal_conductivity(*phys, layout, eos, S1);
    const real_t k2 = transport.thermal_conductivity(*phys, layout, eos, S2);

    EXPECT_CLOSE(k1, k1_expected, tol);
    EXPECT_CLOSE(k2, k2_expected, tol);

#else
    // -------------------------------------------------------------------------
    // Constant-transport behavior
    // -------------------------------------------------------------------------
    const real_t mu1 = transport.viscosity(*phys, layout, eos, S1);
    const real_t mu2 = transport.viscosity(*phys, layout, eos, S2);

    // Viscosity should be equal to phys->mu for any T
    EXPECT_CLOSE(mu1, phys->mu, tol);
    EXPECT_CLOSE(mu2, phys->mu, tol);
    EXPECT_CLOSE(mu1, mu2,    tol);

    // Thermal conductivity: constant k = mu * cp / Pr
    const real_t k_expected = phys->mu * cp * phys->PrInverse;

    const real_t k1 = transport.thermal_conductivity(*phys, layout, eos, S1);
    const real_t k2 = transport.thermal_conductivity(*phys, layout, eos, S2);

    EXPECT_CLOSE(k1, k_expected, tol);
    EXPECT_CLOSE(k2, k_expected, tol);
#endif

    return 0;
}
