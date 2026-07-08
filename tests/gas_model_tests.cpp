// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "unit_test.hpp"
#include "test_helpers.hpp"

#include "GasModel.hpp"

#include <vector>
#include <cmath>

using namespace Theseus;

// -----------------------------------------------------------------------------
// GasModel thermo test: pressure, temperature, sound-speed, density, and
// specific internal energy for an ideal single-species gas in dim = 1,2,3.
//
// For each dimension:
//  - Choose rho, u, and internal energy density e_int such that
//        e_int = 1 / (gamma - 1)
//
//    so that, for the ideal gas, the pressure should be
//        p = (gamma - 1) * e_int = 1
//
//  - Check:
//       * gas.pressure(S) == 1
//       * gas.temperature(S) == p / (rho * R_gas)
//       * gas.sound_speed(S) == sqrt(gamma * p / rho)
//       * gas.density(S) == rho
//       * gas.specific_internal_energy(S) == e_int / rho
// -----------------------------------------------------------------------------
TEST(GasModel_IdealGas_EOS)
{
    // Reasonably physical values
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(gamma, Pr, R_gas, mu);

    const real_t tol = 1.0e-12;

    for (int dim = 1; dim <= 3; ++dim)
    {
        const int ndofs = 1;
        StateLayout layout(dim, ndofs);  // no scalars
        IdealGasModel gas(*phys, layout);

        const int num_eq = layout.eq_energy + 1; // dim+2

        std::vector<real_t> U(num_eq * ndofs);

        // Fixed density.
        const real_t rho = 2.0;

        // Velocity (only first 'dim' components matter).
        const real_t u[3] = {10.0, -3.0, 5.0};

        // Internal energy density chosen so that p = 1, regardless of u.
        //
        // For ideal gas with total energy density:
        //   rhoE = e_int + 0.5 * rho * |u|^2
        //
        // p = (gamma - 1) * (rhoE - 0.5 * rho * |u|^2)
        //   = (gamma - 1) * e_int
        //
        // So choose e_int = 1 / (gamma - 1) => p = 1.
        const real_t e_int_density = 1.0 / (gamma - 1.0);

        fill_single_dof_state(layout, U, dim, rho, u, e_int_density);

        DofStateView S(U.data(), 0);

        // Pressure
        const real_t p = gas.pressure(S);
        EXPECT_CLOSE(p, 1.0, tol);

        // Temperature: p = rho * R * T => T = p / (rho * R)
        const real_t T_expected = p / (rho * R_gas);
        const real_t T = gas.temperature(S);
        EXPECT_CLOSE(T, T_expected, tol);

        // Sound speed: a^2 = gamma * p / rho
        const real_t a_expected = std::sqrt(gamma * p / rho);
        const real_t a = gas.sound_speed(S);
        EXPECT_CLOSE(a, a_expected, tol);

        // Density should just be rho
        const real_t rho_out = gas.density(S);
        EXPECT_CLOSE(rho_out, rho, tol);

        // Specific internal energy e = e_int_density / rho
        const real_t e_expected = e_int_density / rho;
        const real_t e_si = gas.specific_internal_energy(S);
        EXPECT_CLOSE(e_si, e_expected, tol);
    }

    return 0;
}

// -----------------------------------------------------------------------------
// GasModel transport test: verify that GasModel's viscosity and thermal
// conductivity are consistent with EOS + Transport.
//
// For a given state S:
//   - Let T  = eos.temperature(S)
//   - Let cp = eos.cp(S)
//   - Let mu_expected = transport.viscosity(T)
//   - Let k_expected  = transport.thermal_conductivity(T, cp)
//
// We expect:
//   - gas.viscosity(S) == mu_expected
//   - gas.thermal_conductivity(S) == k_expected
//
// This works for both constant and Sutherland transport, because it uses
// the same Transport implementation as GasModel internally.
// -----------------------------------------------------------------------------
TEST(GasModel_IdealGas_Transport)
{
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(gamma, Pr, R_gas, mu);

    IdealSingleGasEOS  eos;
    Transport          transport;

    const real_t tol = 1.0e-12;

    // Just pick dim = 3 here; the exact dimension doesn't really matter
    // as long as the state is consistent.
    const int dim   = 3;
    const int ndofs = 1;
    StateLayout layout(dim, ndofs);
    IdealGasModel gas(*phys, layout);

    const int num_eq = layout.eq_energy + 1;
    std::vector<real_t> U(num_eq * ndofs);

    const real_t rho = 1.5;
    const real_t u[3] = {50.0, -20.0, 5.0};

    // Again choose e_int_density such that p = 1 (just for convenience):
    const real_t e_int_density = 1.0 / (gamma - 1.0);

    fill_single_dof_state(layout, U, dim, rho, u, e_int_density);

    DofStateView S(U.data(), 0);

    // Use the EOS + Transport directly to compute the expected results
    const real_t T  = eos.temperature(*phys, layout, S);
    const real_t cp = eos.cp(*phys, layout, S);

    const real_t mu_expected = transport.viscosity(*phys, layout, eos, S);
    const real_t k_expected  = transport.thermal_conductivity(*phys, layout, eos, S);

    const real_t mu_gas = gas.viscosity(S);
    const real_t k_gas  = gas.thermal_conductivity(S);

    EXPECT_EQ(mu_gas, mu_expected);
    EXPECT_EQ(k_gas,  k_expected);

    return 0;
}

// -----------------------------------------------------------------------------
// GasModel entropy-gradient -> primitive-gradient transform regression test.
//
// Goal: ensure the refactored IdealGasModel::grad_entropy_to_grad_prim(...) is
// algebraically consistent with the legacy EntropyGrad2PrimGrad(...) routine.
//
// The legacy routine operated point-wise (per DOF) on:
//   - conservative state Q = [rho, rho*u_0.., rhoE]
//   - an input "entropy gradient" vector dE (same slot layout as entropy vars)
//
// and produced an output vector (same slot layout as Q) via:
//   out_mom[i]  = p/rho * ( dE_mom[i] + u_i * dE_last )
//   out_mass    = rho*dE_mass - dE_last*(KE - p/(gamma-1)) + (rho/p)*sum(mom_i*out_mom[i])
//   out_last    = p/rho * ( out_mass + p*dE_last )
//
// Here, dE_last corresponds to the last slot of the entropy-variable vector
// (w_E = -beta for ideal gas), and KE is the kinetic energy density.
// -----------------------------------------------------------------------------
static void
legacy_entropy_grad_to_prim_grad(const real_t* Q,
                                 const real_t* dE,
                                 real_t* out,
                                 int dim,
                                 real_t gamma)
{
    const real_t rho = Q[0];

    // Velocity
    real_t u[3] = {0, 0, 0};
    real_t u2 = 0;
    for (int i = 0; i < dim; ++i)
    {
        u[i] = Q[1 + i] / rho;
        u2 += u[i] * u[i];
    }

    const real_t KE = real_t(0.5) * rho * u2;          // kinetic energy density
    const real_t rhoE = Q[dim + 1];
    const real_t p = (gamma - real_t(1)) * (rhoE - KE);

    const real_t gammaM1Inv = real_t(1) / (gamma - real_t(1));
    const real_t ie = p * gammaM1Inv;                  // internal energy density

    const real_t dE_last = dE[dim + 1];

    // out_mom[i]
    real_t mom_dot = 0;
    for (int i = 0; i < dim; ++i)
    {
        const real_t out_i = (p / rho) * (dE[1 + i] + u[i] * dE_last);
        out[1 + i] = out_i;
        mom_dot += Q[1 + i] * out_i; // (rho*u_i) * out_i
    }

    // out_mass
    const real_t out_mass = rho * dE[0] - dE_last * (KE - ie) + (rho / p) * mom_dot;
    out[0] = out_mass;

    // out_last
    out[dim + 1] = (p / rho) * (out_mass + p * dE_last);
}

TEST(GasModel_IdealGas_GradEntropyToGradPrim_MatchesLegacy)
{
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(gamma, Pr, R_gas, mu);

    const real_t tol = 5.0e-11;

    for (int dim = 1; dim <= 3; ++dim)
    {
        const int ndofs = 1;
        StateLayout layout(dim, ndofs); // no scalars
        IdealGasModel gas(*phys, layout);
        const int num_eq = layout.nequations();

        std::vector<real_t> Q(num_eq);

        // Choose a state with positive pressure.
        const real_t rho = 1.7;
        const real_t uvec[3] = {35.0, -7.0, 2.5};
        const real_t e_int_density = 2.3; // arbitrary positive

        fill_single_dof_state(layout, Q, dim, rho, uvec, e_int_density);

        // Synthetic entropy-variable gradient input (dE); does not need to be physical.
        std::vector<real_t> dE(num_eq);
        dE[0] = 0.11;
        for (int i = 0; i < dim; ++i)
            dE[1 + i] = 0.2 + 0.05 * (i + 1);
        dE[dim + 1] = -0.33;

        // Compute expected (legacy) output.
        std::vector<real_t> out_expected(num_eq, 0.0);
        legacy_entropy_grad_to_prim_grad(Q.data(), dE.data(), out_expected.data(), dim, gamma);

        // Compute refactored output.
        std::vector<real_t> out(num_eq, 0.0);
        PointStateView   S{Q.data()};
        PointStateView   dE_view{dE.data()};
        PointStateViewRW out_view{out.data()};

        gas.grad_entropy_to_grad_prim(S, dE_view, out_view);

        // Compare slots.
        for (int eq = 0; eq < num_eq; ++eq){
            EXPECT_CLOSE(out[eq], out_expected[eq], tol);
            if (std::abs(out[eq] - out_expected[eq]) > tol) {
              std::cerr << "eq=" << eq
                        << " out=" << out[eq]
                        << " ref=" << out_expected[eq]
                        << " diff=" << (out[eq] - out_expected[eq]) << "\n";
            }
        }
    }
    return 0;
}
