// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "unit_test.hpp"
#include "state_semantics.hpp"
#include "legacy_state_adapter.hpp"
#include "GasState.hpp"
#include "gas_state_adapter.hpp"

using real_t = mfem::real_t;

TEST(LegacyState_MassMomentumEnergy_2D)
{
    const int dim   = 2;   // or test dim=3 in another test
    const int ndofs = 5;   // small number is fine

    LegacyConservativeState state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(LegacyState_MassMomentumEnergy_3D)
{
    const int dim   = 3;   // or test dim=3 in another test
    const int ndofs = 5;   // small number is fine

    LegacyConservativeState state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(GasState_MassMomentumEnergy_2D)
{
    const int dim   = 2;
    const int ndofs = 5;

    GasStateSemanticsAdapter state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(GasState_MassMomentumEnergy_3D)
{
    const int dim   = 3;
    const int ndofs = 5;

    GasStateSemanticsAdapter state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(StateLayout_Indexing_NoScalars_2D)
{
    const int dim   = 2;
    const int ndofs = 5;

    Theseus::StateLayout layout(dim, ndofs);

    // Basic metadata
    EXPECT_CLOSE(layout.dim,             dim,     0.0);
    EXPECT_CLOSE(layout.num_dofs_scalar, ndofs,   0.0);
    EXPECT_CLOSE(layout.eq_mass,         0,       0.0);
    EXPECT_CLOSE(layout.eq_mom0,         1,       0.0);
    EXPECT_CLOSE(layout.eq_mom[0],       1,       0.0);
    EXPECT_CLOSE(layout.eq_mom[1],       2,       0.0);
    EXPECT_CLOSE(layout.eq_mom[2],      -1,       0.0); // unused in 2D
    EXPECT_CLOSE(layout.eq_energy,       dim+1,   0.0);
    EXPECT_CLOSE(layout.eq_scalar0,     -1,       0.0);
    EXPECT_CLOSE(layout.num_scalars,     0,       0.0);

    // Flat index should be eq * ndofs + dof
    const int num_eq = dim + 2; // rho + dim momenta + energy
    for (int eq = 0; eq < num_eq; ++eq)
    {
        for (int i = 0; i < ndofs; ++i)
        {
            const int expected = eq * ndofs + i;
            EXPECT_CLOSE(layout.index(eq, i), expected, 0.0);
        }
    }

    return 0;
}

TEST(StateLayout_Indexing_WithScalars_3D)
{
    const int dim        = 3;
    const int ndofs      = 4;
    const int num_scalars = 2;

    Theseus::StateLayout layout(dim, ndofs, num_scalars);

    EXPECT_CLOSE(layout.dim,             dim,                  0.0);
    EXPECT_CLOSE(layout.num_dofs_scalar, ndofs,                0.0);
    EXPECT_CLOSE(layout.eq_mass,         0,                    0.0);
    EXPECT_CLOSE(layout.eq_mom0,         1,                    0.0);
    EXPECT_CLOSE(layout.eq_mom[0],       1,                    0.0);
    EXPECT_CLOSE(layout.eq_mom[1],       2,                    0.0);
    EXPECT_CLOSE(layout.eq_mom[2],       3,                    0.0);
    EXPECT_CLOSE(layout.eq_energy,       dim + 1,              0.0); // 4
    EXPECT_CLOSE(layout.eq_scalar0,      dim + 2,              0.0); // 5
    EXPECT_CLOSE(layout.num_scalars,     num_scalars,          0.0);

    const int num_eq = dim + 2 + num_scalars; // rho, 3 mom, E, 2 scalars

    for (int eq = 0; eq < num_eq; ++eq)
    {
        for (int i = 0; i < ndofs; ++i)
        {
            const int expected = eq * ndofs + i;
            EXPECT_CLOSE(layout.index(eq, i), expected, 0.0);
        }
    }

    // Specifically check scalar blocks start where we expect
    for (int k = 0; k < num_scalars; ++k)
    {
        const int eq_scalar_k = layout.eq_scalar0 + k;
        for (int i = 0; i < ndofs; ++i)
        {
            const int expected = eq_scalar_k * ndofs + i;
            EXPECT_CLOSE(layout.index(eq_scalar_k, i), expected, 0.0);
        }
    }

    return 0;
}

TEST(DofStateView_ReadsExpectedComponents)
{
    const int dim   = 3;
    const int ndofs = 4;
    const int nscalars = 2;

    Theseus::StateLayout layout(dim, ndofs, nscalars);
    const int num_eq = dim + 2 + nscalars;             // rho, dim*mom, E, scalars

    std::vector<real_t> U(num_eq * ndofs);

    // Fill equation-blocked storage with a simple pattern:
    //   U(eq, i) = 10*eq + i + 1.0
    for (int eq = 0; eq < num_eq; ++eq)
    {
        for (int i = 0; i < ndofs; ++i)
        {
            U[eq * ndofs + i] = 10.0 * eq + i + 1;
        }
    }

    for (int i = 0; i < ndofs; ++i)
    {
        Theseus::DofStateView S{U.data(), i};

        // Mass
        EXPECT_EQ(S.mass(layout), 10.0 * layout.eq_mass + i + 1);

        // Momentum components
        for (int d = 0; d < dim; ++d)
        {
            const int eq_m = layout.eq_mom[d];
            EXPECT_EQ(S.momentum(layout, d), 10.0 * eq_m + i + 1);
            if (d == 0){
              EXPECT_EQ(S.momentum_x(layout), S.momentum(layout, d));
            }
            if (d == 1){
              EXPECT_EQ(S.momentum_y(layout), S.momentum(layout, d));
            }
            if (d == 2){
              EXPECT_EQ(S.momentum_z(layout), S.momentum(layout, d));
            }
        }

        // Velocity components
        for (int d = 0; d < dim; ++d)
        {
          const int eq_m = layout.eq_mom[d];
          EXPECT_CLOSE(S.velocity(layout, d),(10.0*eq_m+i+1)/(i+1), 1e-16);
          if (d == 0){
            EXPECT_EQ(S.velocity_x(layout), S.velocity(layout, d));
          }
          if (d == 1){
            EXPECT_EQ(S.velocity_y(layout), S.velocity(layout, d));
          }
          if (d == 2){
            EXPECT_EQ(S.velocity_z(layout), S.velocity(layout, d));
          }
        }
        
        // Energy
        EXPECT_EQ(S.energy(layout), 10.0 * layout.eq_energy + i + 1);
        
        // Scalars
        for(int s = 0; s < nscalars; s++){
          const int eq_s = layout.eq_scalar0;
          EXPECT_EQ(S.scalar(layout, s),10.0*(eq_s+s) + i + 1);
        }
    }    
    return 0;
}

TEST(PointStateView_ReadsExpectedComponents)
{
  const int dim   = 3;
  const int ndofs = 4;
  const int nscalars = 2;
  
  Theseus::StateLayout layout(dim, ndofs, nscalars);
  const int num_eq = dim + 2 + nscalars;             // rho, dim*mom, E, scalars
  
  std::vector<real_t> U(num_eq);
  
  // Fill equation-blocked storage with a simple pattern:
  //   U(eq, i) = 10*eq + i + 1.0
  for (int eq = 0; eq < num_eq; ++eq)
    {
      U[eq] = 10.0 * eq + 2;
    }
  
  Theseus::PointStateView S{U.data()};
  
  // Mass
  EXPECT_EQ(S.mass(layout), 10.0 * layout.eq_mass + 2);
  
  // Momentum components
  for (int d = 0; d < dim; ++d)
    {
      const int eq_m = layout.eq_mom[d];
      EXPECT_EQ(S.momentum(layout, d), 10.0 * eq_m + 2);
      if (d == 0){
        EXPECT_EQ(S.momentum_x(layout), S.momentum(layout, d));
      }
      if (d == 1){
        EXPECT_EQ(S.momentum_y(layout), S.momentum(layout, d));
      }
      if (d == 2){
        EXPECT_EQ(S.momentum_z(layout), S.momentum(layout, d));
      }
      EXPECT_CLOSE(S.velocity(layout, d),(10.0*eq_m+2)/2, 1e-16);
      if (d == 0){
        EXPECT_EQ(S.velocity_x(layout), S.velocity(layout, d));
      }
      if (d == 1){
        EXPECT_EQ(S.velocity_y(layout), S.velocity(layout, d));
      }
      if (d == 2){
        EXPECT_EQ(S.velocity_z(layout), S.velocity(layout, d));
      }
    }
  
  // Energy
  EXPECT_EQ(S.energy(layout), 10.0 * layout.eq_energy + 2);
  
  // Scalars
  for(int s = 0; s < nscalars; s++){
    const int eq_s = layout.eq_scalar0;
    EXPECT_EQ(S.scalar(layout, s),10.0*(eq_s+s) + 2);
  }
  return 0;
}

TEST(FieldStateView_ReadWriteRoundTrip)
{
    const int dim        = 3;
    const int ndofs      = 3;
    const int num_scalars = 2;

    Theseus::StateLayout layout(dim, ndofs, num_scalars);
    const int num_eq = dim + 2 + num_scalars;

    std::vector<real_t> U(num_eq * ndofs, 0.0);
    Theseus::FieldStateView S{U.data()};

    // Write using named accessors
    for (int i = 0; i < ndofs; ++i)
    {
      S.mass(layout, i)       = 1.0 + i;
      S.momentum_x(layout, i) = 2.0 + i;
      S.momentum_y(layout, i) = 3.0 + i;
      S.momentum_z(layout, i) = 4.0 + i;
      S.energy(layout, i)     = 5.0 + i;
      S.scalar(layout, 0, i)  = 6.0 + i;
      S.scalar(layout, 1, i)  = 7.0 + i;
    }

    // Read back
    for (int i = 0; i < ndofs; ++i)
    {
      EXPECT_EQ(S.mass(layout,i),       1.0 + i);
      EXPECT_EQ(S.momentum_x(layout, i), 2.0 + i);
      EXPECT_EQ(S.momentum_y(layout, i), 3.0 + i);
      EXPECT_EQ(S.momentum_z(layout, i), 4.0 + i);
      EXPECT_EQ(S.energy(layout, i),     5.0 + i);
      EXPECT_EQ(S.scalar(layout, 0, i),  6.0 + i);
      EXPECT_EQ(S.scalar(layout, 1, i),  7.0 + i);
      EXPECT_CLOSE(S.velocity_x(layout, i), (2.0 + i)/(1.0 + i), 1e-16);
      EXPECT_CLOSE(S.velocity_y(layout, i), (3.0 + i)/(1.0 + i), 1e-16);
      EXPECT_CLOSE(S.velocity_z(layout, i), (4.0 + i)/(1.0 + i), 1e-16);
    }

    return 0;
}


TEST(PointStateViewRW_WritesExpectedComponents)
{
    const int dim   = 3;
    const int ndofs = 1;
    const int nscalars = 2;

    Theseus::StateLayout layout(dim, ndofs, nscalars);
    const int num_eq = dim + 2 + nscalars;

    std::vector<real_t> U(num_eq, -1.0);

    Theseus::PointStateViewRW E{U.data()};

    // Write via setters
    E.set_mass(layout, 2.0);
    E.set_momentum(layout, 0, 3.0);
    E.set_momentum(layout, 1, -4.0);
    E.set_momentum(layout, 2, 5.0);
    E.set_energy(layout, 9.0);
    E.set_scalar(layout, 0, 11.0);
    E.set_scalar(layout, 1, 12.0);

    // Read back via RW view
    EXPECT_EQ(E.mass(layout), 2.0);
    EXPECT_EQ(E.momentum(layout, 0), 3.0);
    EXPECT_EQ(E.momentum(layout, 1), -4.0);
    EXPECT_EQ(E.momentum(layout, 2), 5.0);
    EXPECT_EQ(E.energy(layout), 9.0);
    EXPECT_EQ(E.scalar(layout,0), 11.0);
    EXPECT_EQ(E.scalar(layout,1), 12.0);

    // Cross-check via const view over same buffer
    Theseus::PointStateView S{U.data()};

    EXPECT_EQ(S.mass(layout), 2.0);
    EXPECT_EQ(S.momentum(layout, 0), 3.0);
    EXPECT_EQ(S.momentum(layout, 1), -4.0);
    EXPECT_EQ(S.momentum(layout, 2), 5.0);
    EXPECT_EQ(S.energy(layout), 9.0);
    EXPECT_EQ(S.scalar(layout, 0), 11.0);
    EXPECT_EQ(S.scalar(layout, 1), 12.0);

    return 0;
}


TEST(PointStateViewRW_WritesToMFEMVector)
{
  Theseus::StateLayout layout(/*dim=*/2, /*ndofs=*/1, /*num_scalars=*/0);
  mfem::Vector v(layout.nequations());
  v = 0.0;

  Theseus::PointStateViewRW E(v.GetData());
  E.set_mass(layout, 2.0);
  E.set_momentum(layout, 0, 3.0);
  E.set_momentum(layout, 1, -4.0);
  E.set_energy(layout, 9.0);

  EXPECT_EQ(v[layout.eq_mass], 2.0);
  EXPECT_EQ(v[layout.eq_mom[0]], 3.0);
  EXPECT_EQ(v[layout.eq_mom[1]], -4.0);
  EXPECT_EQ(v[layout.eq_energy], 9.0);

  return 0;
}
