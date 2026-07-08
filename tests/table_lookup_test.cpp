// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "unit_test.hpp"
#include "test_helpers.hpp"
#include "plato_helpers.hpp"
#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTETable.hpp"
#include "LTEEOS.hpp"
#include "TheseusConfig.hpp"

using real_t = mfem::real_t;

using namespace Theseus;
using namespace Theseus::LTETable;

TEST(plato_library_init_test)
{
    // CL NOTE : To make sure the database path is correctly set and avoiding the initialization overhead in the subsequent tests.
    std::string solver  = "LTE_table_rhoT_(air5)";
    std::string mixture = "air5";
    std::string path(Theseus::BuildConfig::PlatoDBPath);
    if(Theseus::LTETable::check_plato_database_path(path)){
      std::cerr << "Failed to find PLATO database: " << path << std::endl;
      return 1;
    }
    //"/home/cherith2/Workspace/SOURCE_CODES/database";
    std::string empty_str = "empty";
    std::cout << "Calling PLATO Initialize with: " << std::endl
	      << "Solver: " << solver << std::endl
	      << "Mixture: " << mixture << std::endl
	      << "Path: " << path << std::endl;

    plato_initialize(solver.c_str(), mixture.c_str(), empty_str.c_str(), empty_str.c_str(), path.c_str());
    return 0;
}

TEST(hunt_cpu_test)
{
    int n = 100;
    mfem::Vector arr(n);
    for (int i = 0; i < n; i++)
    {
        arr[i] = i + 1;
    }

    real_t x = 34.5;

    // Hunt Right (guess near left)
    int ind_lo = 2;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Hunt Left (guess near right)
    ind_lo = 70;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Left Boundary
    x = 1;
    ind_lo = 50;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 0, 1e-14);

    // Right Boundary
    x = n;
    ind_lo = 20;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, n-2, 1e-14);

    return 0;
}

TEST(hunt_gpu_test)
{
    const int n = 100;
    mfem::Vector arr(n);
    for (int i = 0; i < n; i++) { arr[i] = i + 1; }

    const real_t *a = arr.Read(); // device-safe read pointer
    arr.UseDevice();

    mfem::Vector outv(4);
    outv.UseDevice();
    real_t *out_d = outv.Write(); // device-safe write pointer

    mfem::forall(4, [=] MFEM_HOST_DEVICE (int i)
    {
        real_t x;
        int guess;

        if (i == 0)      { x = 34.5;  guess = 2;  }   // hunt right
        else if (i == 1) { x = 34.5;  guess = 70; }   // hunt left
        else if (i == 2) { x = 1.0;   guess = 50; }   // left boundary
        else             { x = 100.0; guess = 20; }   // right boundary

        int idx = hunt(a, n, x, guess);
        out_d[i] = (real_t) idx;
    });

    const real_t *out_h = outv.HostRead();

    EXPECT_CLOSE(out_h[0], 33.0, 1e-14);
    EXPECT_CLOSE(out_h[1], 33.0, 1e-14);
    EXPECT_CLOSE(out_h[2],  0.0, 1e-14);
    EXPECT_CLOSE(out_h[3], 98.0, 1e-14); // n-2

    return 0;
}

TEST(plato_Temperature_solve_test)
{
    // Range and resolution of the table
    int nx = 1001, ny = 1001;
    real_t rho_min  = 1e-4  , rho_max  = 1.1 ;
    real_t T_min = 250.0, T_max = 10000.0;

    int num_properties = 9;

    mfem::Vector lte_table( (num_properties) * (nx*ny) ), inv_table( nx*ny );
    mfem::Vector rho_grid, T_grid, e_grid;

    const int dim = 3, ndofs = 1;
    StateLayout L(dim, ndofs);
    Theseus::LTETable::LTETables lteT;
    lteT.L.setup(nx, ny);

    const int num_eq = L.eq_energy + 1;
    const real_t u1[3] = {10.0, -3.0, 5.0};
    std::vector<real_t> U(num_eq * ndofs);

    // Generation of LTE table
    log_grid(nx, rho_min, rho_max, rho_grid);
    log_grid(ny, T_min, T_max, T_grid);

    real_t e_min, e_max;
    fill_table(lteT.L, rho_grid.GetData(), T_grid.GetData(),
	       lte_table.GetData(), e_min, e_max);

    uniform_grid(ny, e_min, e_max, e_grid);
    fill_inv_table(lteT.L, rho_grid.GetData(), e_grid.GetData(), T_grid.GetData(), inv_table.GetData());

    lteT.tables = {
      lte_table.HostRead(), inv_table.HostRead(),
      rho_grid.HostRead(), T_grid.HostRead(),
      e_grid.HostRead()
    };

    std::shared_ptr<PhysicsConstants> phys = std::make_shared<PhysicsConstants>(1.4, 0.72, 287.05, 0.02);
    LTEGasEOS eos;

    // ------------------------------------------------------------------------------------
    // PLATO setup
    PlatoMixture mix;

    // TEST : (Temperature inversion at an arbitrary point in the table)
    real_t rho = 0.233525*rho_grid[nx/4] + 0.766475*rho_grid[5*nx/6];
    real_t T0   = 0.768256*T_grid[ny/12] + 0.231744*T_grid[4*ny/5];

    plato_set_state(rho, T0, mix);
    real_t e = mix.e;

    real_t rhoe = rho * e;
    fill_single_dof_state(L, U, dim, rho, u1, rhoe);
    DofStateView S1(U.data(), 0);

    real_t T_inv_table = eos.biinterp_inverse_table(*phys, L, S1, lteT);
    real_t T_newton    = eos.temp_from_internal_energy(*phys, L, S1, lteT);

    real_t P_true = mix.P;
    real_t P_tab_direct  = eos.biinterp_lte_table(lteT.L.P_idx,*phys, L, S1, T0, lteT);
    real_t P_tab_newton  = eos.pressure(*phys, L, S1, lteT);
    real_t e_newton = eos.biinterp_lte_table(lteT.L.e_idx,*phys, L, S1, T_newton, lteT);

    EXPECT_CLOSE(e/e, e_newton/e, 1e-14);

    std::cout << "\n";
    std::cout << "T_inv_table - T_true = " << T_inv_table - T0 << std::endl;
    std::cout << "T_newton - T_true = " << T_newton - T0 << std::endl;

    std::cout << "\n";
    std::cout << "P_tab_direct - P_true = " << (P_tab_direct - P_true)/P_true << std::endl;
    std::cout << "P_tab_newton - P_true = " << (P_tab_newton - P_true)/P_true << std::endl;
    std::cout << "P_tab_newton - P_tab_direct = " << (P_tab_newton - P_tab_direct)/P_tab_direct << "\n" << std::endl;

    // TEST : Obtaining internal energy from the from pressure (inverse table lookup)
    real_t T_random = 0.12456*T_grid[ny/5] + 0.87544*T_grid[3*ny/4];
    plato_set_state(rho, T_random, mix);
    fill_single_dof_state(L, U, dim, rho, u1, rho*mix.e);
    DofStateView S_random(U.data(), 0);
    real_t rhoe_inverse = eos.internal_energy_from_pressure(*phys, L, S_random, P_true, lteT);
    real_t rel_err_inverse = std::abs(rhoe_inverse - rho*e)/std::abs(rho*e);
    EXPECT_SMALL(rel_err_inverse, 1e-7);

    return 0;
}

TEST(plato_Tablelookup_test)
{
    // Range and resolution of the table
    int nx = 101, ny = 101;
    real_t rho_min  = 0.01  , rho_max  = 0.11 ;
    real_t T_min = 250.0, T_max = 500.0;

    int num_properties = 9;

    mfem::Vector lte_table( (num_properties) * (nx*ny) ), inv_table( nx*ny );
    mfem::Vector rho_grid, T_grid, e_grid;

    const int dim = 3, ndofs = 1;
    StateLayout L(dim, ndofs);
    Theseus::LTETable::LTETables lteT(nx, ny);

    const int num_eq = L.nequations();
    const real_t u1[3] = {10.0, -3.0, 5.0};
    std::vector<real_t> U(num_eq * ndofs);

    // Generation of LTE table
    // uniform_grid(nx, rho_min, rho_max, rho_grid);
    // uniform_grid(ny, T_min, T_max, T_grid);
    log_grid(nx, rho_min, rho_max, rho_grid);
    log_grid(ny, T_min, T_max, T_grid);
    real_t e_min, e_max;

    fill_table(lteT.L, rho_grid.GetData(), T_grid.GetData(),
	       lte_table.GetData(), e_min, e_max);

    uniform_grid(ny, e_min, e_max, e_grid);
    fill_inv_table(lteT.L, rho_grid.GetData(), e_grid.GetData(), T_grid.GetData(), inv_table.GetData());

    lteT.tables = {
      lte_table.HostRead(), inv_table.HostRead(),
      rho_grid.HostRead(), T_grid.HostRead(),
      e_grid.HostRead()
    };

    std::shared_ptr<PhysicsConstants> phys = std::make_shared<PhysicsConstants>(1.4, 0.72, 287.05, 0.02);
    LTEGasEOS eos;

    // ------------------------------------------------------------------------------------
    // PLATO setup
    PlatoMixture mix;

    // Test 1 : (Table values at the corner of the table)
    real_t rho  = rho_grid[3];
    real_t T = T_grid[7];
    plato_set_state(rho, T, mix);
    real_t e = mix.e;

    real_t rhoe = rho * e;
    fill_single_dof_state(L, U, dim, rho, u1, rhoe);
    DofStateView S1(U.data(), 0);

    real_t P_table  = eos.pressure(*phys, L, S1, lteT);
    real_t P_corner = lte_table[lteT.L.property_index(lteT.L.P_idx, 3, 7)];
    real_t rel_err = std::abs(mix.P - P_table)/std::abs(mix.P);

    EXPECT_SMALL(rel_err, 1e-14);
    EXPECT_CLOSE(mix.P, P_corner, 1e-14);

    // TEST 2 : (Table values at a mid-point of a cell in the table)
    rho = 0.5*(rho_grid[3] + rho_grid[4]);
    T   = 0.5*(T_grid[7] + T_grid[8]);
    plato_set_state(rho, T, mix);
    e = mix.e;

    rhoe = rho * e;
    fill_single_dof_state(L, U, dim, rho, u1, rhoe);
    DofStateView S2(U.data(), 0);

    P_table  = eos.pressure(*phys, L, S2, lteT);
    rel_err = std::abs(mix.P - P_table)/std::abs(mix.P);

    real_t P_expected = 0;
    int l_x = hunt(rho_grid.Read(), nx, rho, 0), l_y = hunt(T_grid.Read(), ny, T, 0);
    for(int i=0; i < 2; i++)
    {
        for(int j=0; j < 2; j++)
        {
	  P_expected += lte_table[lteT.L.property_index(lteT.L.P_idx, l_x + i, l_y + j)];
        }
    }
    P_expected /= 4.0;

    EXPECT_SMALL(rel_err, 1e-6);
    EXPECT_CLOSE(P_table/P_expected, 1.0, 1e-6);

    // TEST 3 : (Table values at an arbitrary point in the table)
    rho = 0.233525*rho_grid[3] + 0.766475*rho_grid[4];
    T   = 0.768256*T_grid[7] + 0.231744*T_grid[8];
    plato_set_state(rho, T, mix);
    e = mix.e;
    rhoe = rho * e;
    fill_single_dof_state(L, U, dim, rho, u1, rhoe);
    DofStateView S3(U.data(), 0);
    P_table  = eos.pressure(*phys, L, S3, lteT);
    rel_err = std::abs(mix.P - P_table)/std::abs(mix.P);
    EXPECT_SMALL(rel_err, 1e-7);

    // TEST 4 : Set (rho,rhoe) Obtain P and then see if we can obtain same rhoe from P using the inverse table lookup
    for(real_t rho_true = rho_grid[3]; rho_true <= rho_grid[4] ; rho_true += 0.0001)
    {
        real_t rhoe_true = rho_true * e_grid[50];

        fill_single_dof_state(L, U, dim, rho_true, u1, rhoe_true);
        DofStateView S4(U.data(), 0);

        real_t P_table = eos.pressure(*phys, L, S4, lteT);
        real_t rhoe_new = rho_true*e_grid[10]; // Initial guess for rho*e
        fill_single_dof_state(L, U, dim, rho_true, u1, rhoe_new);
        PointStateView S5(U.data());
        real_t rhoe_inverse = eos.internal_energy_from_pressure(*phys, L, S5, P_table, lteT);
        EXPECT_CLOSE(rhoe_true/rhoe_true, rhoe_inverse/rhoe_true, 1e-15);
    }

    plato_finalize();
    return 0;
}
