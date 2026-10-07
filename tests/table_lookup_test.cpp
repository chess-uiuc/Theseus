// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"
#include "test_helpers.hpp"
#include "plato_helpers.hpp"
#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTETable.hpp"
#include "LTEEOS.hpp"
#include "LTEGasModel.hpp"
#include "TheseusConfig.hpp"
#include "json.hpp"

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

// Small real-air table validates the public startup API independently of the
// analytic fixture. This test can also be selected by name for device handoff.
TEST(plato_PhysicalState_conversion_test)
{
    const std::string path(Theseus::BuildConfig::PlatoDBPath);
    plato_initialize("LTE_table_rhoT_(air5)", "air5", "empty", "empty", path.c_str());
    LTETable::Data data;
    LTETables tables(25,25);
    log_grid(25,0.1,1.1,data.rho_grid);
    log_grid(25,250,3000,data.T_grid);
    data.lte_table.SetSize(9*25*25);
    data.inv_table.SetSize(25*25);
    real_t e_min,e_max;
    fill_table(tables.L,data.rho_grid.GetData(),data.T_grid.GetData(),
               data.lte_table.GetData(),e_min,e_max);
    uniform_grid(25,e_min,e_max,data.e_grid);
    fill_inv_table(tables.L,data.rho_grid.GetData(),data.e_grid.GetData(),
                   data.T_grid.GetData(),data.inv_table.GetData());
    plato_finalize();
    tables.tables={data.lte_table.HostRead(),data.inv_table.HostRead(),
                   data.rho_grid.HostRead(),data.T_grid.HostRead(),data.e_grid.HostRead()};
    PhysicsConstants phys(1.4,0.72,287.05,0.02);
    LTEGas gas(phys,StateLayout(2,1),tables);
    IdealGasModel cpg(phys,StateLayout(2,1));
    for (real_t rho : {0.2,0.7}) for (real_t temperature : {300.,1200.,2500.}) {
      mfem::Vector reference;
      gas.ConservativeFromPhysical({DensityTemperature{rho,temperature},{30,-2}},reference);
      PointStateView state(reference.GetData());
      const real_t pressure=gas.pressure(state);
      for (const ThermodynamicInput &pair : std::vector<ThermodynamicInput>{
          PressureTemperature{pressure,temperature},DensityPressure{rho,pressure}}) {
        mfem::Vector converted;
        gas.ConservativeFromPhysical({pair,{30,-2}},converted);
        PointStateView result(converted.GetData());
        EXPECT_CLOSE(gas.density(result)/rho,1,1e-9);
        EXPECT_CLOSE(gas.temperature(result)/temperature,1,1e-9);
        EXPECT_CLOSE(gas.pressure(result)/pressure,1,1e-9);
        EXPECT_CLOSE(converted[3]/reference[3],1,1e-9);
      }
      mfem::Vector ideal;
      cpg.ConservativeFromPhysical({PressureTemperature{pressure,temperature},{30,-2}},ideal);
      // At room temperature real air may closely match CPG. Require a
      // model difference in the high-temperature fixture, not at every T.
      if (temperature == 2500)
        EXPECT_TRUE(std::abs(ideal[3]-reference[3]) > 1e-3*std::abs(reference[3]));
    }
    return 0;
}

TEST(plato_Visualization_reference)
{
  const std::string database(Theseus::BuildConfig::PlatoDBPath);
  plato_initialize("LTE_table_rhoT_(air5)", "air5", "empty", "empty", database.c_str());

  LTETable::Data data;
  LTETables tables(25, 25);
  log_grid(25, 0.05, 1.1, data.rho_grid);
  log_grid(25, 250, 3000, data.T_grid);
  data.lte_table.SetSize(9 * 25 * 25);
  data.inv_table.SetSize(25 * 25);
  real_t minimum_energy;
  real_t maximum_energy;
  fill_table(tables.L, data.rho_grid.GetData(), data.T_grid.GetData(),
             data.lte_table.GetData(), minimum_energy, maximum_energy);
  uniform_grid(25, minimum_energy, maximum_energy, data.e_grid);
  fill_inv_table(tables.L, data.rho_grid.GetData(), data.e_grid.GetData(),
                 data.T_grid.GetData(), data.inv_table.GetData());
  plato_finalize();
  tables.tables = {data.lte_table.HostRead(), data.inv_table.HostRead(),
                   data.rho_grid.HostRead(), data.T_grid.HostRead(), data.e_grid.HostRead()};

  const PhysicsConstants physics(1.4, 0.72, 287.05, 0.02);
  const LTEGas lte(physics, StateLayout(2, 1), tables);
  const IdealGasModel cpg(physics, StateLayout(2, 1));
  PhysicalStateInput input;
  input.thermo = PressureTemperature{60000, 1200};
  input.velocity = {10, -2};

  const auto print_reference = [&](const auto &gas, const char *model)
    {
      mfem::Vector conservative;
      gas.ConservativeFromPhysical(input, conservative);
      const PointStateView state(conservative.HostRead());
      const auto density = gas.density(state);
      const auto internal_energy = gas.specific_internal_energy(state);
      const auto sound_speed = gas.sound_speed(state);
      nlohmann::json reference;
      reference["Density"] = density;
      reference["Velocity"] = {10, -2};
      reference["Pressure"] = gas.pressure(state);
      reference["Temperature"] = gas.temperature(state);
      reference["Sound Speed"] = sound_speed;
      reference["Mach Number"] = std::sqrt(104.0) / sound_speed;
      reference["Specific Internal Energy"] = internal_energy;
      reference["Internal Energy Density"] = density * internal_energy;
      reference["Specific Total Energy"] = state.energy(gas.L) / density;
      reference["Viscosity"] = gas.viscosity(state);
      reference["Thermal Conductivity"] = gas.thermal_conductivity(state);
      std::cout << "VISUALIZATION_REFERENCE " << model << " " << reference.dump() << '\n';
    };

  print_reference(cpg, "cpg");
  print_reference(lte, "lte");
  return 0;
}
