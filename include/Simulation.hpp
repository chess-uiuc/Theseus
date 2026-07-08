// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "Theseus.hpp"
#include "RHSOperator.hpp"
#include "Indicator.hpp"

namespace Theseus
{

  class Simulation
  {
  private:
    int numProcs, myRank;
    int order;
    int dim;
    int num_equations;
    int ref_levels;
    int vis_steps;
    int nancheck_steps;
    int precision;
    int num_dofs_scalar;
    int num_dofs_system;
    int print_interval;
    int ti;
    int nsteps_max;
    int checkpoint_cycle;
  
    bool done = false;
    bool variable_dt = false;
    bool clock_simulation = true;
    bool nancheck = false;
    bool visualize = true;
    bool visit = false;
    bool paraview = true;
    bool checkpoint_load = false;
    bool checkpoint_save = false;
  
    std::string output_file_path;
    std::string paraview_folder;
    std::string checkpoints_folder;
  
    mfem::real_t t, t_final, dt, dt_real;
    mfem::real_t cfl;
    mfem::real_t hmin;
    mfem::real_t Re, Ma;
    mfem::real_t next_save_t;
    mfem::real_t save_dt1;
    mfem::real_t save_dt2;
    mfem::real_t trigger_t;
    mfem::real_t save_dt;
    mfem::real_t next_checkpoint_t;
    mfem::real_t checkpoint_dt;
  
    mfem::real_t V_sq;
  
    mfem::real_t alpha_max;
  
    mfem::Array<int> mesh_ordering;
    std::shared_ptr<mfem::ParMesh> pmesh;
  
    int btype = mfem::BasisType::GaussLobatto;
    int ordering = mfem::Ordering::byNODES;

    std::shared_ptr<mfem::DG_FECollection> fec;
    std::shared_ptr<mfem::DG_FECollection> fec0;
    std::shared_ptr<mfem::ParFiniteElementSpace> vfes;
    std::shared_ptr<mfem::ParFiniteElementSpace> fes0;
    std::unique_ptr<mfem::ParFiniteElementSpace> fes;
    std::unique_ptr<mfem::ParFiniteElementSpace> dfes;
  
    std::unique_ptr<mfem::VectorFunctionCoefficient> u0;
  
    std::shared_ptr<mfem::ParGridFunction> sol;
    std::shared_ptr<mfem::ParGridFunction> dudx;
    std::shared_ptr<mfem::ParGridFunction> dudy;
    std::shared_ptr<mfem::ParGridFunction> dudz;
    std::shared_ptr<mfem::ParGridFunction> r_gf;

    // Subcell blending : nullptr if OFF
    std::shared_ptr<mfem::ParGridFunction> eta;
    std::shared_ptr<mfem::ParGridFunction> alpha;
    std::shared_ptr<Prandtl::Indicator> indicator;

    std::vector<std::shared_ptr<mfem::VectorFunctionCoefficient>> BC_coeff;
  
    mfem::FunctionCoefficient r_coef, z_coef;
  
    mfem::ParGridFunction rho, mom, energy;
  
    std::unique_ptr<mfem::ParGridFunction> u, v, w;
    std::unique_ptr<mfem::ParGridFunction> p, rho_axi;
  
    std::unique_ptr<mfem::ParaViewDataCollection> pd;
    std::unique_ptr<mfem::VisItDataCollection> vd;

    std::shared_ptr<mfem::ODESolver> ode_solver;
    std::unique_ptr<Theseus::RHSOperatorBase> rhsOp;

    int signature;
  
    std::vector<mfem::Array<int>> bdr_marker_vector;
    mfem::Array<int> set_marker;
    int max_bdr_attr;
    void InitDevice(std::string);
    std::unique_ptr<mfem::Device> device_;

#ifdef AXISYMMETRIC
    void ConservativeToPrimitive(const mfem::Vector &U_cons,
                                 mfem::ParGridFunction &rho_out,
                                 mfem::ParGridFunction &uz_out,
                                 mfem::ParGridFunction &ur_out,
                                 mfem::ParGridFunction &p_out) const;  
#endif
  
    Simulation(std::string);
  
  public:    
    static Simulation& SimulationCreate(std::string);
    int LoadConfig(const std::string &config_file_path);

    ~Simulation();

    void Run();

    Simulation(const Simulation&) = delete;
    Simulation& operator = (const Simulation&) = delete;
  };
  
}
