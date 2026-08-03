// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "Theseus.hpp"
#include "CheckpointConfig.hpp"
#include "RHSOperator.hpp"
#include "RunControl.hpp"
#include "VisualizationConfig.hpp"

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
  
    bool done = false;
    bool variable_dt = false;
    bool clock_simulation = true;
    bool nancheck = false;
    bool visualize = true;
    bool visit = false;
    bool paraview = true;
    CheckpointConfig checkpoint_config;
    VisualizationConfig visualization_config;
  
    std::string output_file_path;
    std::string paraview_folder;
  
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
    std::unique_ptr<mfem::VectorFunctionCoefficient> exact_solution;
  
    std::shared_ptr<mfem::ParGridFunction> sol;
    std::shared_ptr<mfem::ParGridFunction> dudx;
    std::shared_ptr<mfem::ParGridFunction> dudy;
    std::shared_ptr<mfem::ParGridFunction> dudz;

    // Subcell blending : nullptr if OFF
    std::shared_ptr<mfem::ParGridFunction> eta;
    std::shared_ptr<mfem::ParGridFunction> alpha;
    std::shared_ptr<Prandtl::PerssonPeraireIndicator> indicator;

    std::vector<std::shared_ptr<mfem::VectorFunctionCoefficient>> BC_coeff;
  
    mfem::ParGridFunction rho, mom, energy;
  
    std::unique_ptr<mfem::ParGridFunction> velocity;
    std::unique_ptr<mfem::ParGridFunction> p;
  
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

    void UpdateVisualizationFields();
    void SaveVisualization();
    CheckpointCompatibility CurrentCheckpointCompatibility() const;
    void LoadCheckpoint();
    void SaveCheckpoint();
  
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
