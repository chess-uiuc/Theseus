// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "Simulation.hpp"
#include "AxisymmetryConfig.hpp"
#include "SimFactory.hpp"
#include "StateInit.hpp"
#include "json.hpp"
#include <cstdint>
#include <filesystem>
#include <mpi.h>

namespace
{
  class ScopedGeometryRefinerType
  {
  private:
    int previous_type_;

  public:
    explicit ScopedGeometryRefinerType(int type)
      : previous_type_(mfem::GlobGeometryRefiner.GetType())
    {
      mfem::GlobGeometryRefiner.SetType(type);
    }

    ~ScopedGeometryRefinerType()
    {
      mfem::GlobGeometryRefiner.SetType(previous_type_);
    }
  };
}

namespace Theseus
{

  Simulation& Simulation::SimulationCreate(std::string device_cfg)
  {
    static Simulation sim(device_cfg);
    return sim;
  }

  void Simulation::InitDevice(std::string device_cfg)
  {
    if(device_cfg.empty() || device_cfg == "off"){
      device_cfg = "cpu";
    }
    if (myRank == 0) {
      std::cout << "Initializing compute device: " << device_cfg << std::endl;
    }
    if(!device_){
      device_ = std::make_unique<mfem::Device>(device_cfg);
    }
    if(myRank == 0){
      std::cout << "Theseus/MFEM device configuration: " << std::endl;
      device_->Print();
    }
  }

  Simulation::Simulation(std::string device_cfg)
  {
    mfem::Mpi::Init();
    numProcs = mfem::Mpi::WorldSize();
    myRank = mfem::Mpi::WorldRank();
    mfem::Hypre::Init();
    InitDevice(device_cfg);

    if (mfem::Mpi::Root())
      {
        std::cout << "================================================" << std::endl;
        std::cout << "Theseus Simulation with " << numProcs << " MPI ranks" << std::endl;
        std::cout << "================================================" << std::endl;

#ifdef AXISYMMETRIC
        std::cout << "Simulation is Axisymmetric" << std::endl;
#endif

#ifdef SUBCELL_FV_BLENDING
        std::cout << "Subcell Blending: ON" << std::endl;
#else
        std::cout << "Subcell Blending: OFF" << std::endl;
#endif
      }
  }

  Simulation::~Simulation()
  {
    if (mfem::Mpi::Root())
      {
        std::cout << "================================================" << std::endl;
        std::cout << "Theseus Simulation completed" << std::endl;
        std::cout << "================================================" << std::endl;
      }
  }

  constexpr bool debug_simulation = false;

  int Simulation::LoadConfig(const std::string &config_file_path)
  {
    std::ifstream config_file(config_file_path);
    if (!config_file.is_open())
      {
        std::cerr << "Error Opening Configuration File at " << config_file_path << std::endl;
        return 1;
      }

    nlohmann::json config;
    config_file >> config;
    auto runtime = config["runTime"];

#ifdef AXISYMMETRIC
    constexpr bool axisymmetric_build = true;
#else
    constexpr bool axisymmetric_build = false;
#endif
    try
      {
        AxisymmetryConfig::Validate(config, runtime, axisymmetric_build);
      }
    catch (const std::exception &error)
      {
        if (mfem::Mpi::Root())
          {
            std::cerr << "Invalid axisymmetry configuration: "
                      << error.what() << std::endl;
          }
        return 1;
      }

    order = runtime.value("order", 3);
    dim = runtime.value("dim", 2);
    num_equations = runtime.value("num_equations", 4);

    precision = runtime.value("precision", 15);
    std::cout.precision(precision);

    cfl = runtime.value("cfl", 1.0);
    print_interval = runtime.value("print_interval", debug_simulation ?  1 : 100);
    output_file_path = runtime["output_file_path"].get<std::string>();
    paraview_folder = runtime.value("paraview_folder", "ParaView");
    try
      {
        checkpoint_config = CheckpointConfig::FromRuntime(runtime, output_file_path);
      }
    catch (const std::exception &error)
      {
        if (mfem::Mpi::Root())
          {
            std::cerr << "Invalid checkpoint configuration: " << error.what() << std::endl;
          }
        return 1;
      }
    if (checkpoint_config.SaveEnabled())
      {
        std::error_code ec;
        std::filesystem::create_directories(checkpoint_config.Directory(), ec);
        if (ec)
          {
            if (mfem::Mpi::Root())
              {
                std::cerr << "Failed to create checkpoint directory "
                          << checkpoint_config.Directory() << ": " << ec.message() << std::endl;
              }
            return 1;
          }
      }

    visualize = runtime["visualize"].get<bool>();
    if (visualize)
      {
#ifdef SUBCELL_FV_BLENDING
        constexpr bool blending_available = true;
#else
        constexpr bool blending_available = false;
#endif
        try
          {
            visualization_config = VisualizationConfig::FromRuntime(runtime, blending_available);
          }
        catch (const std::invalid_argument &error)
          {
            if (mfem::Mpi::Root())
              {
                std::cerr << "Invalid visualization configuration: " << error.what() << std::endl;
              }
            return 1;
          }

        save_dt1 = runtime.value("initial_save_dt", 0.01);
        save_dt2 = runtime.value("refined_save_dt", save_dt1);
        trigger_t = runtime.value("refined_trigger", 2.0);
        save_dt = save_dt1;
        vis_steps = runtime.value("vis_steps", 100);
        paraview = runtime["paraview"].get<bool>();
        visit = runtime["visit"].get<bool>();
        paraview = !visit; // default to paraview
      }

    nancheck = runtime["nancheck"].get<bool>();
    if(debug_simulation)
      nancheck = true;
    if (nancheck)
      {
        nancheck_steps = runtime.value("nancheck_steps", debug_simulation ? 1 : 1000);
        if(mfem::Mpi::Root()){
          std::cout << "Checking for NANs every " << nancheck_steps << " steps" << std::endl;
        }
      }

    clock_simulation = runtime["clock_simulation"].get<bool>();
    variable_dt = runtime["variable_dt"].get<bool>();

    if (runtime.contains("dt") && !variable_dt)
      {
        dt = runtime.value("dt", 1e-4);
      }
    t_final = runtime["final_time"].get<mfem::real_t>();

    nsteps_max = -1;
    if (runtime.contains("nsteps_max")){
      nsteps_max = runtime["nsteps_max"].get<int>();
    }

    std::string ode_solver_string = runtime["ode_solver"].get<std::string>();
    if (ode_solver_string == "ForwardEuler")
      {
        ode_solver = std::make_shared<mfem::ForwardEulerSolver>();
      }
    else if (ode_solver_string == "RK2")
      {
        ode_solver = std::make_shared<mfem::RK2Solver>();
      }
    else if (ode_solver_string == "RK3SSP")
      {
        ode_solver = std::make_shared<mfem::RK3SSPSolver>();
      }
    else if (ode_solver_string == "RK4")
      {
        ode_solver = std::make_shared<mfem::RK4Solver>();
      }
    else if (ode_solver_string == "RK6")
      {
        ode_solver = std::make_shared<mfem::RK6Solver>();
      }
    else if (ode_solver_string == "RK8")
      {
        ode_solver = std::make_shared<mfem::RK8Solver>();
      }
    else
      {
        std::cerr << "Error: Invalid ODE solver specified." << std::endl;
        return 1;
      }

    signature = runtime["conditions"]["initial_conditions"].value("signature", 0);
    std::string IC_key = runtime["conditions"]["initial_conditions"].value("function", "LidDrivenCavityIC");

    if (signature == 0)
      {
        u0 = std::make_unique<mfem::VectorFunctionCoefficient>
          (num_equations,
           Prandtl::ConditionFactory::Instance().GetInitialCondition0(IC_key)());
      }
    else if (signature == 1)
      {
        mfem::real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        u0 = std::make_unique<mfem::VectorFunctionCoefficient>
          (num_equations,
           Prandtl::ConditionFactory::Instance().GetInitialCondition1(IC_key)(x1));
      }
    else if (signature == 2)
      {
        mfem::real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        mfem::real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        u0 = std::make_unique<mfem::VectorFunctionCoefficient>
          (num_equations,
           Prandtl::ConditionFactory::Instance().GetInitialCondition2(IC_key)(x1, x2));
      }
    else if (signature == 3)
      {
        mfem::real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        mfem::real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        mfem::real_t x3 = runtime["conditions"]["initial_conditions"]["params"].value("x3", 0.0);
        u0 = std::make_unique<mfem::VectorFunctionCoefficient>
          (num_equations,
           Prandtl::ConditionFactory::Instance().GetInitialCondition3(IC_key)(x1, x2, x3));
      }
    else if (signature == 4)
      {
        mfem::real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        mfem::real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        mfem::real_t x3 = runtime["conditions"]["initial_conditions"]["params"].value("x3", 0.0);
        mfem::real_t x4 = runtime["conditions"]["initial_conditions"]["params"].value("x4", 0.0);
        u0 = std::make_unique<mfem::VectorFunctionCoefficient>
          (num_equations,
           Prandtl::ConditionFactory::Instance().GetInitialCondition4(IC_key)(x1, x2, x3, x4));
      }
    else if (signature == 5)
      {
        mfem::real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        mfem::real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        mfem::real_t x3 = runtime["conditions"]["initial_conditions"]["params"].value("x3", 0.0);
        mfem::real_t x4 = runtime["conditions"]["initial_conditions"]["params"].value("x4", 0.0);
        mfem::real_t x5 = runtime["conditions"]["initial_conditions"]["params"].value("x5", 0.0);
        u0 = std::make_unique<mfem::VectorFunctionCoefficient>
          (num_equations,
           Prandtl::ConditionFactory::Instance().GetInitialCondition5(IC_key)(x1, x2, x3, x4, x5));
      }
    else
      {
        std::cerr << "Error: Invalid initial condition signature." << std::endl;
        return 1;
      }

    if (runtime.contains("exact_solution"))
      {
        const auto &exact = runtime["exact_solution"];
        const std::string key = exact["function"].get<std::string>();
        const int exact_signature = exact.value("signature", 0);
        if (exact_signature == 0)
          {
            exact_solution = std::make_unique<mfem::VectorFunctionCoefficient>(
              num_equations,
              Prandtl::ConditionFactory::Instance().
                GetVectorTDFunctionBoundaryCondition0(key)());
          }
        else if (exact_signature == 1)
          {
            const mfem::real_t x1 = exact["params"].value("x1", 0.0);
            exact_solution = std::make_unique<mfem::VectorFunctionCoefficient>(
              num_equations,
              Prandtl::ConditionFactory::Instance().
                GetVectorTDFunctionBoundaryCondition1(key)(x1));
          }
        else
          {
            std::cerr << "Error: unsupported exact solution signature."
                      << std::endl;
            return 1;
          }
      }

    mfem::Mesh *mesh;
    std::string mesh_file_name(runtime["mesh_file"].get<std::string>());
    {
      Theseus::ScopedTimer timer("ReadMesh");
      mesh = new mfem::Mesh(mesh_file_name);
    }

    bool periodic;
    if (runtime.contains("periodic"))
      {
        periodic = runtime["periodic"].get<bool>();
      }
    else
      {
        periodic = false;
      }

    if (dim == 1 && !periodic)
      {
        mfem::Array<int> left;
        mfem::Array<int> right;
        left.Append(1);
        right.Append(2);
        mesh->bdr_attribute_sets.SetAttributeSet("left", left);
        mesh->bdr_attribute_sets.SetAttributeSet("right", right);
      }
    if (!periodic && mesh->GetNBE() == 0){
      mesh->GenerateBoundaryElements();
    }
    if (runtime.contains("ser_ref_levels"))
      {
        ref_levels = runtime.value("ser_ref_levels", 0);
        if(mfem::Mpi::Root()){
          std::cout << "Serial mesh refinement levels: " << ref_levels << std::endl;
        }
        for (int lev = 0; lev < ref_levels; lev++)
          {
            mesh->UniformRefinement();
          }
      }
    mesh->FinalizeTopology();

    if (mesh->GetNE() < mfem::Mpi::WorldSize())
      {
        std::cerr << "Error: Number of elements is less than number of processors." << std::endl;
        return 1;
      }

    if (runtime.contains("mesh_ordering"))
      {
        if (runtime["mesh_ordering"].get<std::string>() == "Hilbert")
          {
            mesh->GetHilbertElementOrdering(mesh_ordering);
          }
        else if (runtime["mesh_ordering"].get<std::string>() == "Gecko")
          {
            mesh->GetGeckoElementOrdering(mesh_ordering);
          }
        mesh->ReorderElements(mesh_ordering);
      }

    // NOTE: Critical setting for correctness using TPEs
    if (dim > 1 && runtime.value("use_nc_mesh", true))
      {
        mesh->EnsureNCMesh();
      }

    mesh->FinalizeMesh(0, true);
#ifdef AXISYMMETRIC
    try
      {
        AxisymmetryConfig::ValidateMesh(*mesh);
      }
    catch (const std::exception &error)
      {
        if (mfem::Mpi::Root())
          {
            std::cerr << "Invalid axisymmetric mesh: "
                      << error.what() << std::endl;
          }
        delete mesh;
        return 1;
      }
#endif
    pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);
    mesh->Clear();
    delete mesh;

    if(myRank == 0 && debug_simulation){
      std::cout << "Mesh distributed" << std::endl;
    }
    if (runtime.contains("par_ref_levels"))
      {
        ref_levels = runtime.value("par_ref_levels", 0);
        if(mfem::Mpi::Root()){
          std::cout << "Parallel mesh refinement levels: " << ref_levels << std::endl;
        }
        for (int lev = 0; lev < ref_levels; lev++)
          {
            pmesh->UniformRefinement();
          }
      }

    pmesh->ExchangeFaceNbrData();

#ifdef SUBCELL_FV_BLENDING
    fec0 = std::make_shared<mfem::DG_FECollection>(0, dim);
    fes0 = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), fec0.get());
    fes0->ExchangeFaceNbrData();
#endif

    fec = std::make_shared<mfem::DG_FECollection>(order, dim, btype);
    vfes = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), fec.get(), num_equations, ordering);
    dfes = std::make_unique<mfem::ParFiniteElementSpace>(pmesh.get(), fec.get(), dim, ordering);
    fes = std::make_unique<mfem::ParFiniteElementSpace>(pmesh.get(), fec.get());

    // NOTE: Initial exchange populates some data structures we need to build restrictions
    vfes->ExchangeFaceNbrData();
    dfes->ExchangeFaceNbrData();
    fes->ExchangeFaceNbrData();

    num_dofs_scalar = fes->GetNDofs();
    num_dofs_system = vfes->GetVSize();
    std::int64_t ndofscalar = fes->GetNDofs();
    std::int64_t ndofsys = vfes->GetVSize();
    int points_per_element = std::pow(order+1, dim);
    std::int64_t num_elements_local = ndofscalar / points_per_element;
    std::int64_t num_elements_total = num_elements_local;

if (debug_simulation && numProcs > 1) {
      for(int irank = 0;irank < numProcs;irank++){
       if(myRank == irank){
         std::cout << "Rank(" << myRank << ") Number of elements: "
                   << num_elements_local << std::endl;
       }
       MPI_Barrier(pmesh->GetComm());
      }
    }
    if(numProcs > 1){
      std::int64_t max_nel = num_elements_local;
      std::int64_t  min_nel = num_elements_local;
      MPI_Allreduce(MPI_IN_PLACE, &max_nel, 1, MPI_LONG_LONG, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &min_nel, 1, MPI_LONG_LONG, MPI_MIN, pmesh->GetComm());
      if(myRank == 0){
       std::cout << "Partition NumElements (min, max) = (" << min_nel << ", " << max_nel
                 << ")" << std::endl;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &num_elements_total, 1, MPI_LONG_LONG, MPI_SUM, pmesh->GetComm());

    if(myRank == 0 && debug_simulation){
      std::cout << "Initial exchanges complete." << std::endl;
    }

    std::vector<std::shared_ptr<mfem::ParGridFunction> > grad_u(dim);
    for(int idim = 0;idim < dim;idim++)
      grad_u[idim] = std::make_shared<mfem::ParGridFunction>(vfes.get());

    if (checkpoint_config.LoadEnabled())
      {
        LoadCheckpoint();
      }
    else
      {
        sol = std::make_shared<mfem::ParGridFunction>(vfes.get());
        sol->ProjectCoefficient(*u0);

        t = 0.0;
        ti = 0;

        if(myRank == 0 && debug_simulation){
          std::cout << "Run is not a restart: (step=0, t=0)" << std::endl;
        }
      }

#ifdef SUBCELL_FV_BLENDING
    mfem::Geometry::Type gtype = vfes->GetFE(0)->GetGeomType();
    eta = std::make_shared<mfem::ParGridFunction>(fes0.get());
    alpha = std::make_shared<mfem::ParGridFunction>(fes0.get());
    indicator = std::make_shared<Prandtl::PerssonPeraireIndicator>
      (vfes, fes0, eta, std::make_unique<Prandtl::ModalBasis>(*fec, gtype, order, dim));
    if (runtime.contains("alpha_max"))
      {
        alpha_max = runtime["alpha_max"].get<mfem::real_t>();
      }
    else
      {
        alpha_max = 0.5;
      }
    *alpha = 0.0;
#endif

    rhsOp = Theseus::MakeRHSOperator(runtime, vfes, fes0, pmesh, eta, alpha, grad_u,
                                     indicator, alpha_max);

    if(!rhsOp){
      std::cerr << "Failed to create RHS Operator." << std::endl;
      return 1;
    } else if(myRank == 0 && debug_simulation){
      std::cout << "RHS Operator created." << std::endl;
    }
    if(myRank == 0){
      std::cout << "Theseus RHS Operator:" << std::endl
                << "FlowModel: " << rhsOp->FlowModelName() << std::endl
                << "GasModel:  " << rhsOp->GasModelName() << std::endl
                << "NumFlux:   " << rhsOp->NumFluxName() << std::endl;
    }

    const auto &gasModel = rhsOp->GetGasModelInterface();
    auto stateLayout = gasModel.layout();
    auto physicsConstants = gasModel.phys();

    mfem::Vector bc_vector_data;
    mfem::Vector bc_scalar_data;
    mfem::Array<Theseus::BCDescriptor> bc_descriptors;

    if (runtime["conditions"].contains("boundary_conditions"))
      {
        max_bdr_attr = pmesh->bdr_attributes.Max();
        bdr_marker_vector.reserve(max_bdr_attr + 1);
        auto boundaries = runtime["conditions"]["boundary_conditions"];
        for (auto& boundary : boundaries.items())
          {
            std::string boundaryName = boundary.key();
            if (!pmesh->bdr_attribute_sets.AttributeSetExists(boundaryName))
              {
                // This rank has no faces with this boundary name; skip
                continue;
              }
            mfem::Array<int> marker(max_bdr_attr);
            marker = 0;
            // bdr_marker_vector.push_back(Array<int>(max_bdr_attr));
            set_marker = pmesh->bdr_attribute_sets.GetAttributeSetMarker(boundaryName);
            for (int b = 0; b < max_bdr_attr; b++)
              {
                if (set_marker[b])
                  {
                    // bdr_marker_vector.back()[b] = 1;
                    marker[b] = 1;
                  }
              }
            bdr_marker_vector.push_back(marker);

            Theseus::BCDescriptor bc_descr{};
            bc_descr.flags = 0;
            bc_descr.data_index = -1;
            bc_descr.bdr_attr = -1;
            bc_descr.data_kind = int(Theseus::BCDataKind::None);
            bc_descr.type = int(Theseus::BCType::Invalid);

            auto bc_props = boundary.value();  // This is a JSON object.
            std::string type = bc_props["type"].get<std::string>();

            if (type == "symmetry")
              {
                rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                bc_descr.type = int(Theseus::BCType::Symmetry);
                bc_descr.data_kind = int(Theseus::BCDataKind::None);
              }
            else if (type == "axis")
              {
#ifndef AXISYMMETRIC
                MFEM_ABORT("AXIS BC requires axisymmetry build.");
#else
                rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                bc_descr.type = int(Theseus::BCType::Axis);
                bc_descr.data_kind = int(Theseus::BCDataKind::None);
#endif
              }
            else if (type == "slip")
              {
                rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                bc_descr.type = int(Theseus::BCType::SlipWall);
                bc_descr.data_kind = int(Theseus::BCDataKind::None);
              }
            else if (type == "no-slip-adiabatic")
              {
                // if(debug_simulation && mfem::Mpi::Root()){
                //  std::cout << "Detected noslip boundary... configuring." << std::endl;
                // }
                bc_descr.type = int(Theseus::BCType::NoSlipAdiab);
                bc_descr.data_kind = int(Theseus::BCDataKind::VectorAndScalarConstant);
                if (bc_props["velocity"].contains("vector"))
                  {
                    std::string velBC_key = bc_props["velocity"]["vector"].get<std::string>();
                    std::string heatBC_key = bc_props["heat"]["scalar"].get<std::string>();
                    // std::string state_key = bc_props["vector"].get<std::string>();
                    auto vel_bc = Prandtl::ConditionFactory::Instance().GetVectorBoundaryCondition(velBC_key);
                    auto heat_bc = Prandtl::ConditionFactory::Instance().GetScalarBoundaryCondition(heatBC_key);

                    mfem::Vector bc_data(vel_bc.Size() + 1);
                    std::ostringstream Ostr;
                    Ostr << "Wall velocity: < ";
                    for(int ivec = 0;ivec < vel_bc.Size();ivec++){
                      bc_data[ivec] = vel_bc[ivec];
                      Ostr << vel_bc[ivec] << " ";
                    }
                    Ostr << ">" << std::endl;
                    Ostr << "Heat: " << heat_bc << std::endl;
                    bc_data[vel_bc.Size()] = heat_bc;
                    bc_descr.data_index = Theseus::AppendBCVectorPayload(bc_vector_data, bc_data);
                    Ostr << "bc_vector_data index: " << bc_descr.data_index << std::endl
                         << "BC Data So Far: [";
                    for(int ivec=0;ivec < bc_vector_data.Size();ivec++){
                      Ostr << bc_vector_data[ivec] << " ";
                    }
                    Ostr << "]" << std::endl;
                    // std::cout << Ostr.str();
                    rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                  }
                else if (bc_props["velocity"].contains("function"))
                  {
                    std::cerr << "Unsupported BC type: noslip-adiabatic w/function or time dep." << std::endl;
                    return 1;
                    std::shared_ptr<mfem::VectorFunctionCoefficient> velBC;
                    std::shared_ptr<mfem::FunctionCoefficient> heatBC;

                    std::string velBC_key = bc_props["velocity"]["function"].get<std::string>();
                    std::string heatBC_key = bc_props["heat"]["function"].get<std::string>();

                    bool td;
                    if (bc_props["velocity"].contains("time_dependent"))
                      {
                        td = bc_props["velocity"]["time_dependent"].get<bool>();
                      }
                    else
                      {
                        td = false;
                      }

                    signature = bc_props["velocity"]["signature"].get<int>();
                    if (signature == 0)
                      {
                        velBC = std::make_shared<mfem::VectorFunctionCoefficient>
                          (dim, Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition0(velBC_key)());
                      }
                    else if (signature == 1)
                      {
                        mfem::real_t x1 = bc_props["velocity"]["params"].value("x1", 0.0);
                        velBC = std::make_shared<mfem::VectorFunctionCoefficient>
                          (dim, Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition1(velBC_key)(x1));
                      }
                    else if (signature == 2)
                      {
                        mfem::real_t x1 = bc_props["velocity"]["params"].value("x1", 0.0);
                        mfem::real_t x2 = bc_props["velocity"]["params"].value("x2", 0.0);
                        velBC = std::make_shared<mfem::VectorFunctionCoefficient>
                          (dim, Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition2(velBC_key)(x1, x2));
                      }
                    else
                      {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return 1;
                      }

                    signature = bc_props["heat"]["signature"].get<int>();
                    if (signature == 0)
                      {
                        if (td)
                          {
                            heatBC = std::make_shared<mfem::FunctionCoefficient>
                              (Prandtl::ConditionFactory::Instance().GetScalarTDFunctionBoundaryCondition0(heatBC_key)());
                          }
                        else
                          {
                            heatBC = std::make_shared<mfem::FunctionCoefficient>
                              (Prandtl::ConditionFactory::Instance().GetScalarFunctionBoundaryCondition0(heatBC_key)());
                          }
                      }
                    else if (signature == 1)
                      {
                        mfem::real_t x1 = bc_props["heat"]["params"].value("x1", 0.0);
                        if (td)
                          heatBC = std::make_shared<mfem::FunctionCoefficient>
                            (Prandtl::ConditionFactory::Instance().GetScalarTDFunctionBoundaryCondition1(heatBC_key)(x1));
                        else
                          {
                            heatBC = std::make_shared<mfem::FunctionCoefficient>
                              (Prandtl::ConditionFactory::Instance().GetScalarFunctionBoundaryCondition1(heatBC_key)(x1));
                          }
                      }
                    else if (signature == 2)
                      {
                        mfem::real_t x1 = bc_props["heat"]["params"].value("x1", 0.0);
                        mfem::real_t x2 = bc_props["heat"]["params"].value("x2", 0.0);
                        if (td)
                          heatBC = std::make_shared<mfem::FunctionCoefficient>
                            (Prandtl::ConditionFactory::Instance().GetScalarTDFunctionBoundaryCondition2(heatBC_key)(x1, x2));
                        else
                          {
                            heatBC = std::make_shared<mfem::FunctionCoefficient>
                              (Prandtl::ConditionFactory::Instance().GetScalarFunctionBoundaryCondition2(heatBC_key)(x1, x2));
                          }
                      }
                    else
                      {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return 1;
                      }
                    rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());

                  }
                else
                  {
                    std::cerr << "Error: Invalid boundary condition type specified." << std::endl;
                    return 1;
                  }
              }
            else if (type == "no-slip-isothermal")
              {
                bc_descr.type = int(Theseus::BCType::NoSlipIso);
                bc_descr.data_kind = int(Theseus::BCDataKind::VectorAndScalarConstant);
if (!(bc_props.contains("velocity") && bc_props["velocity"].contains("vector") &&
      bc_props.contains("temperature") && bc_props["temperature"].contains("scalar")))
  { std::cerr << "Error: no-slip-isothermal requires velocity.vector and temperature.scalar." << std::endl; return 1; }
{
                    std::string velBC_key = bc_props["velocity"]["vector"].get<std::string>();
                    std::string tempBC_key = bc_props["temperature"]["scalar"].get<std::string>();
                    // std::string state_key = bc_props["vector"].get<std::string>();
                    auto vel_bc = Prandtl::ConditionFactory::Instance().GetVectorBoundaryCondition(velBC_key);
                    auto temp_bc = Prandtl::ConditionFactory::Instance().GetScalarBoundaryCondition(tempBC_key);

                    mfem::Vector bc_data(vel_bc.Size() + 1);
                    std::ostringstream Ostr;
                    Ostr << "Wall velocity: < ";
                    for(int ivec = 0;ivec < vel_bc.Size();ivec++){
                      bc_data[ivec] = vel_bc[ivec];
                      Ostr << vel_bc[ivec] << " ";
                    }
                    Ostr << ">" << std::endl;
                    Ostr << "Wall temperature: " << temp_bc << std::endl;
                    bc_data[vel_bc.Size()] = temp_bc;
                    bc_descr.data_index = Theseus::AppendBCVectorPayload(bc_vector_data, bc_data);
                    Ostr << "bc_vector_data index: " << bc_descr.data_index << std::endl
                         << "BC Data So Far: [";
                    for(int ivec=0;ivec < bc_vector_data.Size();ivec++){
                      Ostr << bc_vector_data[ivec] << " ";
                    }
                    Ostr << "]" << std::endl;
		    if(debug_simulation && mfem::Mpi::Root()){
		      std::cout << Ostr.str();
		    }
                    rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                  }
              }
            else if (type == "supersonic-outflow")
              {
                rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                bc_descr.type = int(Theseus::BCType::SupersonicOutflow);
                bc_descr.data_kind = int(Theseus::BCDataKind::None);
              }
            else if (type == "supersonic-inflow")
              {
                if (bc_props.contains("vector"))
                  {
                    std::string state_key = bc_props["vector"].get<std::string>();
                    mfem::Vector bc_state = Prandtl::ConditionFactory::Instance().GetVectorBoundaryCondition(state_key);
                    rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                    const int data_offset = Theseus::AppendBCVectorPayload(bc_vector_data, bc_state);
                    bc_descr.type = int(Theseus::BCType::SupersonicInflow);
                    bc_descr.data_kind = int(Theseus::BCDataKind::VectorConstant);
                    bc_descr.data_index = data_offset;
                  }
                else
                  {
                    std::cerr << "Unsupported BC type: time-dependent" << std::endl;
                    return 1;
                    std::string state_key = bc_props["function"].get<std::string>();
                    signature = bc_props["signature"].get<int>();
                    std::shared_ptr<mfem::VectorFunctionCoefficient> stateBC;
                    bool td;
                    if (bc_props.contains("time_dependent"))
                      {
                        td = bc_props["time_dependent"].get<bool>();
                      }
                    else
                      {
                        td = false;
                      }

                    if (signature == 0)
                      {
                        if (td)
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition0(state_key)());
                          }
                        else
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition0(state_key)());
                          }

                      }
                    else if (signature == 1)
                      {
                        mfem::real_t x1 = bc_props["params"].value("x1", 0.0);
                        if (td)
                          stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                            (num_equations,
                             Prandtl::ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition1(state_key)(x1));
                        else
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition1(state_key)(x1));
                          }
                      }
                    else if (signature == 2)
                      {
                        mfem::real_t x1 = bc_props["params"].value("x1", 0.0);
                        mfem::real_t x2 = bc_props["params"].value("x2", 0.0);
                        if (td)
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition2(state_key)(x1, x2));
                          }
                        else
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition2(state_key)(x1, x2));
                          }

                      }
                    else
                      {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return 1;
                      }
                    rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                  }
              }
            else if (type == "specified-state")
              {
                std::cerr << "Unsupported BC type: specified-state" << std::endl;
                return 1;
                if (bc_props.contains("vector"))
                  {
                    std::string state_key = bc_props["vector"].get<std::string>();
                    rhsOp->AddBdrFaceMarker(bdr_marker_vector.back());
                  }
                else
                  {
                    std::string state_key = bc_props["function"].get<std::string>();
                    signature = bc_props["signature"].get<int>();
                    std::shared_ptr<mfem::VectorFunctionCoefficient> stateBC;
                    bool td;
                    if (bc_props.contains("time_dependent"))
                      {
                        td = bc_props["time_dependent"].get<bool>();
                      }
                    else
                      {
                        td = false;
                      }

                    if (signature == 0)
                      {
                        if (td)
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition0(state_key)());
                          }
                        else
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition0(state_key)());
                          }

                      }
                    else if (signature == 1)
                      {
                        mfem::real_t x1 = bc_props["params"].value("x1", 0.0);
                        if (td)
                          stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                            (num_equations,
                             Prandtl::ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition1(state_key)(x1));
                        else
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition1(state_key)(x1));
                          }
                      }
                    else if (signature == 2)
                      {
                        mfem::real_t x1 = bc_props["params"].value("x1", 0.0);
                        mfem::real_t x2 = bc_props["params"].value("x2", 0.0);
                        if (td)
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition2(state_key)(x1, x2));
                          }
                        else
                          {
                            stateBC = std::make_shared<mfem::VectorFunctionCoefficient>
                              (num_equations,
                               Prandtl::ConditionFactory::Instance().GetVectorFunctionBoundaryCondition2(state_key)(x1, x2));
                          }

                      }
                    else
                      {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return 1;
                      }
                  }
              }
            else
              {
                std::cerr << "Error: Invalid boundary condition type specified." << std::endl;
                return 1;
              }
            bc_descriptors.Append(bc_descr);
          }
      }

    if(myRank == 0 && debug_simulation){
      std::cout << "Boundary conditions configured." << std::endl;
    }

    // Set up the operator cache
    rhsOp->SetBCDescriptorData(bc_descriptors, bc_scalar_data, bc_vector_data);
    rhsOp->Finalize(t);

    if(myRank == 0 && debug_simulation){
      std::cout << "Theseus RHS Operator finalized." << std::endl;
    }

    if (mfem::Mpi::Root())
      {
        std::cout << "The Number of Equations being Solved: " << num_equations << std::endl
		  << "The Total Number of Order " << order << " Elements in the Simulation: "
		  << num_elements_total << std::endl
		  << "The Number of Nodes per Element: "
		  << points_per_element << std::endl
		  << "The Total Number of DOFs in the Simulation (All Eqns/All Ranks): "
                  << num_elements_total*points_per_element*num_equations << std::endl
		  << "Per Rank Averages:" << std::endl
                  << "  Number of elements:     " << num_elements_total / numProcs << std::endl
                  << "  Number of nodes:        " << num_dofs_scalar << std::endl
                  << "  Total DOFs (all eqns):  " << num_dofs_system << std::endl;
      }

    ode_solver->Init(*rhsOp);

    rho.MakeRef(fes.get(), *sol, offset_mass(stateLayout));
    mom.MakeRef(dfes.get(), *sol, offset_momentum(stateLayout));
    energy.MakeRef(fes.get(), *sol, offset_energy(stateLayout));

    velocity = std::make_unique<mfem::ParGridFunction>(dfes.get());
    p = std::make_unique<mfem::ParGridFunction>(fes.get());

    if (visualize)
      {
        if (paraview &&
            visualization_config.MeshMode() == VisualizationMeshMode::gll_subcells)
          {
            const mfem::Geometry::Type tensor_geometry =
              dim == 1 ? mfem::Geometry::SEGMENT :
              dim == 2 ? mfem::Geometry::SQUARE : mfem::Geometry::CUBE;
            for (int element = 0; element < pmesh->GetNE(); element++)
              {
                MFEM_VERIFY(pmesh->GetElementBaseGeometry(element) == tensor_geometry,
                            "Visualization mesh mode 'gll_subcells' supports only "
                            "tensor-product segment, quadrilateral, or hexahedral elements");
              }
          }

        if (paraview)
          {
            pd = std::make_unique<mfem::ParaViewDataCollection>(paraview_folder, pmesh.get());
            pd->SetPrefixPath(output_file_path);
            if (visualization_config.Has(VisualizationField::density))
              {
                pd->RegisterField("Density", &rho);
              }
            if (visualization_config.Has(VisualizationField::velocity))
              {
                pd->RegisterField("Velocity", velocity.get());
              }
            if (visualization_config.Has(VisualizationField::pressure))
              {
                pd->RegisterField("Pressure", p.get());
              }
#ifdef SUBCELL_FV_BLENDING
            if (visualization_config.Has(VisualizationField::blending_coefficient))
              {
                pd->RegisterField("Blending Coeff", alpha.get());
              }
#endif
            pd->SetLevelsOfDetail(order);
            pd->SetDataFormat(mfem::VTKFormat::BINARY);
            pd->SetHighOrderOutput(
              visualization_config.MeshMode() == VisualizationMeshMode::vtk_high_order);
          }
        else if (visit)
          {
            vd = std::make_unique<mfem::VisItDataCollection>("VisIt", pmesh.get());
            vd->SetPrefixPath(output_file_path);
            vd->SetPrecision(precision);
            vd->SetFormat(mfem::DataCollection::PARALLEL_FORMAT);

            if (visualization_config.Has(VisualizationField::density))
              {
                vd->RegisterField("Density", &rho);
              }
            if (visualization_config.Has(VisualizationField::velocity))
              {
                vd->RegisterField("Velocity", velocity.get());
              }
            if (visualization_config.Has(VisualizationField::pressure))
              {
                vd->RegisterField("Pressure", p.get());
              }
#ifdef SUBCELL_FV_BLENDING
            if (visualization_config.Has(VisualizationField::blending_coefficient))
              {
                vd->RegisterField("Blending Coeff", alpha.get());
              }
#endif
          }
      }

    next_checkpoint_t = t + checkpoint_config.Interval();
    if (visualize) { next_save_t = t + save_dt; }

    return 0;
  }

  void Simulation::Run()
  {
    if (mfem::Mpi::Root())
      {
        std::cout << "================================================" << std::endl
                  << "Theseus Simulation Running Now" << std::endl
                  << "================================================" << std::endl;
      }
    const auto &gasModel = rhsOp->GetGasModelInterface();
    auto stateLayout = gasModel.layout();
    auto physicsConstants = gasModel.phys();

    IntegralMeasures diag;
    diag.mass = 0.0;
    diag.ke = 0.0;
    diag.en = 0.0;
    diag.max_press = 0.0;
    diag.min_press = 0.0;
    diag.max_dens = 0.0;
    diag.min_dens = 0.0;
    diag.max_temp = 0.0;
    diag.min_temp = 0.0;

    rhsOp->ComputeIntegralMeasures(*sol, diag);

    Theseus::IntegralMeasures diag0 = rhsOp->GetIntegralMeasuresBaseline();

    // Get the minimum characteristic element size and compute the initial time step if time step is variable
    mfem::real_t heff = mfem::infinity();
    hmin = mfem::infinity();
    for (int i = 0; i < pmesh->GetNE(); i++)
      {
        hmin = std::min(pmesh->GetElementSize(i, 1), hmin);
      }
    if(mfem::Mpi::Root() && debug_simulation){
      std::cout << "Found minimum cell size: " << hmin << std::endl;
    }
    MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type,
                  MPI_MIN, pmesh->GetComm());
    // Asymptotically should be hmin / (p+1)^2 due to node clustering, but is pretty wrong at low
    // order.  This form attempts to smoothly transition to asymptotic form with increasing order
    mfem::real_t p1 = order + 1;
    mfem::real_t alpha1 = std::min(mfem::real_t(1.0),
                                   std::max(mfem::real_t(0.0), (p1 - 3.0) / 3.0));
    heff = hmin / ((1.0 - alpha1) * p1 + alpha1 * p1 * p1);

    if (debug_simulation && mfem::Mpi::Root()){
      std::cout << "Minimum nodal spacing (h_eff): " << heff << std::endl;
    }
    const mfem::real_t nuscale = \
      std::max(1.0, physicsConstants.gamma/physicsConstants.Pr);
#ifdef PARABOLIC
    if (debug_simulation && mfem::Mpi::Root()){
      std::cout << "Viscosity scale (nuscale): " << nuscale << std::endl;
    }
#endif
    if (variable_dt && cfl > 0.0)
      {
        mfem::Vector z(sol->Size());
        rhsOp->Mult(*sol, z);
        mfem::real_t max_char_speed = rhsOp->GetMaxCharSpeed();
        MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1,  mfem::MPITypeMap<mfem::real_t>::mpi_type,
                      MPI_MAX, pmesh->GetComm());
        mfem::real_t dt_adv = heff / max_char_speed;
        dt = cfl / dim * dt_adv;
        if(debug_simulation && mfem::Mpi::Root()){
          std::cout << "Found max_char_speed = " << max_char_speed << std::endl
                    << "Advective timestep = " << dt_adv << std::endl
                    << "Inviscid DT = " << dt << std::endl;
        }
#ifdef PARABOLIC
        mfem::real_t nu_eff = nuscale * physicsConstants.mu / diag0.min_dens;
        mfem::real_t dt_diff = heff * heff / nu_eff;
        dt = cfl / dim / (1.0/dt_adv + 1.0/dt_diff);
        if(debug_simulation && mfem::Mpi::Root()){
          std::cout << "Found minimum density: " << diag0.min_dens << std::endl
                    << "Found effective visc: " << nu_eff << std::endl
                    << "Diffusive timestep: " << dt_diff << std::endl;
        }
#endif
        if(mfem::Mpi::Root()){
          std::cout << "Fixed CFL: " << cfl << std::endl
                    << "Initial Timestep DT: " << dt << std::endl;
        }
      } else {
      if(mfem::Mpi::Root()){
        std::cout << "Fixed Timestep DT: " << dt << std::endl;
      }
    }

    // Clock the simulation?
    if (clock_simulation)
      {
        mfem::tic_toc.Clear();
        mfem::tic_toc.Start();
      }

    if (mfem::Mpi::Root())
      {
        int i = 0;
        const mfem::real_t *sol_state = sol->HostRead();
        Theseus::DofStateView dofState{sol_state, i};
        mfem::real_t rhoi = gasModel.density(dofState);
        mfem::real_t ui = gasModel.velocity(dofState, 0);
        mfem::real_t vi = dim > 1 ? gasModel.velocity(dofState, 1) : 0.0;
        mfem::real_t wi = dim > 2 ? gasModel.velocity(dofState, 2) : 0.0;
        mfem::real_t pi = gasModel.pressure(dofState);
        std::cout << "***  initial at dof #" << i << ":  "
                  << "rho = " << std::round(rhoi*10000)/10000 << ",  velocity = <"
                  << std::round(ui*100)/100;
        if(dim > 1){
          std::cout << ", " << std::round(vi*100)/100;
          if(dim > 2){
            std::cout << ", " << std::round(wi*100)/100;
          }
        }
        std::cout << ">, p = " << std::round(pi*100)/100 << std::endl
                  << "Total Mass:           " << diag0.mass << std::endl
                  << "Total Energy:         " << diag0.en << std::endl
                  << "Total Kinetic Energy: " << diag0.ke << std::endl;
      }
    // Visualize the initial condition?
    if (visualize)
      {
        if(mfem::Mpi::Root()){
          std::cout << "Writing initial soln..." << std::endl;
        }
        Theseus::ScopedTimer timer("VisInit");

        UpdateVisualizationFields();

        SaveVisualization();
      }


    while (!done)
      {

        if (debug_simulation)
          {
            MPI_Barrier(pmesh->GetComm());
            if(mfem::Mpi::Root()){
              std::cout << "############################################"
                        << std::endl
                        << "[TIME STEP = " << ti << ", TIME = " << t << "]"
                        << std::endl
                        << "############################################"
                        << std::endl;
            }
          }

        // Compute the time step size
        dt_real = std::min(dt, t_final - t);

        // Perform the time step
        {
          Theseus::ScopedTimer timer("Timestep");
          ode_solver->Step(*sol, t, dt_real);
        }
        ti++;

        mfem::real_t cfl_rep = 0.0;
        if (ti % print_interval == 0 || (variable_dt && cfl > 0) || debug_simulation){
          rhsOp->ComputeIntegralMeasures(*sol, diag);
        }
        // Update the time step size with CFL?
        if ((variable_dt && cfl > 0) || (ti%print_interval == 0) || debug_simulation)
          {
            mfem::real_t max_char_speed = rhsOp->GetMaxCharSpeed();
            MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type,
                          MPI_MAX, pmesh->GetComm());
            mfem::real_t dt_adv = heff / max_char_speed;
            mfem::real_t dt_est = dt_adv;
#ifdef PARABOLIC
            mfem::real_t nu_eff = nuscale * physicsConstants.mu / diag.min_dens;
            mfem::real_t dt_diff = heff * heff / nu_eff;
            mfem::real_t dt_m1 = 1.0 / (1.0/dt_adv + 1.0/dt_diff);
            dt_est = dt_m1;
#endif
            if(variable_dt){
              dt = cfl / dim * dt_est;
            } else {
              cfl_rep = dim * dt / dt_est;
            }

            if(debug_simulation && mfem::Mpi::Root()){
#ifdef PARABOLIC
              std::cout << "DT(adv, diff, sim): (" << dt_adv << ", " << dt_diff
                        << ", " << dt << ")" << std::endl
                        << "Effective viscosity: " << nu_eff << std::endl;
#else
              std::cout << "DT(adv, sim): (" << dt_adv << ", " << dt << ")" << std::endl;
#endif
              if(!variable_dt){
                std::cout << "CFL: "<< cfl_rep << std::endl;
              }
              std::cout << "Max wavespeed: " << max_char_speed << std::endl
                        << "Max specific volume: " << 1.0 / diag.min_dens << std::endl;
            }

          }

        // Check for completion
        done = ((t >= t_final - 1e-8 * dt) || StepLimitReached(ti, nsteps_max));

        // Check for NaN/Inf values?
        rho.HostRead();
        if (nancheck && ti % nancheck_steps == 0)
          {
            for (const mfem::real_t &val : rho)
              {
                if (std::isnan(val) || std::isinf(val))
                  {
                    MFEM_ABORT("NaN/Inf Detected at Time Step " + std::to_string(ti) +
                               " on Rank " + std::to_string(myRank));
                    break;
                  }
              }
          }
        // Visualize the solution?
        if (visualize && (done || t >= next_save_t || ti % vis_steps == 0))
          {

            UpdateVisualizationFields();

            SaveVisualization();


            save_dt = (t < trigger_t) ? save_dt1 : save_dt2;
            next_save_t += save_dt;

          }


        if (checkpoint_config.SaveEnabled() && (done || t >= next_checkpoint_t))
          {
            SaveCheckpoint();
            next_checkpoint_t += checkpoint_config.Interval();
          }

        if (ti % print_interval == 0 || debug_simulation)
          {
            mfem::real_t ke0 = diag0.ke;
            if(ke0 == 0.0){ ke0 = 1.0; };
            if (mfem::Mpi::Root())
              {
                std::ostringstream Ostr;
                Ostr << "time step: " << ti << ", time: " << t;
                if(variable_dt){
                  Ostr << ", dt: " << dt;
                } else {
                  Ostr << ", cfl: " << cfl_rep;
                }
                Ostr << std::endl
                     << "rho(" << diag.min_dens << "," << diag.max_dens << "), "
                     << "p(" << diag.min_press << "," << diag.max_press << "), "
                     << "T(" << diag.min_temp << "," << diag.max_temp << ")" << std::endl
                     << "TotalChange: Mass: " << (diag.mass - diag0.mass) / diag0.mass
                     << ", Energy: " <<  (diag.en - diag0.en) / diag0.en
                     << ", K.E.: " << (diag.ke - diag0.ke) / ke0 << std::endl;
                std::cout << Ostr.str();
              }
          }
      }

    if (exact_solution)
      {
        exact_solution->SetTime(t);
        mfem::FunctionCoefficient cylindrical_weight(
          [](const mfem::Vector &x) {
            return AxisymmetricGeometry::MeasureMultiplier(
              AxisymmetricGeometry::enabled,
              AxisymmetricGeometry::enabled ?
                AxisymmetricGeometry::Radius(x.GetData()) : 0.0);
          });
        const mfem::real_t l1_error =
          sol->ComputeLpError(1.0, *exact_solution, &cylindrical_weight);
        const mfem::real_t l2_error =
          sol->ComputeLpError(2.0, *exact_solution, &cylindrical_weight);
        if (mfem::Mpi::Root())
          {
            std::cout << "Exact L1 Error: " << l1_error << std::endl
                      << "Exact L2 Error: " << l2_error << std::endl;
          }
      }

    // VectorFunctionCoefficient u_final(num_equations,
    //    Prandtl::ConditionFactory::Instance().\
    // GetVectorTDFunctionBoundaryCondition1("AcousticWaveExactSolution")(1.4));
    // u_final.SetTime(1.0);
    // mfem::ParGridFunction u_final_gf(vfes.get());
    // u_final_gf.ProjectCoefficient(u_final);
    // mfem::real_t error_L1 = sol->ComputeLpError(1.0, *u0);
    // mfem::real_t error_L2 = sol->ComputeLpError(2.0, *u0);
    // mfem::real_t error_Linf = sol->ComputeLpError(infinity(), *u0);
    // if (mfem::Mpi::Root())
    // {
    //     std::cout << "L1 Error: " << error_L1 << std::endl;
    //     std::cout << "L2 Error: " << error_L2 << std::endl;
    //     std::cout << "Linf Error: " << error_Linf << std::endl;
    // }

    // Stop the clock if enabled
    if (clock_simulation)
      {
        mfem::tic_toc.Stop();
        if (mfem::Mpi::Root())
          {
            std::cout << "================================================" << std::endl
                      << "Theseus Simulation Completed in " << mfem::tic_toc.RealTime()
                      << " [s]" << std::endl
                      << "================================================" << std::endl;
          }
      }
    else
      {
        if (mfem::Mpi::Root())
          {
            std::cout << "================================================" << std::endl
                      << "Theseus Simulation Completed" << std::endl
                      << "================================================" << std::endl;
          }
      }
  }

  void Simulation::UpdateVisualizationFields()
  {
    const auto &gasModel = rhsOp->GetGasModelInterface();
    const mfem::real_t *sol_state = sol->HostRead();
    for (int i = 0; i < num_dofs_scalar; i++)
      {
        Theseus::DofStateView dofState{sol_state, i};
        if (visualization_config.Has(VisualizationField::velocity))
          {
            for (int component = 0; component < dim; component++)
              {
                (*velocity)(i + component*num_dofs_scalar) =
                  gasModel.velocity(dofState, component);
              }
          }
        if (visualization_config.Has(VisualizationField::pressure))
          {
            (*p)(i) = gasModel.pressure(dofState);
          }
      }
  }

  void Simulation::SaveVisualization()
  {
    if (paraview)
      {
        pd->SetCycle(ti);
        pd->SetTime(t);
        if (visualization_config.MeshMode() == VisualizationMeshMode::gll_subcells)
          {
            ScopedGeometryRefinerType refiner_type(mfem::Quadrature1D::GaussLobatto);
            pd->Save();
          }
        else
          {
            pd->Save();
          }
      }
    else if (visit)
      {
        vd->SetCycle(ti);
        vd->SetTime(t);
        vd->Save();
      }
  }

  CheckpointCompatibility Simulation::CurrentCheckpointCompatibility() const
  {
#ifdef AXISYMMETRIC
    constexpr bool axisymmetric = true;
#else
    constexpr bool axisymmetric = false;
#endif
    return {numProcs,
            order,
            dim,
            num_equations,
            static_cast<int>(sizeof(mfem::real_t)),
            static_cast<long long>(pmesh->GetGlobalNE()),
            static_cast<long long>(vfes->GlobalVSize()),
            axisymmetric};
  }

  void Simulation::LoadCheckpoint()
  {
    const int requested_cycle = checkpoint_config.Cycle();
    const auto metadata_file = checkpoint_config.MetadataFile(requested_cycle);
    std::ifstream metadata_stream(metadata_file);
    MFEM_VERIFY(metadata_stream, "Failed to open checkpoint metadata: " << metadata_file);

    nlohmann::json metadata;
    metadata_stream >> metadata;
    const int stored_cycle = metadata.value("cycle", 0);
    MFEM_VERIFY(stored_cycle == requested_cycle,
                "Checkpoint cycle mismatch: requested " << requested_cycle
                << " but metadata contains " << stored_cycle);
    MFEM_VERIFY(metadata.contains("time"),
                "Checkpoint metadata does not contain a simulation time");

    bool current_format = false;
    try
      {
        current_format = CheckpointConfig::ValidateMetadata(
          metadata, CurrentCheckpointCompatibility());
        if (!current_format && mfem::Mpi::Root())
          {
            std::cerr << "Warning: loading a legacy checkpoint without compatibility metadata; "
                      << "the MPI layout and discretization cannot be verified." << std::endl;
          }
      }
    catch (const std::exception &error)
      {
        MFEM_ABORT(error.what());
      }

    t = metadata.at("time").get<mfem::real_t>();
    ti = stored_cycle;

    const auto rank_file = checkpoint_config.RankFile(ti, myRank);
    std::ifstream checkpoint_stream(rank_file, std::ios::binary);
    MFEM_VERIFY(checkpoint_stream,
                "Failed to open checkpoint state for reading: " << rank_file);

    if (current_format)
      {
        std::string magic;
        std::getline(checkpoint_stream, magic);
        MFEM_VERIFY(magic == "THESEUS_CHECKPOINT_RAW_V1",
                    "Invalid checkpoint state header in " << rank_file);
        std::uint64_t stored_size = 0;
        checkpoint_stream.read(reinterpret_cast<char *>(&stored_size), sizeof(stored_size));
        MFEM_VERIFY(stored_size == static_cast<std::uint64_t>(vfes->GetVSize()),
                    "Checkpoint state size " << stored_size
                    << " does not match current finite-element space size " << vfes->GetVSize());
        sol = std::make_shared<mfem::ParGridFunction>(vfes.get());
        checkpoint_stream.read(
          reinterpret_cast<char *>(sol->HostWrite()),
          static_cast<std::streamsize>(stored_size * sizeof(mfem::real_t)));
        MFEM_VERIFY(checkpoint_stream,
                    "Failed while reading checkpoint state: " << rank_file);
      }
    else
      {
        mfem::ParGridFunction loaded(pmesh.get(), checkpoint_stream);
        MFEM_VERIFY(loaded.Size() == vfes->GetVSize(),
                    "Legacy checkpoint state size " << loaded.Size()
                    << " does not match current finite-element space size " << vfes->GetVSize());
        sol = std::make_shared<mfem::ParGridFunction>(vfes.get());
        const auto *loaded_space = loaded.ParFESpace();
        for (int i = 0; i < loaded.Size(); i++)
          {
            // MFEM serializes parallel grid functions using their space's DOF
            // signs. Convert from the reconstructed legacy space's convention
            // to the current solver space's convention while copying.
            (*sol)[i] = loaded[i] * loaded_space->GetDofSign(i) * vfes->GetDofSign(i);
          }
      }

    MPI_Barrier(pmesh->GetComm());
    if (mfem::Mpi::Root())
      {
        std::cout << "Checkpoint loaded from " << checkpoint_config.CycleDirectory(ti)
                  << " at cycle " << ti << ", time " << t << std::endl;
      }
  }

  void Simulation::SaveCheckpoint()
  {
    // GetGlobalNE is collective, so every rank must build this before the
    // root-only metadata write below.
    const auto compatibility = CurrentCheckpointCompatibility();
    const auto cycle_directory = checkpoint_config.CycleDirectory(ti);
    std::error_code ec;
    std::filesystem::create_directories(cycle_directory, ec);
    MFEM_VERIFY(!ec, "Failed to create checkpoint directory " << cycle_directory
                << ": " << ec.message());

    const auto rank_file = checkpoint_config.RankFile(ti, myRank);
    std::ofstream checkpoint_stream(rank_file, std::ios::binary);
    MFEM_VERIFY(checkpoint_stream,
                "Failed to open checkpoint state for writing: " << rank_file);
    const std::uint64_t state_size = static_cast<std::uint64_t>(sol->Size());
    checkpoint_stream << "THESEUS_CHECKPOINT_RAW_V1\n";
    checkpoint_stream.write(reinterpret_cast<const char *>(&state_size), sizeof(state_size));
    checkpoint_stream.write(
      reinterpret_cast<const char *>(sol->HostRead()),
      static_cast<std::streamsize>(state_size * sizeof(mfem::real_t)));
    checkpoint_stream.close();
    MFEM_VERIFY(checkpoint_stream,
                "Failed while writing checkpoint state: " << rank_file);

    // Publish the metadata only after every rank's state file is complete.
    MPI_Barrier(pmesh->GetComm());
    if (mfem::Mpi::Root())
      {
        const auto metadata_file = checkpoint_config.MetadataFile(ti);
        std::ofstream metadata_stream(metadata_file);
        MFEM_VERIFY(metadata_stream,
                    "Failed to open checkpoint metadata for writing: " << metadata_file);
        metadata_stream << std::setw(2)
                        << CheckpointConfig::Metadata(t, ti, compatibility)
                        << std::endl;
        metadata_stream.close();
        MFEM_VERIFY(metadata_stream,
                    "Failed while writing checkpoint metadata: " << metadata_file);
        std::cout << "Checkpoint saved to " << cycle_directory
                  << " at cycle " << ti << ", time " << t << std::endl;
      }
    MPI_Barrier(pmesh->GetComm());
  }


}
