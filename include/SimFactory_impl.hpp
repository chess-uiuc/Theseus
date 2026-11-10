// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>

#include "parse_helpers.hpp"
#include "SimFactory.hpp"
#include "EulerOperator.hpp"
#include "NSOperator.hpp"
#include "LaxFriedrichsFlux.hpp"
#include "ChandrashekarFlux.hpp"
#include "RanochaFlux.hpp"
#include "PirozzoliFlux.hpp"
#include "HLLFlux.hpp"
#include "RoeFlux.hpp"
#include "LTETable.hpp"
#include "TheseusConfig.hpp"
#include "LTEECFlux.hpp"
#include "LTEHelmholtzTableBuilder.hpp"

namespace Theseus {

  // TODO: Update for LTE Table Data, and LTE Gas Model setup
  template <typename Physics, typename GasModelT>
  std::unique_ptr<Theseus::RHSOperatorBase>
  MakeTypedRHSOperator(
                       bool inviscid,
                       const nlohmann::json &runtime,
                       std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
                       std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
                       std::shared_ptr<mfem::ParMesh> pmesh,
                       std::shared_ptr<mfem::ParGridFunction> eta,
                       std::shared_ptr<mfem::ParGridFunction> alpha,
                       std::vector<std::shared_ptr<mfem::ParGridFunction> > &grad_u,
                       std::shared_ptr<Prandtl::PerssonPeraireIndicator> indicator,
                       mfem::real_t alpha_max,
                       std::shared_ptr<const GasModelT> gas_,
                       const std::string &gasModelName,
                       const std::string &numFluxName)
  {

    if (inviscid) {
      return std::make_unique<Theseus::EulerOperator<Physics>>(vfes, fes0, pmesh, eta, alpha,
                                                               indicator,
                                                               gas_,
                                                               gasModelName,
                                                               numFluxName,
                                                               alpha_max);
    } else {
      return std::make_unique<Theseus::NSOperator<Physics>>(vfes, fes0, pmesh, eta, alpha, grad_u,
                                                            indicator,
                                                            gas_,
                                                            gasModelName,
                                                            numFluxName,
                                                            alpha_max);
    }
  }

  std::unique_ptr<Theseus::RHSOperatorBase>
  MakeRHSOperator(const nlohmann::json& runtime,
                  std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
                  std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
                  std::shared_ptr<mfem::ParMesh> pmesh,
                  std::shared_ptr<mfem::ParGridFunction> eta,
                  std::shared_ptr<mfem::ParGridFunction> alpha,
                  std::vector<std::shared_ptr<mfem::ParGridFunction> > &grad_u,
                  std::shared_ptr<Prandtl::PerssonPeraireIndicator> indicator,
                  mfem::real_t alpha_max)
  {

    std::string gas_model_string =
      to_lower(runtime.value("gas_model", std::string{}));

    std::string inv_flux_string =
      to_lower(runtime.value("numerical_flux", std::string{}));

    std::string flow_model_string =
      to_lower(runtime.value("flow_model", std::string{}));

    const bool use_cpg =
      gas_model_string.empty() ||
      gas_model_string == "cpg" ||
      gas_model_string == "ideal" ||
      gas_model_string == "ideal_gas";

    const bool use_lteec = inv_flux_string == "lteec" || inv_flux_string == "lteecflux";
    const bool use_helmholtz = gas_model_string == "lte_helmholtz" ||
                               (gas_model_string == "lte" && use_lteec);

    const bool use_lte =
      gas_model_string == "lte";

    const bool use_chan =
      inv_flux_string.empty() ||
      inv_flux_string == "chandrashekar" ||
      starts_with(inv_flux_string, "chan");

    const bool use_ranocha =
      inv_flux_string == "ranocha";

    const bool use_piflux =
      inv_flux_string == "pirozzoli" ||
      inv_flux_string == "piflux" ||
      starts_with(inv_flux_string, "pi");

    const bool use_hll =
      inv_flux_string == "hll";

    const bool use_roe =
      inv_flux_string == "roe";

    const bool use_llf =
      inv_flux_string == "llf" ||
      inv_flux_string == "lfr" ||
      inv_flux_string == "laxfriedrichs" ||
      inv_flux_string == "lax_friedrichs" ||
      starts_with(inv_flux_string, "lax");

    const bool viscous =
      starts_with(flow_model_string, "visc") ||
      starts_with(flow_model_string, "nav")  ||
      starts_with(flow_model_string, "cns")  ||
      starts_with(flow_model_string, "ns");
    const bool inviscid = !viscous;

    const int dim = pmesh->Dimension();
    const int num_dofs_scalar = vfes->GetNDofs();

    Theseus::PhysicsConstants physics_constants(runtime.value("gamma", 1.4),
                                                runtime.value("Pr", 0.72),
                                                runtime.value("R_gas", 287.05),
                                                runtime.value("mu", 0.02),
                                                runtime.value("mu0", 1.716e-5));
    Theseus::StateLayout layout(dim, num_dofs_scalar);

    if (use_helmholtz)
      {
#if defined(USE_PLATO) && defined(THESEUS_PLATO_HAS_ENTROPY)
        if (!use_lteec && !use_piflux && !use_hll && !use_llf)
          {
            std::cerr << "Helmholtz LTE supports LTEECFlux, Pirozzoli, HLL and LLF" << std::endl;
            return nullptr;
          }
        const int nx = runtime.value("N_rho", 101);
        const int ny = runtime.value("N_T", 101);
        const mfem::real_t rho_min = runtime.value("rho_min", 0.01);
        const mfem::real_t rho_max = runtime.value("rho_max", 3.0);
        const mfem::real_t t_min = runtime.value("T_min", 100.0);
        const mfem::real_t t_max = runtime.value("T_max", 2000.0);
        if (nx < 2 || ny < 2 || !std::isfinite(rho_min) || !std::isfinite(rho_max) ||
            !std::isfinite(t_min) || !std::isfinite(t_max) ||
            rho_min <= 0 || rho_max <= rho_min || t_min <= 0 || t_max <= t_min)
          {
            std::cerr << "Invalid Helmholtz table grid dimensions or bounds" << std::endl;
            return nullptr;
          }
        const std::string mixture = runtime.value("gas_mixture", "air5");
        const std::string solver = runtime.value("plato_solver", "LTE_table_rhoT_(air5)");
        const std::string path = runtime.value("database_path", std::string(BuildConfig::PlatoDBPath));
        auto data = std::make_unique<HelmholtzTable::Data>();
        if (runtime.value("rho_dist", "log") == "log")
          {
            LTETable::log_grid(nx, rho_min, rho_max, data->rho_grid);
          }
        else
          {
            LTETable::uniform_grid(nx, rho_min, rho_max, data->rho_grid);
          }
        if (runtime.value("T_dist", "linear") == "log")
          {
            LTETable::log_grid(ny, t_min, t_max, data->T_grid);
          }
        else
          {
            LTETable::uniform_grid(ny, t_min, t_max, data->T_grid);
          }
        data->lte_table.SetSize(HelmholtzTable::Layout::num_properties*nx*ny);
        data->inv_table.SetSize(nx*ny);
        data->e_grid.SetSize(ny);
        int rank;
        MPI_Comm_rank(pmesh->GetComm(), &rank);
        int constructed = 1;
        if (rank == 0)
          {
            bool initialized = false;
            try
              {
                if (LTETable::check_plato_database_path(path))
                  {
                    throw std::invalid_argument("PLATO database is unavailable");
                  }
                std::cout << "Constructing Helmholtz LTE table for " << mixture << std::endl;
                plato_initialize(solver.c_str(), mixture.c_str(), "empty", "empty", path.c_str());
                initialized = true;
                HelmholtzTable::Build(*data);
              }
            catch (const std::exception &error)
              {
                constructed = 0;
                std::cerr << "Helmholtz table construction failed: " << error.what() << std::endl;
              }
            if (initialized)
              {
                plato_finalize();
              }
          }
        MPI_Bcast(&constructed, 1, MPI_INT, 0, pmesh->GetComm());
        if (!constructed)
          {
            return nullptr;
          }
        MPI_Bcast(data->lte_table.HostWrite(), data->lte_table.Size(), MPI_DOUBLE, 0, pmesh->GetComm());
        MPI_Bcast(data->inv_table.HostWrite(), nx*ny, MPI_DOUBLE, 0, pmesh->GetComm());
        MPI_Bcast(data->e_grid.HostWrite(), ny, MPI_DOUBLE, 0, pmesh->GetComm());
        const auto tables = HelmholtzTable::HostView(*data, HelmholtzTable::Layout{nx, ny});
        const auto gas = std::make_shared<LTEHelmholtzGas>(physics_constants, layout, tables);
        auto make = [&](auto flux, const char *name)
          {
            using Flux = decltype(flux);
            using Physics = PhysicsTraits<LTEHelmholtzGas, Flux>;
            auto op = MakeTypedRHSOperator<Physics, LTEHelmholtzGas>(inviscid, runtime,
              vfes, fes0, pmesh, eta, alpha, grad_u, indicator, alpha_max, gas,
              "LTE-Helmholtz:" + mixture, name);
            auto *typed = dynamic_cast<RHSOperator<Physics> *>(op.get());
            typed->GetOperatorCacheReference().lteTableData = std::move(data);
            return op;
          };
        if (use_lteec)
          {
            return make(LTEECFlux::InviscidFlux{}, "LTEECFlux");
          }
        if (use_piflux)
          {
            return make(PirozzoliFlux::InviscidFlux{}, "Pirozzoli");
          }
        if (use_hll)
          {
            return make(HLLFlux::InviscidFlux{}, "HLL");
          }
        return make(LaxFriedrichsFlux::InviscidFlux{}, "LLF");
#else
        std::cerr << "Helmholtz LTE requires PLATO with plato_get_species_entropy in its header and library; "
                  << "configure with THESEUS_WITH_PLATO=ON and a compatible TPL prefix. "
                  << "Legacy gas_model=lte with existing fluxes remains available." << std::endl;
        return nullptr;
#endif
      }

    if (use_cpg)
      {

        auto gas_model =
          std::make_shared<Theseus::IdealGasModel>(physics_constants, layout);
        std::string gasModelName("CPG1");
        if (use_chan)
          {
            std::string numFluxName("Chandrashekar");
            using Physics =
              Theseus::PhysicsTraits<Theseus::IdealGasModel,
                                     Theseus::ChandrashekarFlux::InviscidFlux>;
            return MakeTypedRHSOperator<Physics, Theseus::IdealGasModel>(inviscid, runtime,
                                                                         vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                         indicator, alpha_max, gas_model,
                                                                         gasModelName, numFluxName);
          }
        if (use_ranocha)
          {
            std::string numFluxName("Ranocha");
            using Physics =
              Theseus::PhysicsTraits<Theseus::IdealGasModel,
                                     Theseus::RanochaFlux::InviscidFlux>;
            return MakeTypedRHSOperator<Physics, Theseus::IdealGasModel>(inviscid, runtime,
                                                                         vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                         indicator, alpha_max, gas_model,
                                                                         gasModelName, numFluxName);
          }
        if (use_piflux)
          {
            std::string numFluxName("Pirozzoli");
            using Physics =
              Theseus::PhysicsTraits<Theseus::IdealGasModel,
                                     Theseus::PirozzoliFlux::InviscidFlux>;
            return MakeTypedRHSOperator<Physics, Theseus::IdealGasModel>(inviscid, runtime,
                                                                         vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                         indicator, alpha_max, gas_model,
                                                                         gasModelName, numFluxName);
          }
        else if (use_hll)
          {
            std::string numFluxName("HLL");
            using Physics =
              Theseus::PhysicsTraits<Theseus::IdealGasModel,
                                     Theseus::HLLFlux::InviscidFlux>;

            return MakeTypedRHSOperator<Physics, Theseus::IdealGasModel>(inviscid, runtime,
                                                                         vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                         indicator, alpha_max, gas_model,
                                                                         gasModelName, numFluxName);
          }
        else if (use_llf)
          {
            std::string numFluxName("LLF");
            using Physics =
              Theseus::PhysicsTraits<Theseus::IdealGasModel,
                                     Theseus::LaxFriedrichsFlux::InviscidFlux>;

            return MakeTypedRHSOperator<Physics, Theseus::IdealGasModel>(inviscid, runtime,
                                                                         vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                         indicator, alpha_max, gas_model,
                                                                         gasModelName, numFluxName);
          }
        else if (use_roe)
          {
            std::string numFluxName("Roe");
            using Physics =
              Theseus::PhysicsTraits<Theseus::IdealGasModel,
                                     Theseus::RoeFlux::InviscidFlux>;

            return MakeTypedRHSOperator<Physics, Theseus::IdealGasModel>(inviscid, runtime,
                                                                         vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                         indicator, alpha_max, gas_model,
                                                                         gasModelName, numFluxName);
          }
        else {
          std::cerr << "Error: Invalid Numerical Flux Type specified: "
                    << inv_flux_string << "\n"
                    << "Supported: Chandrashekar, Ranocha, Pirozzoli, LLF/LFR, HLL, Roe"
                    << std::endl;
          return nullptr;
        }
      } else if(use_lte){
      std::string mixture(runtime.value("gas_mixture", "air5"));
      std::string solver(runtime.value("plato_solver", "LTE_table_rhoT_(air5)"));
      std::string path(runtime.value("database_path", std::string(Theseus::BuildConfig::PlatoDBPath)));
      std::string rho_dist(runtime.value("rho_dist", "log"));
      std::string T_dist(runtime.value("T_dist", "log"));
      int N_rho    = runtime.value("N_rho", 101);
      int N_T      = runtime.value("N_T", 101);
      mfem::real_t rho_min  = runtime.value("rho_min", 0.1);
      mfem::real_t rho_max  = runtime.value("rho_max", 1.1);
      mfem::real_t T_min    = runtime.value("T_min", 250.0);
      mfem::real_t T_max    = runtime.value("T_max", 35.0);
      int num_properties = 9; // CL NOTE : Check LTE EOS
      auto lteData = std::make_unique<Theseus::LTETable::Data>();
      auto &lteTableData = *lteData;
      Theseus::LTETable::LTETables    lteTables;
      lteTableData.lte_table.SetSize(N_rho * N_T * num_properties);
      lteTableData.inv_table.SetSize(N_rho * N_T);
      if(rho_dist == "log")
        {
          Theseus::LTETable::log_grid(N_rho, rho_min, rho_max, lteTableData.rho_grid);
        }
      else
        {
          Theseus::LTETable::uniform_grid(N_rho, rho_min, rho_max, lteTableData.rho_grid);
        }

      if(T_dist == "log")
        {
          Theseus::LTETable::log_grid(N_T, T_min, T_max, lteTableData.T_grid);
        }
      else
        {
          Theseus::LTETable::uniform_grid(N_T, T_min, T_max, lteTableData.T_grid);
        }
      lteTableData.e_grid.SetSize(N_T);
      lteTables.L.setup(N_rho, N_T);
#ifdef USE_PLATO
      if(mfem::Mpi::Root())
        {
          std::cout << "Constructing LTE table for " << mixture
                    << " with solver " << solver << std::endl
                    << "LTE Database: " << path << std::endl;
          if(Theseus::LTETable::check_plato_database_path(path)){
            std::cerr << "Plato Database (" << path << ") not found." << std::endl;
            return nullptr;
          }
          mfem::real_t e_min, e_max;
          std::string empty_str("empty");
          plato_initialize(solver.c_str(), mixture.c_str(), empty_str.c_str(), empty_str.c_str(), path.c_str());
          Theseus::LTETable::fill_table(lteTables.L, lteTableData.rho_grid.GetData(), lteTableData.T_grid.GetData(),
                                        lteTableData.lte_table.GetData(), e_min, e_max);
          std::cout << "Constructing inverse table T = T(rho, e)" << std::endl;
          Theseus::LTETable::uniform_grid(N_T, e_min, e_max, lteTableData.e_grid);
          Theseus::LTETable::fill_inv_table(lteTables.L, lteTableData.rho_grid.GetData(), lteTableData.e_grid.GetData(),
                                            lteTableData.T_grid.GetData(), lteTableData.inv_table.GetData());
          plato_finalize();
        }
#else
      // TODO: Eventually, maybe run with pre-generated tables.
      if(mfem::Mpi::Root()){
        std::cerr << "LTE runs *must* have Theseus build with PLATO support (-DTHESEUS_WITH_PLATO)" << std::endl;
      }
      return nullptr;
#endif
      MPI_Bcast(lteTableData.lte_table.GetData(), N_rho * N_T * num_properties, MPI_DOUBLE, 0, pmesh->GetComm());
      MPI_Bcast(lteTableData.inv_table.GetData(), N_rho * N_T, MPI_DOUBLE, 0, pmesh->GetComm());
      MPI_Bcast(lteTableData.e_grid.GetData(), N_T, MPI_DOUBLE, 0, pmesh->GetComm());
      lteTables.tables = {
        lteTableData.lte_table.HostRead(), lteTableData.inv_table.HostRead(),
        lteTableData.rho_grid.HostRead(), lteTableData.T_grid.HostRead(),
        lteTableData.e_grid.HostRead()
      };
      auto gas_model = std::make_shared<Theseus::LTEGas>(physics_constants, layout, lteTables);
      std::string gasModelName("LTE:"+mixture);
      if (use_chan)
        {
          std::string numFluxName("Chandrashekar");
          using Physics =
            Theseus::PhysicsTraits<Theseus::LTEGas,
                                   Theseus::ChandrashekarFlux::InviscidFlux>;
          // return MakeTypedRHSOperator<Physics>(inviscid, runtime,
          //                                   vfes, fes0, pmesh, eta, alpha, grad_u,
          //                                     indicator, alpha_max, gas_model,
          //                                     gasModelName, numFluxName);
          std::cerr << "Error: Cannot use Chandrashekar flux with LTE" << std::endl;
          return nullptr;
        }
      else if (use_piflux)
        {
          std::string numFluxName("Pirozzoli");
          using Physics =
            Theseus::PhysicsTraits<Theseus::LTEGas,
                                   Theseus::PirozzoliFlux::InviscidFlux>;

          auto rhsOp = MakeTypedRHSOperator<Physics, Theseus::LTEGas>(inviscid, runtime,
                                                                      vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                      indicator, alpha_max, gas_model,
                                                                      gasModelName, numFluxName);
          RHSOperator<Physics> *lteRHSOp = dynamic_cast<RHSOperator<Physics> *>(rhsOp.get());
          auto &operator_cache = lteRHSOp->GetOperatorCacheReference();
          operator_cache.lteTableData = std::move(lteData);
          return rhsOp;
        }
      else if (use_hll)
        {
          std::string numFluxName("HLL");
          using Physics =
            Theseus::PhysicsTraits<Theseus::LTEGas,
                                   Theseus::HLLFlux::InviscidFlux>;

          auto rhsOp = MakeTypedRHSOperator<Physics, Theseus::LTEGas>(inviscid, runtime,
                                                                      vfes, fes0, pmesh, eta, alpha, grad_u,
                                                                      indicator, alpha_max, gas_model,
                                                                      gasModelName, numFluxName);
          RHSOperator<Physics> *lteRHSOp = dynamic_cast<RHSOperator<Physics> *>(rhsOp.get());
          auto &operator_cache = lteRHSOp->GetOperatorCacheReference();
          operator_cache.lteTableData = std::move(lteData);
          return rhsOp;
        }
      else if (use_llf)
        {
          std::string numFluxName("LFR");
          using Physics =
            Theseus::PhysicsTraits<Theseus::LTEGas,
                                   Theseus::LaxFriedrichsFlux::InviscidFlux>;

          auto rhsOp = MakeTypedRHSOperator<Physics, Theseus::LTEGas>
            (inviscid, runtime,vfes, fes0, pmesh, eta, alpha, grad_u,
             indicator, alpha_max, gas_model,
             gasModelName, numFluxName);

          RHSOperator<Physics> *lteRHSOp = dynamic_cast<RHSOperator<Physics> *>(rhsOp.get());
          auto &operator_cache = lteRHSOp->GetOperatorCacheReference();
          operator_cache.lteTableData = std::move(lteData);
          return rhsOp;
        }
      else {
        std::cerr << "Error: Invalid Numerical Flux Type specified: "
                  << inv_flux_string << "\n"
                  << "Supported: Pirozzoli, LLF/LFR, HLL"
                  << std::endl;
        return nullptr;
      }

    } else {
      std::cerr << "Error: Invalid Gas Model specified: "
                << gas_model_string << "\n"
                << "Supported: CPG, LTE, LTE_Helmholtz"
                << std::endl;
      return nullptr;
    }

  }
}
