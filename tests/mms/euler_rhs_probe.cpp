// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "EulerOperator.hpp"
#include "SourceConfiguration.hpp"
#include "verification/mms/TrigonometricEulerMMS.hpp"
#include "LaxFriedrichsFlux.hpp"
#include "RoeFlux.hpp"
#include "HLLFlux.hpp"
#include "ChandrashekarFlux.hpp"
#include "RanochaFlux.hpp"
#include "PirozzoliFlux.hpp"
#include <fstream>
#include <iomanip>

namespace
{
  template<typename Flux>
  struct Physics
  {
    using GasModel = Theseus::IdealGasModel;
    using InviscidFlux = Flux;
  };

  template<typename Flux>
  void Probe(const nlohmann::json &runtime)
  {
    // Pure DG diagnostic: this target deliberately does not compile FV blending.
    mfem::Mesh mesh(runtime.at("mesh_file").get<std::string>().c_str(), 1, 1);
    for (int level = 0; level < runtime.at("ser_ref_levels").get<int>(); ++level)
      {
        mesh.UniformRefinement();
      }
    MFEM_VERIFY(mesh.Dimension() == 2 && mesh.GetNBE() == 0,
                "RHS probe requires a periodic 2D mesh");
    auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection fec(runtime.at("order").get<int>(), 2, mfem::BasisType::GaussLobatto);
    auto vfes = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), &fec, 4,
                                                            mfem::Ordering::byNODES);
    vfes->ExchangeFaceNbrData();
    auto gas = std::make_shared<Theseus::IdealGasModel>(
      Theseus::PhysicsConstants(1.4, .72, 1., 0.), Theseus::StateLayout(2, vfes->GetNDofs()));
    Theseus::EulerOperator<Physics<Flux>> op(vfes, nullptr, pmesh, nullptr, nullptr,
                                            nullptr, gas, "CPG1",
                                            runtime.at("numerical_flux"), 0.);
    op.SetSources(Theseus::Sources::CreateConfigured(runtime, 4, 2, false));
    op.Finalize();
    op.SetStepSize(.0005);
    mfem::ParGridFunction state(vfes.get()), derivative(vfes.get());
    mfem::Vector u, rhs(vfes->GetTrueVSize());
    const auto &rule = *op.GetOperatorCacheReference().ir_vol;
    const bool constant = runtime.value("rhs_constant", false);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int sample = 0;
    for (double time : runtime.at("rhs_times"))
      {
        mfem::VectorFunctionCoefficient exact(4, [=](const mfem::Vector &x, mfem::Vector &value)
        {
          mfem::real_t source[4];
          value.SetSize(4);
          Theseus::Sources::TrigonometricEulerMMS(2, x.GetData(), time, value.GetData(), source);
          if (constant)
            {
              value(0) = 2.;
              value(1) = 1.4;
              value(2) = .6;
              value(3) = 3.08;
            }
        });
        state.ProjectCoefficient(exact);
        state.GetTrueDofs(u);
        op.SetTime(time);
        op.Mult(u, rhs);
        derivative.SetFromTrueDofs(rhs);
        std::ofstream output(runtime.at("rhs_output").get<std::string>() + "-t" +
                             std::to_string(sample++) + "-rank" + std::to_string(rank) + ".csv");
        MFEM_VERIFY(output.good(), "Cannot open RHS output");
        output << std::setprecision(17);
        output << "cx,cy,cz,x,y,z,u0,u1,u2,u3,r0,r1,r2,r3\n";
        mfem::Vector center(2), x(2), value(4), rate(4);
        for (int element = 0; element < pmesh->GetNE(); ++element)
          {
            auto *transformation = pmesh->GetElementTransformation(element);
            transformation->Transform(mfem::Geometries.GetCenter(mfem::Geometry::SQUARE), center);
            for (int point = 0; point < rule.GetNPoints(); ++point)
              {
                const auto &ip = rule.IntPoint(point);
                transformation->Transform(ip, x);
                state.GetVectorValue(element, ip, value);
                derivative.GetVectorValue(element, ip, rate);
                output << center(0) << ',' << center(1) << ",0," << x(0) << ',' << x(1) << ",0";
                for (int component = 0; component < 4; ++component)
                  {
                    output << ',' << value(component);
                  }
                for (int component = 0; component < 4; ++component)
                  {
                    output << ',' << rate(component);
                  }
                output << '\n';
              }
          }
      }
  }
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const char *config = "config.json";
  const char *backend = "cpu";
  mfem::OptionsParser options(argc, argv);
  options.AddOption(&config, "-c", "--config", "Probe configuration.");
  options.AddOption(&backend, "-d", "--device", "MFEM backend.");
  options.Parse();
  if (!options.Good())
    {
      options.PrintUsage(std::cerr);
      return 2;
    }
  mfem::Device device(backend);
  device.Print();
  std::ifstream input(config);
  const auto runtime = nlohmann::json::parse(input).at("runTime");
  const auto flux = runtime.at("numerical_flux").get<std::string>();
  if (flux == "lfr") { Probe<Theseus::LaxFriedrichsFlux::InviscidFlux>(runtime); }
  else if (flux == "Roe") { Probe<Theseus::RoeFlux::InviscidFlux>(runtime); }
  else if (flux == "HLL") { Probe<Theseus::HLLFlux::InviscidFlux>(runtime); }
  else if (flux == "Chandrashekar") { Probe<Theseus::ChandrashekarFlux::InviscidFlux>(runtime); }
  else if (flux == "Ranocha") { Probe<Theseus::RanochaFlux::InviscidFlux>(runtime); }
  else if (flux == "Pirozzoli") { Probe<Theseus::PirozzoliFlux::InviscidFlux>(runtime); }
  else { MFEM_ABORT("Unsupported RHS probe flux"); }
  return 0;
}
