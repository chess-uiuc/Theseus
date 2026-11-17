// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "NSOperator.hpp"
#include "RoeFlux.hpp"
#include "verification/mms/TrigonometricEulerMMS.hpp"
#include <array>
#include <cmath>
#include <iostream>

namespace
{
struct Physics
{
  using GasModel = Theseus::IdealGasModel;
  using InviscidFlux = Theseus::RoeFlux::InviscidFlux;
};

std::array<double, 4>
GradientError (int degree, int elements, bool constant = false, double *rhs_error = nullptr,
               double mean_velocity_scale = 1.)
{
  auto mesh = mfem::Mesh::MakeCartesian2D (elements, elements, mfem::Element::QUADRILATERAL, true,
                                           2., 2., false);
  std::vector<mfem::Vector> shifts (2, mfem::Vector (2));
  shifts[0] = 0.;
  shifts[1] = 0.;
  shifts[0](0) = 2.;
  shifts[1](1) = 2.;
  auto periodic = mfem::Mesh::MakePeriodic (mesh, mesh.CreatePeriodicVertexMapping (shifts));
  periodic.RemoveInternalBoundaries ();
  auto pmesh = std::make_shared<mfem::ParMesh> (MPI_COMM_WORLD, periodic);
  mfem::DG_FECollection fec (degree, 2, mfem::BasisType::GaussLobatto);
  auto vfes = std::make_shared<mfem::ParFiniteElementSpace> (pmesh.get (), &fec, 4,
                                                             mfem::Ordering::byNODES);
  vfes->ExchangeFaceNbrData ();
  std::vector<std::shared_ptr<mfem::ParGridFunction>> storage;
  for (int direction = 0; direction < 2; ++direction)
    {
      storage.push_back (std::make_shared<mfem::ParGridFunction> (vfes.get ()));
    }
  auto gas = std::make_shared<Theseus::IdealGasModel> (
      Theseus::PhysicsConstants (1.4, .72, 1., .02), Theseus::StateLayout (2, vfes->GetNDofs ()));
  Theseus::NSOperator<Physics> op (vfes, nullptr, pmesh, nullptr, nullptr, storage, nullptr, gas,
                                   "CPG1", "Roe", 0.);
  op.Finalize ();
  constexpr double time = .173;
  mfem::VectorFunctionCoefficient exact (4,
                                         [constant, mean_velocity_scale] (const mfem::Vector &x, mfem::Vector &value)
                                           {
                                             mfem::real_t source[4];
                                             Theseus::Sources::TrigonometricEulerMMS (
                                                 2, x.GetData (), time, value.GetData (), source, 1., mean_velocity_scale);
                                             if (constant)
                                               {
                                                 value (0) = 2.;
                                                 value (1) = 1.4;
                                                 value (2) = .6;
                                                 value (3) = 3.08;
                                               }
                                           });
  mfem::ParGridFunction state (vfes.get ());
  state.ProjectCoefficient (exact);
  mfem::Vector u, entropy;
  state.GetTrueDofs (u);
  op.ComputeEntropyState (u, entropy);
  mfem::Vector gx (vfes->GetTrueVSize ()), gy (vfes->GetTrueVSize ());
  std::vector<mfem::Vector *> gradients{ &gx, &gy };
  op.GradOperator (entropy, gradients);
  op.ComputeGradPrimFromGradEntropy (u, gradients);
  mfem::ParGridFunction gradient_x (vfes.get ()), gradient_y (vfes.get ());
  gradient_x.SetFromTrueDofs (gx);
  gradient_y.SetFromTrueDofs (gy);
  auto &cache = op.GetOperatorCacheReference ();
  const auto &rule = *cache.ir_vol;
  const auto *weights = cache.elQuadratureWeights.HostRead ();
  std::array<double, 4> squared{};
  const int fields[] = { 0, 1, 2, 4 };
  mfem::Vector position (2), values (4);
  for (int element = 0; element < pmesh->GetNE (); ++element)
    {
      auto *transformation = pmesh->GetElementTransformation (element);
      for (int point = 0; point < rule.GetNPoints (); ++point)
        {
          const auto &ip = rule.IntPoint (point);
          transformation->Transform (ip, position);
          for (int direction = 0; direction < 2; ++direction)
            {
              auto &gradient = direction == 0 ? gradient_x : gradient_y;
              gradient.GetVectorValue (element, ip, values);
              for (int q = 0; q < 4; ++q)
                {
                  const auto reference = Theseus::Sources::TrigonometricPrimitive (
                      fields[q], 2, position.GetData (), time);
                  const auto error = values (q) - (constant ? 0. : reference.gradient[direction]);
                  squared[q] += weights[element * rule.GetNPoints () + point] * error * error;
                }
            }
        }
    }
  for (auto &value : squared)
    {
      value = std::sqrt (value);
    }
  if (constant && rhs_error)
    {
      mfem::Vector rhs (vfes->GetTrueVSize ());
      op.Mult (u, rhs);
      *rhs_error = rhs.Normlinf ();
    }
  return squared;
}
}

int
main (int argc, char **argv)
{
  mfem::Mpi::Init (argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device (backend.c_str ());
  if (backend == "cuda" && !mfem::Device::Allows (mfem::Backend::CUDA))
    {
      return 1;
    }
  int failures = 0;
  for (int degree : { 2, 3 })
    {
      const auto coarse = GradientError (degree, 6);
      const auto fine = GradientError (degree, 12);
      std::cout << "P" << degree << " primitive-gradient orders:";
      for (int q = 0; q < 4; ++q)
        {
          const auto rate = std::log2 (coarse[q] / fine[q]);
          std::cout << ' ' << rate;
          if (!std::isfinite (rate) || rate < degree - .5)
            {
              ++failures;
            }
        }
      std::cout << '\n';
    }
  // The same MMS gradients at different mean speeds also exercise the nonlinear
  // entropy-to-primitive conversion. A third mesh guards against one lucky ratio.
  for (double velocity_scale : { .25, 5. })
    {
      for (int degree : { 2, 3 })
        {
          const auto coarse = GradientError (degree, 6, false, nullptr, velocity_scale);
          const auto middle = GradientError (degree, 12, false, nullptr, velocity_scale);
          const auto fine = GradientError (degree, 18, false, nullptr, velocity_scale);
          std::cout << "Mean velocity scale " << velocity_scale << " P" << degree << " gradient orders:";
          for (int q = 0; q < 4; ++q)
            {
              const auto first = std::log (coarse[q] / middle[q]) / std::log (2.);
              const auto second = std::log (middle[q] / fine[q]) / std::log (1.5);
              std::cout << ' ' << first << '/' << second;
              if (!std::isfinite (first) || !std::isfinite (second)
                  || first < degree - .5 || second < degree - .5)
                {
                  ++failures;
                }
            }
          std::cout << '\n';
        }
    }
  double rhs_error = 0.;
  const auto uniform = GradientError (2, 6, true, &rhs_error);
  for (const auto error : uniform)
    {
      if (!std::isfinite (error) || error > 1e-11)
        {
          ++failures;
        }
    }
  std::cout << "Constant-state CNS RHS maximum: " << rhs_error << '\n';
  if (!std::isfinite (rhs_error) || rhs_error > 1e-11)
    {
      ++failures;
    }
  return failures ? 1 : 0;
}
