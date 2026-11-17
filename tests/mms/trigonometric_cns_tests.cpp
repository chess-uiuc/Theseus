// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "SourceConfiguration.hpp"
#include "verification/mms/TrigonometricCNSMMS.hpp"
#include "cns_reference.hpp"
#include <cmath>
#include <iostream>

namespace
{
int failures = 0;
int checks = 0;
void
Near (double actual, double expected, double tolerance)
{
  ++checks;
  if (!std::isfinite (actual) || !std::isfinite (expected)
      || std::abs (actual - expected) > tolerance * (1. + std::abs (expected)))
    {
      if (++failures < 10)
        {
          std::cerr << actual << " expected " << expected << '\n';
        }
    }
}

using namespace CNSReference;

void
Check (int dim, double R, double scale, const char *regime = "transonic", double velocity_scale = 1.)
{
  constexpr int points = 12;
  constexpr double mu = .02;
  const double kappa = mu * 3.5 * R / .72;
  mfem::Vector coordinates (points * dim), state (points * (dim + 2)), rhs (state.Size () + 2);
  coordinates.UseDevice ();
  state.UseDevice ();
  rhs.UseDevice ();
  for (int i = 0; i < points * dim; ++i)
    {
      coordinates.HostWrite ()[i] = std::sin (.37 * i + .12);
    }
  Theseus::Sources::MeshView mesh;
  mesh.elements = 1;
  mesh.dofs = points;
  mesh.dimension = dim;
  mesh.equations = dim + 2;
  mesh.layout = Theseus::StateLayout (dim, 1);
  mesh.communicator = MPI_COMM_WORLD;
  mesh.coordinates = coordinates.Read ();
  nlohmann::json runtime
      = { { "gas_model", "cpg" },
          { "flow_model", "navier-stokes" },
          { "gamma", 1.4 },
          { "R_gas", R },
          { "Pr", .72 },
          { "mu", mu },
          { "sources", { { { "type", "trigonometric-cns-mms" }, { "time_scale", scale } } } } };
  runtime["sources"][0]["regime"] = regime;
  auto system = Theseus::Sources::CreateConfigured (runtime, dim + 2, dim, false);
  system->Initialize (mesh);
  state = 7.;
  for (double time : { 0., .173 })
    {
      rhs = .5;
      system->AddStage ({ time, .001, state.Read () }, rhs.ReadWrite () + 1);
      Near (rhs.HostRead ()[0], .5, 0.);
      Near (rhs.HostRead ()[rhs.Size () - 1], .5, 0.);
      for (int point = 0; point < points; ++point)
        {
          double x[3] = {};
          for (int d = 0; d < dim; ++d)
            {
              x[d] = coordinates.HostRead ()[point * dim + d];
            }
          double reference[5], analytic_state[5], analytic_source[5], zero_state[5], zero_source[5];
          Residual (dim, x, time, mu, kappa, R, scale, reference, false, velocity_scale);
          Theseus::Sources::TrigonometricCNSMMS (dim, x, time, mu, kappa, R, analytic_state,
                                                 analytic_source, scale, velocity_scale);
          Theseus::Sources::TrigonometricCNSMMS (dim, x, time, 0., 0., R, zero_state, zero_source,
                                                 scale, velocity_scale);
          double euler_state[5], euler_source[5];
          Theseus::Sources::TrigonometricEulerMMS (dim, x, time, euler_state, euler_source, scale, velocity_scale);
          for (int q = 0; q < dim + 2; ++q)
            {
              Near (analytic_source[q], reference[q], 3.e-7);
              Near (rhs.HostRead ()[1 + q * points + point], .5 + reference[q], 3.e-7);
              Near (zero_source[q], euler_source[q], 0.);
              Near (analytic_state[q], euler_state[q], 0.);
            }
          // All Hessian entries, including mixed derivatives, independently differenced.
          for (int field = 0; field < 5; ++field)
            {
              const auto second
                  = Theseus::Sources::TrigonometricSecondDerivative (field, dim, x, time, scale, velocity_scale);
              for (int i = 0; i < dim; ++i)
                {
                  for (int j = 0; j < dim; ++j)
                    {
                      double value = 0.;
                      for (int sample = 0; sample < 4; ++sample)
                        {
                          double shifted[] = { x[0], x[1], x[2] };
                          shifted[j] += offsets[sample] * step;
                          value += weights[sample]
                                   * Derivative (field, dim, shifted, time * scale, i, velocity_scale)
                                   / (12. * step);
                        }
                      Near (second.hessian[i][j], value, 3.e-7);
                    }
                }
            }
        }
    }
  for (int i = 0; i < state.Size (); ++i)
    {
      Near (state.HostRead ()[i], 7., 0.);
    }
  auto inviscid_forcing = runtime;
  inviscid_forcing["sources"][0]["viscous_scale"] = 0.;
  auto inviscid_system = Theseus::Sources::CreateConfigured (inviscid_forcing, dim + 2, dim, false);
  inviscid_system->Initialize (mesh);
  rhs = .5;
  inviscid_system->AddStage ({ .173, .001, state.Read () }, rhs.ReadWrite () + 1);
  for (int point = 0; point < points; ++point)
    {
      double x[3] = {};
      for (int d = 0; d < dim; ++d)
        {
          x[d] = coordinates.HostRead ()[point * dim + d];
        }
      double reference[5];
      Residual (dim, x, .173, 0., 0., R, scale, reference, false, velocity_scale);
      for (int q = 0; q < dim + 2; ++q)
        {
          Near (rhs.HostRead ()[1 + q * points + point], .5 + reference[q], 3.e-7);
        }
    }
  auto invalid_regime = runtime;
  invalid_regime["sources"][0]["regime"] = "unknown";
  bool rejected_regime = false;
  try
    {
      Theseus::Sources::CreateConfigured (invalid_regime, dim + 2, dim, false);
    }
  catch (const std::invalid_argument &)
    {
      rejected_regime = true;
    }
  Near (rejected_regime ? 1. : 0., 1., 0.);
  for (const char *key : { "mu", "Pr", "R_gas", "time_scale", "viscous_scale" })
    {
      auto invalid = runtime;
      if (std::string (key) == "time_scale" || std::string (key) == "viscous_scale")
        {
          invalid["sources"][0][key] = -1.;
        }
      else
        {
          invalid[key] = -1.;
        }
      bool rejected = false;
      try
        {
          Theseus::Sources::CreateConfigured (invalid, dim + 2, dim, false);
        }
      catch (const std::invalid_argument &)
        {
          rejected = true;
        }
      Near (rejected ? 1. : 0., 1., 0.);
    }
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
#ifdef SUTHERLAND
  nlohmann::json runtime = { { "R_gas", 1. },
                             { "flow_model", "navier-stokes" },
                             { "sources", { { { "type", "trigonometric-cns-mms" } } } } };
  try
    {
      Theseus::Sources::CreateConfigured (runtime, 4, 2, false);
    }
  catch (const std::invalid_argument &error)
    {
      const std::string message = error.what ();
      std::cout << message << '\n';
      return message.find ("constant-transport build") == std::string::npos ? 1 : 0;
    }
  return 1;
#else
  for (int dim : { 2, 3 })
    {
      for (double R : { 1., 1. / 600. })
        {
          Check (dim, R, 1.);
          Check (dim, R, 7.);
          Check (dim, R, 1., "subsonic", .25);
          Check (dim, R, 1., "supersonic", 5.);
        }
    }
  std::cout << checks << " checks; failures=" << failures << '\n';
  return failures ? 1 : 0;
#endif
}
