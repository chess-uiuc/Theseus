// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "GasModel.hpp"
#include "SourceConfiguration.hpp"
#include "verification/mms/TrigonometricSutherlandMMS.hpp"
#include "cns_reference.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
int checks = 0;
int failures = 0;
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

void
Check (int dim, double R, double mu0, double scale)
{
  constexpr int points = 12;
  mfem::Vector coordinates (points * dim), state (points * (dim + 2)), rhs (state.Size () + 2);
  coordinates.UseDevice ();
  state.UseDevice ();
  rhs.UseDevice ();
  for (int i = 0; i < coordinates.Size (); ++i)
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
          { "mu", 777. },
          { "mu0", mu0 },
          { "sources",
            { { { "type", "trigonometric-cns-sutherland-mms" }, { "time_scale", scale } } } } };
  auto system = Theseus::Sources::CreateConfigured (runtime, dim + 2, dim, false);
  system->Initialize (mesh);
  Theseus::IdealGasModel gas (Theseus::PhysicsConstants (1.4, .72, R, 777., mu0),
                              Theseus::StateLayout (dim, 1));
  double minimum_mu = std::numeric_limits<double>::max ();
  double maximum_mu = 0.;
  double momentum_gradient_signal = 0.;
  double energy_gradient_signal = 0.;
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
          double reference[5], analytic_state[5], forcing[5], frozen_state[5], frozen_forcing[5];
          CNSReference::Residual (dim, x, time, mu0, 0., R, scale, reference, true);
          Theseus::Sources::TrigonometricSutherlandMMS (dim, x, time, mu0, .72, R, analytic_state,
                                                        forcing, scale);
          Theseus::Sources::TrigonometricSutherlandMMS (dim, x, time, mu0, .72, R, frozen_state,
                                                        frozen_forcing, scale, 0.);
          double euler_state[5], euler_source[5], zero_state[5], zero_source[5];
          Theseus::Sources::TrigonometricEulerMMS (dim, x, time, euler_state, euler_source, scale);
          Theseus::Sources::TrigonometricSutherlandMMS (dim, x, time, 0., .72, R, zero_state,
                                                        zero_source, scale);
          for (int q = 0; q < dim + 2; ++q)
            {
              Near (forcing[q], reference[q], 3.e-7);
              Near (rhs.HostRead ()[1 + q * points + point], .5 + reference[q], 3.e-7);
              Near (analytic_state[q], euler_state[q], 0.);
              Near (zero_source[q], euler_source[q], 0.);
              Near (zero_state[q], euler_state[q], 0.);
            }
          for (int i = 0; i < dim; ++i)
            {
              momentum_gradient_signal = std::max (
                  momentum_gradient_signal, std::abs (forcing[i + 1] - frozen_forcing[i + 1]));
            }
          energy_gradient_signal = std::max (energy_gradient_signal,
                                             std::abs (forcing[dim + 1] - frozen_forcing[dim + 1]));
          const auto temperature = CNSReference::Primitive (4, dim, x, time * scale)
                                   / (CNSReference::Primitive (0, dim, x, time * scale) * R);
          const auto viscosity = mu0 * std::pow (temperature / 273.15, 1.5) * (273.15 + 110.4)
                                 / (temperature + 110.4);
          Theseus::DofStateView view (analytic_state, 0);
          Near (gas.viscosity (view), viscosity, 1.e-13);
          Near (gas.thermal_conductivity (view), viscosity * 3.5 * R / .72, 1.e-13);
          minimum_mu = std::min (minimum_mu, viscosity);
          maximum_mu = std::max (maximum_mu, viscosity);
        }
    }
  for (int i = 0; i < state.Size (); ++i)
    {
      Near (state.HostRead ()[i], 7., 0.);
    }
  if (R == 1. / 600. && mu0 == .02)
    {
      Near (maximum_mu / minimum_mu > 1.05 ? 1. : 0., 1., 0.);
      Near (momentum_gradient_signal > 1.e-4 ? 1. : 0., 1., 0.);
      Near (energy_gradient_signal > 1.e-4 ? 1. : 0., 1., 0.);
      std::cout << dim << "D mu range " << minimum_mu << ' ' << maximum_mu
                << "; coefficient-gradient signal momentum/energy " << momentum_gradient_signal
                << ' ' << energy_gradient_signal << '\n';
    }
  for (const char *control : { "viscous_scale", "coefficient_gradient_scale" })
    {
      auto diagnostic = runtime;
      diagnostic["sources"][0][control] = 0.;
      auto diagnostic_system = Theseus::Sources::CreateConfigured (diagnostic, dim + 2, dim, false);
      diagnostic_system->Initialize (mesh);
      rhs = .5;
      diagnostic_system->AddStage ({ .173, .001, state.Read () }, rhs.ReadWrite () + 1);
      Near (rhs.HostRead ()[0], .5, 0.);
      Near (rhs.HostRead ()[rhs.Size () - 1], .5, 0.);
      for (int point = 0; point < points; ++point)
        {
          double x[3] = {};
          for (int d = 0; d < dim; ++d)
            {
              x[d] = coordinates.HostRead ()[point * dim + d];
            }
          double expected_source[5];
          if (std::string (control) == "viscous_scale")
            {
              CNSReference::Residual (dim, x, .173, 0., 0., R, scale, expected_source, true);
            }
          else
            {
              const auto temperature = CNSReference::Primitive (4, dim, x, .173 * scale)
                                       / (CNSReference::Primitive (0, dim, x, .173 * scale) * R);
              const auto local_mu = mu0 * std::pow (temperature / 273.15, 1.5) * (273.15 + 110.4)
                                    / (temperature + 110.4);
              // Independently difference the flux with this point's coefficients
              // held fixed, rather than calling the analytic diagnostic source.
              CNSReference::Residual (dim, x, .173, local_mu, local_mu * 3.5 * R / .72, R, scale,
                                      expected_source);
            }
          for (int q = 0; q < dim + 2; ++q)
            {
              Near (rhs.HostRead ()[1 + q * points + point], .5 + expected_source[q], 3.e-7);
            }
        }
    }
  for (const char *key :
       { "mu0", "Pr", "R_gas", "time_scale", "viscous_scale", "coefficient_gradient_scale" })
    {
      auto invalid = runtime;
      const std::string name (key);
      if (name == "time_scale" || name == "viscous_scale" || name == "coefficient_gradient_scale")
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
#ifndef SUTHERLAND
  const nlohmann::json runtime
      = { { "flow_model", "navier-stokes" },
          { "sources", { { { "type", "trigonometric-cns-sutherland-mms" } } } } };
  try
    {
      Theseus::Sources::CreateConfigured (runtime, 4, 2, false);
    }
  catch (const std::invalid_argument &error)
    {
      std::cout << error.what () << '\n';
      return std::string (error.what ()).find ("requires a Sutherland build") == std::string::npos;
    }
  return 1;
#else
  for (int dim : { 2, 3 })
    {
      for (double R : { 1., 1. / 600. })
        {
          for (double mu0 : { .02, 1.716e-5 })
            {
              Check (dim, R, mu0, 1.);
              Check (dim, R, mu0, 7.);
            }
        }
    }
  std::cout << checks << " checks; failures=" << failures << '\n';
  return failures ? 1 : 0;
#endif
}
