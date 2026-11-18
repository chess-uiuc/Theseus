// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "EulerOperator.hpp"
#include "NSOperator.hpp"
#include "RoeFlux.hpp"
#include "SourceRegistry.hpp"
#include "verification/mms/TrigonometricSutherlandMMS.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
struct Physics
{
  using GasModel = Theseus::IdealGasModel;
  using InviscidFlux = Theseus::RoeFlux::InviscidFlux;
};

struct Frame
{
  mfem::real_t cosine = 1.;
  mfem::real_t sine = 0.;
};

void
Require (bool condition, const std::string &message)
{
  if (!condition)
    {
      throw std::runtime_error (message);
    }
}

// Rotate the physical mesh, exact velocity and momentum forcing together.
// This is an orthogonal coordinate change, not a shear of the governing PDE.
MFEM_HOST_DEVICE void
Exact (Frame frame, int physics, const mfem::real_t *x, mfem::real_t time, mfem::real_t *state,
       mfem::real_t *source)
{
  const mfem::real_t canonical[]
      = { frame.cosine * x[0] + frame.sine * x[1], -frame.sine * x[0] + frame.cosine * x[1] };
  mfem::real_t u[4], s[4];
  if (physics == 0)
    {
      Theseus::Sources::TrigonometricEulerMMS (2, canonical, time, u, s);
    }
  else if (physics == 1)
    {
      Theseus::Sources::TrigonometricCNSMMS (2, canonical, time, .02, .02 * 3.5 / .72, 1., u, s);
    }
  else
    {
      Theseus::Sources::TrigonometricSutherlandMMS (2, canonical, time, .02, .72, 1. / 600., u, s);
    }
  state[0] = u[0];
  state[1] = frame.cosine * u[1] - frame.sine * u[2];
  state[2] = frame.sine * u[1] + frame.cosine * u[2];
  state[3] = u[3];
  source[0] = s[0];
  source[1] = frame.cosine * s[1] - frame.sine * s[2];
  source[2] = frame.sine * s[1] + frame.cosine * s[2];
  source[3] = s[3];
}

struct ManufacturedSource : Theseus::Sources::AnalyticModel
{
  Frame frame;
  int physics;
  Theseus::Sources::Requirements
  Needs () const
  {
    return { true };
  }
  ManufacturedSource
  MakeEvaluator (const Theseus::Sources::Stage &) const
  {
    return *this;
  }
  MFEM_HOST_DEVICE void
  operator() (const Theseus::Sources::Point &point,
              const Theseus::Sources::SourceAccumulator &output) const
  {
    const mfem::real_t x[] = { point.Coordinate (0), point.Coordinate (1) };
    mfem::real_t state[4], forcing[4];
    Exact (frame, physics, x, point.stage.time, state, forcing);
    for (int q = 0; q < 4; ++q)
      {
        output.Add (q, forcing[q]);
      }
  }
};

// Explicit integer hashing keeps perturbations reproducible across standard libraries.
double
RandomSigned (std::uint32_t value)
{
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return 2. * double (value >> 8) / 16777216. - 1.;
}

mfem::Mesh
Mesh (int cells, const std::string &kind, Frame frame)
{
  auto mesh = mfem::Mesh::MakeCartesian2D (cells, cells, mfem::Element::QUADRILATERAL, true, 2., 2.,
                                           false);
  const double pi = std::acos (-1.);
  for (int vertex = 0; vertex < mesh.GetNV (); ++vertex)
    {
      auto *point = mesh.GetVertex (vertex);
      const double x = point[0], y = point[1];
      if (kind == "graded")
        {
          point[0] += .35 * std::sin (pi * x) / pi;
          point[1] += .25 * std::sin (pi * y) / pi;
        }
      else if (kind == "sheared")
        {
          // Period shifts (2,0) and (1,2) preserve the unit-period exact fields.
          point[0] += .5 * y;
        }
      else if (kind == "rotated")
        {
          point[0] = frame.cosine * x - frame.sine * y;
          point[1] = frame.sine * x + frame.cosine * y;
        }
      else if (kind == "jitter17" || kind == "jitter91")
        {
          // Pin the outer boundary, preserving exact periodic face matching.
          if (x > 1.e-12 && x < 2. - 1.e-12 && y > 1.e-12 && y < 2. - 1.e-12)
            {
              const std::uint32_t seed = kind == "jitter17" ? 17u : 91u;
              point[0] += .12 * (2. / cells) * RandomSigned (seed + 2u * vertex);
              point[1] += .12 * (2. / cells) * RandomSigned (seed + 2u * vertex + 1u);
            }
        }
    }
  std::vector<mfem::Vector> shifts (2, mfem::Vector (2));
  shifts[0] = 0.;
  shifts[1] = 0.;
  shifts[0](0) = 2.;
  shifts[1](1) = 2.;
  if (kind == "rotated")
    {
      shifts[0](0) = 2. * frame.cosine;
      shifts[0](1) = 2. * frame.sine;
      shifts[1](0) = -2. * frame.sine;
      shifts[1](1) = 2. * frame.cosine;
    }
  if (kind == "sheared")
    {
      shifts[1](0) = 1.;
    }
  auto periodic = mfem::Mesh::MakePeriodic (mesh, mesh.CreatePeriodicVertexMapping (shifts));
  periodic.RemoveInternalBoundaries ();
  return periodic;
}

std::array<std::array<double, 2>, 4>
ExactGradient (Frame frame, const mfem::Vector &x, double time)
{
  const mfem::real_t canonical[]
      = { frame.cosine * x (0) + frame.sine * x (1), -frame.sine * x (0) + frame.cosine * x (1) };
  const double rotation[2][2] = { { frame.cosine, -frame.sine }, { frame.sine, frame.cosine } };
  const int fields[] = { 0, 1, 2, 4 };
  std::array<std::array<double, 2>, 4> result{};
  for (int q = 0; q < 4; ++q)
    {
      for (int d = 0; d < 2; ++d)
        {
          for (int direction = 0; direction < 2; ++direction)
            {
              if (q == 0 || q == 3)
                {
                  const auto p
                      = Theseus::Sources::TrigonometricPrimitive (fields[q], 2, canonical, time);
                  result[q][d] += rotation[d][direction] * p.gradient[direction];
                }
              else
                {
                  for (int component = 0; component < 2; ++component)
                    {
                      const auto p = Theseus::Sources::TrigonometricPrimitive (component + 1, 2,
                                                                               canonical, time);
                      result[q][d] += rotation[q - 1][component] * p.gradient[direction]
                                      * rotation[d][direction];
                    }
                }
            }
        }
    }
  return result;
}

struct Norms
{
  std::array<double, 3> solution{};
  std::array<double, 4> gradient{};
  double minimum_jacobian = std::numeric_limits<double>::max ();
  double maximum_jacobian = 0.;
  double volume = 0.;
};

Norms
Integrate (mfem::ParMesh &mesh, mfem::ParGridFunction &state,
           const std::array<mfem::ParGridFunction *, 2> &gradients, Frame frame, int physics,
           double time, int quadrature_points)
{
  Norms result;
  const auto &rule = mfem::IntRules.Get (mfem::Geometry::SQUARE, 2 * quadrature_points - 1);
  mfem::Vector x (2), value (4), gradient (4);
  for (int e = 0; e < mesh.GetNE (); ++e)
    {
      auto *transformation = mesh.GetElementTransformation (e);
      for (int point = 0; point < rule.GetNPoints (); ++point)
        {
          const auto &ip = rule.IntPoint (point);
          transformation->SetIntPoint (&ip);
          const double determinant = transformation->Jacobian ().Det ();
          Require (std::isfinite (determinant) && determinant > 0., "Invalid/inverted cell");
          result.minimum_jacobian = std::min (result.minimum_jacobian, determinant);
          result.maximum_jacobian = std::max (result.maximum_jacobian, determinant);
          const double weight = ip.weight * determinant;
          result.volume += weight;
          transformation->Transform (ip, x);
          state.GetVectorValue (e, ip, value);
          mfem::real_t exact[4], source[4];
          Exact (frame, physics, x.GetData (), time, exact, source);
          Require (
              std::isfinite (value.Normlinf ()) && value (0) > 0.
                  && .4
                             * (value (3)
                                - .5 * (value (1) * value (1) + value (2) * value (2)) / value (0))
                         > 0.,
              "Nonfinite/nonphysical evolved state");
          result.solution[0] += weight * std::pow (value (0) - exact[0], 2);
          result.solution[1]
              += weight * (std::pow (value (1) - exact[1], 2) + std::pow (value (2) - exact[2], 2));
          result.solution[2] += weight * std::pow (value (3) - exact[3], 2);
          if (gradients[0])
            {
              const auto reference = ExactGradient (frame, x, time);
              for (int d = 0; d < 2; ++d)
                {
                  gradients[d]->GetVectorValue (e, ip, gradient);
                  for (int q = 0; q < 4; ++q)
                    {
                      result.gradient[q] += weight * std::pow (gradient (q) - reference[q][d], 2);
                    }
                }
            }
        }
    }
  for (auto &error : result.solution)
    {
      error = std::sqrt (error);
    }
  for (auto &error : result.gradient)
    {
      error = std::sqrt (error);
    }
  return result;
}

nlohmann::json
Run (int cells, int degree, const std::string &kind, int physics)
{
  Frame frame;
  if (kind == "rotated")
    {
      frame.cosine = std::cos (27. * std::acos (-1.) / 180.);
      frame.sine = std::sin (27. * std::acos (-1.) / 180.);
    }
  auto mesh = Mesh (cells, kind, frame);
  auto pmesh = std::make_shared<mfem::ParMesh> (MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection collection (degree, 2, mfem::BasisType::GaussLobatto);
  auto space = std::make_shared<mfem::ParFiniteElementSpace> (pmesh.get (), &collection, 4,
                                                              mfem::Ordering::byNODES);
  space->ExchangeFaceNbrData ();
  std::vector<std::shared_ptr<mfem::ParGridFunction>> storage;
  for (int d = 0; d < 2; ++d)
    {
      storage.push_back (std::make_shared<mfem::ParGridFunction> (space.get ()));
    }
  const double gas_constant = physics == 2 ? 1. / 600. : 1.;
  auto gas = std::make_shared<Theseus::IdealGasModel> (
      Theseus::PhysicsConstants (1.4, .72, gas_constant, .02, .02),
      Theseus::StateLayout (2, space->GetNDofs ()));
  std::unique_ptr<Theseus::RHSOperator<Physics>> op;
  if (physics == 0)
    {
      op = std::make_unique<Theseus::EulerOperator<Physics>> (
          space, nullptr, pmesh, nullptr, nullptr, nullptr, gas, "CPG1", "Roe", 0.);
    }
  else
    {
      op = std::make_unique<Theseus::NSOperator<Physics>> (
          space, nullptr, pmesh, nullptr, nullptr, storage, nullptr, gas, "CPG1", "Roe", 0.);
    }
  op->Finalize ();
  mfem::ParGridFunction state (space.get ());
  mfem::Vector u, rhs (space->GetTrueVSize ());
  mfem::VectorFunctionCoefficient uniform (4,
                                           [] (const mfem::Vector &, mfem::Vector &value)
                                             {
                                               value (0) = 2.;
                                               value (1) = 1.4;
                                               value (2) = .6;
                                               value (3) = 3.08;
                                             });
  state.ProjectCoefficient (uniform);
  state.GetTrueDofs (u);
  op->Mult (u, rhs);
  const double freestream = rhs.Normlinf ();
  Require (std::isfinite (freestream) && freestream < 1.e-10, "Free-stream RHS exceeds tolerance");

  mfem::VectorFunctionCoefficient initial (
      4,
      [frame, physics] (const mfem::Vector &x, mfem::Vector &value)
        {
          mfem::real_t forcing[4];
          Exact (frame, physics, x.GetData (), 0., value.GetData (), forcing);
        });
  state.ProjectCoefficient (initial);
  state.GetTrueDofs (u);
  Norms gradient_norm;
  if (physics != 0)
    {
      auto &ns = dynamic_cast<Theseus::NSOperator<Physics> &> (*op);
      mfem::Vector entropy, gx (space->GetTrueVSize ()), gy (space->GetTrueVSize ());
      ns.ComputeEntropyState (u, entropy);
      std::vector<mfem::Vector *> gradients{ &gx, &gy };
      ns.GradOperator (entropy, gradients);
      ns.ComputeGradPrimFromGradEntropy (u, gradients);
      mfem::ParGridFunction xgradient (space.get ()), ygradient (space.get ());
      xgradient.SetFromTrueDofs (gx);
      ygradient.SetFromTrueDofs (gy);
      gradient_norm = Integrate (*pmesh, state, { &xgradient, &ygradient }, frame, physics, 0., 10);
    }
  auto sources = std::make_unique<Theseus::Sources::System> ();
  ManufacturedSource model;
  model.frame = frame;
  model.physics = physics;
  sources->Add (std::make_unique<Theseus::Sources::PointSource<ManufacturedSource>> (model));
  op->SetSources (std::move (sources));
  op->InitializeSources ();
  constexpr double final_time = .05;
  constexpr int steps = 200;
  constexpr double timestep = final_time / steps;
  op->SetStepSize (timestep);
  mfem::RK4Solver solver;
  solver.Init (*op);
  double time = 0.;
  for (int step = 0; step < steps; ++step)
    {
      double dt = timestep;
      solver.Step (u, time, dt);
    }
  state.SetFromTrueDofs (u);
  const auto norms = Integrate (*pmesh, state, { nullptr, nullptr }, frame, physics, time, 12);
  const auto lower = Integrate (*pmesh, state, { nullptr, nullptr }, frame, physics, time, 10);
  for (int q = 0; q < 3; ++q)
    {
      Require (std::abs (norms.solution[q] - lower.solution[q])
                   < 1.e-6 * norms.solution[q] + 1.e-12,
               "Physical error quadrature is unresolved");
    }
  Require (std::abs (norms.volume - 4.) < 1.e-10, "Incorrect physical domain area");
  const char *name = physics == 0 ? "euler" : (physics == 1 ? "constant" : "sutherland");
  nlohmann::json record = { { "physics", name },
                            { "mesh", kind },
                            { "degree", degree },
                            { "cells", cells },
                            { "h", 2. / cells },
                            { "solution_l2", norms.solution },
                            { "gradient_l2", gradient_norm.gradient },
                            { "free_stream_rhs_linf", freestream },
                            { "minimum_scaled_jacobian", norms.minimum_jacobian * cells * cells },
                            { "jacobian_ratio", norms.maximum_jacobian / norms.minimum_jacobian },
                            { "area", norms.volume },
                            { "steps", steps },
                            { "dt", timestep },
                            { "final_time", time } };
  if (cells == 12 && degree == 2)
    {
      const auto &vertices = *mfem::Geometries.GetVertices (mfem::Geometry::SQUARE);
      auto &preview = record["preview_quads"];
      preview = nlohmann::json::array ();
      for (int element = 0; element < pmesh->GetNE (); ++element)
        {
          auto *transformation = pmesh->GetElementTransformation (element);
          nlohmann::json quad = nlohmann::json::array ();
          mfem::Vector x (2);
          for (int corner = 0; corner < 4; ++corner)
            {
              transformation->Transform (vertices.IntPoint (corner), x);
              quad.push_back ({ x (0), x (1) });
            }
          preview.push_back (quad);
        }
    }
  std::cout << name << ' ' << kind << " P" << degree << " n=" << cells << " errors "
            << norms.solution[0] << ' ' << norms.solution[1] << ' ' << norms.solution[2]
            << std::endl;
  return record;
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
  const std::string profile = argc > 3 ? argv[3] : "quick";
  Require (profile == "quick" || profile == "extended", "Profile must be quick or extended");
  const std::vector<int> degrees
      = profile == "extended" ? std::vector<int>{ 2, 3 } : std::vector<int>{ 2 };
  const std::vector<int> resolutions
      = profile == "extended" ? std::vector<int>{ 12, 18, 27 } : std::vector<int>{ 12 };
  int ranks;
  MPI_Comm_size (MPI_COMM_WORLD, &ranks);
  Require (ranks == 1, "Mesh sensitivity fixture is a one-rank operator study");
  nlohmann::json report = { { "backend", backend },
                            { "dimension", 2 },
                            { "flux", "Roe" },
                            { "pure_dg", true },
                            { "records", nlohmann::json::array () },
                            { "issues", nlohmann::json::array () } };
  report["profile"] = profile;
  report["checks"]
      = profile == "quick"
            ? "Geometry, free stream, physical evolution, quadrature and rotation covariance; no "
              "order qualification"
            : "Three-mesh solution/gradient refinement plus geometry and rotation checks";
#ifdef SUTHERLAND
  const std::vector<int> models{ 2 };
#else
  const std::vector<int> models{ 0, 1 };
#endif
  for (int physics : models)
    {
      for (const std::string kind :
           { "uniform", "graded", "rotated", "sheared", "jitter17", "jitter91" })
        {
          for (int degree : degrees)
            {
              std::vector<nlohmann::json> group;
              for (int cells : resolutions)
                {
                  try
                    {
                      auto record = Run (cells, degree, kind, physics);
                      report["records"].push_back (record);
                      group.push_back (record);
                    }
                  catch (const std::exception &error)
                    {
                      report["issues"].push_back ({ { "mesh", kind },
                                                    { "degree", degree },
                                                    { "physics", physics },
                                                    { "cells", cells },
                                                    { "error", error.what () } });
                    }
                }
              if (group.size () == 3)
                {
                  for (int q = 0; q < 3; ++q)
                    {
                      const double order = std::log (group[1]["solution_l2"][q].get<double> ()
                                                     / group[2]["solution_l2"][q].get<double> ())
                                           / std::log (1.5);
                      if (!std::isfinite (order) || order < degree + .5)
                        {
                          report["issues"].push_back (
                              { { "mesh", kind },
                                { "degree", degree },
                                { "physics", physics },
                                { "quantity", q },
                                { "observed_order", order },
                                { "error", "Evolved solution order below nominal minus .5" } });
                        }
                    }
                  if (physics != 0)
                    {
                      for (int q = 0; q < 4; ++q)
                        {
                          const double order
                              = std::log (group[1]["gradient_l2"][q].get<double> ()
                                          / group[2]["gradient_l2"][q].get<double> ())
                                / std::log (1.5);
                          if (!std::isfinite (order) || order < degree - .5)
                            {
                              report["issues"].push_back (
                                  { { "mesh", kind },
                                    { "degree", degree },
                                    { "physics", physics },
                                    { "quantity", q },
                                    { "observed_order", order },
                                    { "error", "Gradient order below p minus .5" } });
                            }
                        }
                    }
                }
            }
        }
    }
  // Rotation covariance is stronger than simply observing convergence.
  for (const auto &record : report["records"])
    {
      if (record["mesh"] != "rotated")
        {
          continue;
        }
      for (const auto &reference : report["records"])
        {
          if (reference["mesh"] != "uniform" || reference["physics"] != record["physics"]
              || reference["degree"] != record["degree"] || reference["cells"] != record["cells"])
            {
              continue;
            }
          for (int q = 0; q < 3; ++q)
            {
              const double error = record["solution_l2"][q];
              const double baseline = reference["solution_l2"][q];
              if (std::abs (error - baseline) > 1.e-6 * baseline + 2.e-12)
                {
                  report["issues"].push_back ({ { "mesh", "rotated" },
                                                { "physics", record["physics"] },
                                                { "degree", record["degree"] },
                                                { "cells", record["cells"] },
                                                { "error", "Rotation covariance failed" } });
                }
            }
        }
    }
  report["passed"] = report["issues"].empty ();
  if (argc > 2)
    {
      std::ofstream output (argv[2]);
      output << report.dump (2) << '\n';
      Require (bool (output), "Cannot write mesh sensitivity evidence");
    }
  std::cout << "Mesh sensitivity: " << report["records"].size () << " records; "
            << report["issues"].size () << " issues\n"
            << report["issues"].dump (2) << std::endl;
  return report["passed"].get<bool> () ? 0 : 1;
}
