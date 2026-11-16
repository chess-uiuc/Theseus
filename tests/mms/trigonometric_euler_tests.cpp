// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "SourceConfiguration.hpp"
#include "verification/mms/TrigonometricEulerMMS.hpp"
#include <cmath>
#include <iostream>

namespace
{
  int checks = 0;
  int failures = 0;

  void Near(double actual, double expected, double tolerance)
  {
    ++checks;
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > tolerance*(1.0 + std::abs(expected)))
      {
        ++failures;
        if (failures < 12)
          {
            std::cerr << actual << " expected " << expected << '\n';
          }
      }
  }

  // Independent primitive formulas: no production field, source or flux calls.
  double Primitive(int field, int dimension, const double *x, double time)
  {
    const double mean[] = {2., .7, .3, .2, 1.};
    const double ax[] = {.12, .07, -.05, .04, .08};
    const double ay[] = {.08, -.04, .06, -.03, -.06};
    const double az[] = {-.06, .03, -.04, .05, .04};
    const double axy[] = {.04, .02, .03, -.02, .03};
    const double axz[] = {.03, -.02, .01, .03, -.02};
    const double ayz[] = {-.02, .01, -.02, .02, .02};
    const double px = 2.*std::acos(-1.)*(x[0] - .2*time) + .17*field;
    const double py = 2.*std::acos(-1.)*(x[1] + .13*time) + .29*field;
    double value = mean[field] + ax[field]*std::sin(px) + ay[field]*std::cos(py)
      + axy[field]*std::sin(px)*std::cos(py);
    if (dimension == 3)
      {
        const double pz = 2.*std::acos(-1.)*(x[2] - .11*time) + .31*field;
        value += az[field]*std::sin(pz) + axz[field]*std::cos(px)*std::sin(pz)
          + ayz[field]*std::sin(py)*std::cos(pz);
      }
    return value;
  }

  void StateOrFlux(int dimension, const double *x, double time, int direction, double *value)
  {
    const double rho = Primitive(0, dimension, x, time);
    const double pressure = Primitive(4, dimension, x, time);
    double velocity[3] = {};
    double energy = pressure/(1.4 - 1.);
    for (int d = 0; d < dimension; ++d)
      {
        velocity[d] = Primitive(d + 1, dimension, x, time);
        energy += .5*rho*velocity[d]*velocity[d];
      }
    if (direction == dimension)
      {
        value[0] = rho;
        value[dimension + 1] = energy;
        for (int d = 0; d < dimension; ++d)
          {
            value[d + 1] = rho*velocity[d];
          }
        return;
      }
    value[0] = rho*velocity[direction];
    value[dimension + 1] = (energy + pressure)*velocity[direction];
    for (int d = 0; d < dimension; ++d)
      {
        value[d + 1] = rho*velocity[d]*velocity[direction] + (d == direction ? pressure : 0.);
      }
  }

  void Residual(int dimension, const double *x, double time, double *source, double time_scale)
  {
    constexpr double step = 2.e-4;
    const int offsets[] = {-2, -1, 1, 2};
    const double weights[] = {1., -8., 8., -1.};
    for (int q = 0; q < dimension + 2; ++q)
      {
        source[q] = 0.;
      }
    for (int direction = 0; direction <= dimension; ++direction)
      {
        for (int sample = 0; sample < 4; ++sample)
          {
            double position[] = {x[0], x[1], x[2]};
            double shifted_time = time;
            if (direction == dimension)
              {
                shifted_time += offsets[sample]*step/time_scale;
              }
            else
              {
                position[direction] += offsets[sample]*step;
              }
            double value[5];
            StateOrFlux(dimension, position, shifted_time*time_scale, direction, value);
            for (int q = 0; q < dimension + 2; ++q)
              {
                source[q] += weights[sample]*value[q]/(12.*step)*(direction == dimension ? time_scale : 1.);
              }
          }
      }
  }

  void CheckDimension(int dimension, int rank, int orientation, double time_scale)
  {
    constexpr int points = 48;
    Theseus::Sources::MeshView mesh;
    mesh.elements = 1;
    mesh.dofs = points;
    mesh.dimension = dimension;
    mesh.equations = dimension + 2;
    mesh.layout = Theseus::StateLayout(dimension, 1);
    mesh.communicator = MPI_COMM_WORLD;
    mfem::Vector coordinates(points*dimension), state(points*mesh.equations);
    mfem::Vector rhs(points*mesh.equations + 2);
    coordinates.UseDevice();
    state.UseDevice();
    rhs.UseDevice();
    auto *coordinate_host = coordinates.HostWrite();
    for (int point = 0; point < points; ++point)
      {
        for (int d = 0; d < dimension; ++d)
          {
            coordinate_host[point*dimension + d] = std::sin(.37*point + .53*d + .19*rank);
          }
      }
    mesh.coordinates = coordinates.Read();
    auto config = nlohmann::json::parse(
      R"({"R_gas":1,"sources":[{"type":"trigonometric-euler-mms"}]})");
    config["sources"][0]["orientation"] = orientation;
    config["sources"][0]["time_scale"] = time_scale;
    auto system = Theseus::Sources::CreateConfigured(config, mesh.equations, dimension, false);
    system->Initialize(mesh);
    state = 7.;
    for (double time : {0., .173, .431})
      {
        rhs = .5;
        system->AddStage({time, .01, state.Read()}, rhs.ReadWrite() + 1);
        const auto *actual = rhs.HostRead();
        Near(actual[0], .5, 0.);
        Near(actual[rhs.Size() - 1], .5, 0.);
        for (int point = 0; point < points; ++point)
          {
            double x[3] = {};
            for (int d = 0; d < dimension; ++d)
              {
                x[d] = coordinates.HostRead()[point*dimension + d];
              }
            double reference[5], exact_state[5], exact_source[5], independent_state[5];
            double canonical_position[3] = {};
            for (int d = 0; d < dimension; ++d)
              {
                canonical_position[d] = x[(d + orientation) % dimension];
              }
            double canonical_source[5], canonical_state[5];
            Residual(dimension, canonical_position, time, canonical_source, time_scale);
            StateOrFlux(dimension, canonical_position, time*time_scale, dimension, canonical_state);
            reference[0] = canonical_source[0];
            independent_state[0] = canonical_state[0];
            reference[dimension + 1] = canonical_source[dimension + 1];
            independent_state[dimension + 1] = canonical_state[dimension + 1];
            for (int d = 0; d < dimension; ++d)
              {
                reference[(d + orientation) % dimension + 1] = canonical_source[d + 1];
                independent_state[(d + orientation) % dimension + 1] = canonical_state[d + 1];
              }
            Theseus::Sources::OrientedTrigonometricEulerMMS(
              dimension, orientation, x, time, exact_state, exact_source, time_scale);
            for (int q = 0; q < mesh.equations; ++q)
              {
                Near(actual[1 + q*points + point], .5 + reference[q], 2.e-9);
                Near(exact_state[q], independent_state[q], 2.e-14);
              }
            for (int d = 0; d < dimension; ++d)
              {
                x[d] += 1.;
                double shifted_state[5], shifted_source[5];
                Theseus::Sources::OrientedTrigonometricEulerMMS(
                  dimension, orientation, x, time, shifted_state, shifted_source, time_scale);
                for (int q = 0; q < mesh.equations; ++q)
                  {
                    Near(shifted_state[q], exact_state[q], 2.e-13);
                    Near(shifted_source[q], exact_source[q], 2.e-13);
                  }
                x[d] -= 1.;
              }
          }
      }
    const auto *unchanged = state.HostRead();
    for (int i = 0; i < state.Size(); ++i)
      {
        Near(unchanged[i], 7., 0.);
      }
  }

  void CheckRejections()
  {
    using nlohmann::json;
    const auto valid = json::parse(R"({"R_gas":1,"sources":[{"type":"trigonometric-euler-mms"}]})");
    for (int test = 0; test < 12; ++test)
      {
        auto config = valid;
        if (test == 0) { config["gamma"] = 1.3; }
        if (test == 1) { config["gas_model"] = "lte"; }
        if (test == 2) { config["flow_model"] = "navier-stokes"; }
        if (test == 3) { config["R_gas"] = 287.; }
        if (test == 7) { config["sources"][0]["orientation"] = -1; }
        if (test == 8) { config["sources"][0]["orientation"] = 2; }
        if (test == 9) { config["sources"][0]["orientation"] = .5; }
        if (test == 10) { config["sources"][0]["time_scale"] = 0.; }
        if (test == 11) { config["sources"][0]["time_scale"] = -1.; }
        ++checks;
        try
          {
            Theseus::Sources::CreateConfigured(config, test == 5 ? 5 : 4,
              test == 6 ? 1 : 2, test == 4);
            ++failures;
          }
        catch (const std::invalid_argument &) {}
      }
  }
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA))
    {
      return 1;
    }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int dimension : {2, 3})
    {
      for (int orientation = 0; orientation < dimension; ++orientation)
        {
          CheckDimension(dimension, rank, orientation, 1.);
          CheckDimension(dimension, rank, orientation, 31.);
        }
    }
  CheckRejections();
  int total_failures;
  MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    {
      std::cout << checks << " checks/rank; failures=" << total_failures << '\n';
    }
  return total_failures ? 1 : 0;
}
