// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "SourceConfiguration.hpp"
#include "sources/Affine.hpp"
#include "sources/EulerMMS.hpp"
#include "sources/LinearDrag.hpp"
#include <cmath>
#include <iostream>
#include <limits>

using namespace Theseus::Sources;
namespace
{
  // Same generic operation used by the production RHS; no model dispatch here.
  void AddPhysicalSources(System &system, const Stage &stage, mfem::real_t *rhs)
  {
    system.AddStage(stage, rhs);
  }
  int checks = 0;
  int failures = 0;
  void Near(double actual, double expected, double tolerance = 2e-12)
  {
    ++checks;
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > tolerance*(1.0 + std::abs(expected)))
      {
        if (++failures < 10)
          {
            std::cerr << actual << " expected " << expected << '\n';
          }
      }
  }
  template<typename Function> void Reject(Function action)
  {
    ++checks;
    try
      {
        action();
        ++failures;
      }
    catch (const std::exception &) {}
  }

  // Test-only third implementation. Neither core nor RHSDriver knows this type.
  struct Lifecycle
  {
    int initialize = 0;
    int prepare = 0;
    const mfem::real_t *allocation = nullptr;
  };
  struct PreparedProbe : AnalyticModel
  {
    std::shared_ptr<Lifecycle> lifecycle;
    mfem::Vector coefficients;
    explicit PreparedProbe(std::shared_ptr<Lifecycle> value) : lifecycle(std::move(value)) {}
    void Initialize(const MeshView &)
    {
      ++lifecycle->initialize;
      coefficients.SetSize(2);
      coefficients.UseDevice();
      coefficients = 0.0;
      lifecycle->allocation = coefficients.Read();
    }
    void Prepare(const Stage &stage)
    {
      ++lifecycle->prepare;
      const auto time = stage.time;
      const auto dt = stage.step_size;
      auto *values = coefficients.Write();
      mfem::forall(2, [=] MFEM_HOST_DEVICE (int i)
      {
        values[i] = i == 0 ? time : dt;
      });
      // Pointer comparison is host metadata, not a data readback.
      Near(coefficients.Read() == lifecycle->allocation, 1);
    }
    struct Evaluator
    {
      const mfem::real_t *coefficients;
      MFEM_HOST_DEVICE void operator()(const Point &point, const SourceAccumulator &out) const
      {
        const auto mass = point.mesh.layout.eq_mass;
        out.Add(mass, coefficients[0] + coefficients[1]*point.State(mass));
      }
    };
    Evaluator MakeEvaluator(const Stage &) const { return {coefficients.Read()}; }
  };

  // Independent ideal-gas state/flux formulas for a numerical residual reference.
  void StateAndFlux(double x, double y, double t, int direction, double *value)
  {
    const double sine = std::sin(2.0*std::acos(-1.0)*(x + y - .4*t));
    const double rho = 2.0 + .2*sine;
    const double u = .7 + .1*sine;
    const double v = .3 - .05*sine;
    const double pressure = 1.0 + .15*sine;
    const double energy = pressure/.4 + .5*rho*(u*u + v*v);
    if (direction == 2)
      {
        value[0] = rho;
        value[1] = rho*u;
        value[2] = rho*v;
        value[3] = energy;
      }
    else
      {
        const double velocity = direction == 0 ? u : v;
        value[0] = rho*velocity;
        value[1] = rho*u*velocity + (direction == 0 ? pressure : 0.0);
        value[2] = rho*v*velocity + (direction == 1 ? pressure : 0.0);
        value[3] = (energy + pressure)*velocity;
      }
  }
  void Residual(double x, double y, double t, double *result)
  {
    for (int q = 0; q < 4; ++q) { result[q] = 0.0; }
    const int shifts[4] = {-2, -1, 1, 2};
    const double weights[4] = {1., -8., 8., -1.};
    constexpr double h = 1e-4;
    for (int direction = 0; direction < 3; ++direction)
      {
        for (int i = 0; i < 4; ++i)
          {
            double values[4];
            StateAndFlux(x + (direction == 0 ? h*shifts[i] : 0.0),
                         y + (direction == 1 ? h*shifts[i] : 0.0),
                         t + (direction == 2 ? h*shifts[i] : 0.0), direction, values);
            for (int q = 0; q < 4; ++q)
              {
                result[q] += weights[i]*values[q]/(12.0*h);
              }
          }
      }
  }

  void TestSources(int rank)
  {
    Registry registry;
    const nlohmann::json runtime{{"gas_model", "cpg"}, {"flow_model", "euler"}, {"gamma", 1.4}};
    const BuildContext context{2, 4, false, runtime};
    auto create = [&](const nlohmann::json &configuration)
    {
      return registry.Create(configuration, context);
    };
    RegisterMMS(registry);
    RegisterDrag(registry);
    auto lifecycle = std::make_shared<Lifecycle>();
    registry.Register("probe", [lifecycle](const nlohmann::json &, const BuildContext &)
    {
      return std::make_unique<PointSource<PreparedProbe>>(PreparedProbe(lifecycle));
    });
    Reject([&] { create({{"type", "unknown"}}); });
    Reject([&] { create({{"type", "linear-drag"}, {"coefficient", -1.0}}); });
    Reject([&] { create({{"type", "linear-drag"},
                                 {"coefficient", std::numeric_limits<double>::infinity()}}); });
    Reject([&] { RegisterDrag(registry); });
    MeshView mesh;
    mesh.elements = 3 + rank;
    mesh.dofs = 5;
    mesh.equations = 4;
    mesh.dimension = 2;
    mesh.layout = Theseus::StateLayout(2, 1);
    mesh.communicator = MPI_COMM_WORLD;
    const int points = mesh.elements*mesh.dofs;
    const int size = points*mesh.equations;
    mfem::Vector input(size), rhs(size + 2), coordinates(points*2);
    mfem::Array<int> attributes(mesh.elements), marker(2);
    input.UseDevice();
    rhs.UseDevice();
    coordinates.UseDevice();
    attributes.UseDevice(true);
    marker.UseDevice(true);
    marker[0] = 1;
    marker[1] = 0;
    for (int e = 0; e < mesh.elements; ++e)
      {
        attributes[e] = e == 1 ? 2 : 1;
        for (int point = 0; point < mesh.dofs; ++point)
          {
            auto *state = input.HostReadWrite() + e*mesh.dofs*4 + point;
            state[0] = 2.0 + .1*rank + .01*point;
            state[5] = 3.0;
            state[10] = -1.0;
            state[15] = 8.0;
            auto *x = coordinates.HostReadWrite() + (e*mesh.dofs + point)*2;
            x[0] = .07*point + .2*e;
            x[1] = -.13 + .04*rank;
          }
      }
    const mfem::Vector saved(input);
    mesh.element_attributes = attributes.Read();
    mesh.attribute_marker = marker.Read();
    System missing_coordinates;
    missing_coordinates.Add(create({{"type", "periodic-euler-mms"}}));
    Reject([&] { missing_coordinates.Initialize(mesh); });
    mesh.coordinates = coordinates.Read();
    System system;
    system.Add(create({{"type", "periodic-euler-mms"}}));
    system.Add(create({{"type", "linear-drag"}, {"coefficient", .25}}));
    system.Add(create({{"type", "probe"}}));
    Near(system.Needs().coordinates, 1);
    Stage stage;
    Reject([&] { AddPhysicalSources(system, stage, rhs.Write()); });
    system.Initialize(mesh);
    Reject([&] { system.Add(create({{"type", "linear-drag"}, {"coefficient", .25}})); });
    for (int call = 0; call < 3; ++call)
      {
        stage.time = .03*call;
        stage.step_size = call == 1 ? .02 : .01;
        stage.state = input.Read();
        rhs = .5;
        AddPhysicalSources(system, stage, rhs.ReadWrite() + 1);
        const auto *actual = rhs.HostRead();
        Near(actual[0], .5);
        Near(actual[size + 1], .5);
        for (int e = 0; e < mesh.elements; ++e)
          {
            for (int point = 0; point < mesh.dofs; ++point)
              {
                double expected[4];
                Residual(.07*point + .2*e, -.13 + .04*rank, stage.time, expected);
                const double rho = 2.0 + .1*rank + .01*point;
                expected[0] += stage.time + stage.step_size*rho;
                expected[1] -= .25*3.0;
                expected[2] -= .25*(-1.0);
                expected[3] -= .25*(9.0 + 1.0)/rho;
                for (int q = 0; q < 4; ++q)
                  {
                    Near(actual[1 + (e*4 + q)*5 + point],
                         .5 + (e == 1 ? 0.0 : expected[q]), 1e-8);
                  }
              }
          }
      }
    Near(lifecycle->initialize, 1);
    Near(lifecycle->prepare, 3);
    const auto *input_host = input.HostRead();
    const auto *saved_host = saved.HostRead();
    for (int i = 0; i < size; ++i) { Near(input_host[i], saved_host[i]); }
    System empty;
    Near(empty.Needs().coordinates, 0);
    empty.Initialize(mesh);
    rhs = .5;
    AddPhysicalSources(empty, stage, rhs.ReadWrite());
    const auto *unchanged = rhs.HostRead();
    for (int i = 0; i < rhs.Size(); ++i) { Near(unchanged[i], .5); }
  }

  void TestDragDimensions()
  {
    for (int dim = 1; dim <= 3; ++dim)
      {
        MeshView mesh;
        mesh.elements = 1;
        mesh.dofs = 3;
        mesh.dimension = dim;
        mesh.equations = dim + 3;
        mesh.layout = Theseus::StateLayout(dim, 1, 1);
        mesh.communicator = MPI_COMM_WORLD;
        mfem::Vector state(3*mesh.equations), rhs(3*mesh.equations);
        state.UseDevice();
        rhs.UseDevice();
        state = 2.0;
        rhs = .5;
        System system;
        system.Add(std::make_unique<PointSource<DragModel>>(DragModel(.25)));
        Near(system.Needs().coordinates, 0);
        system.Initialize(mesh);
        Stage stage{.3, .001, state.Read()};
        AddPhysicalSources(system, stage, rhs.ReadWrite());
        const auto *actual = rhs.HostRead();
        for (int q = 0; q < mesh.equations; ++q)
          {
            double expected = .5;
            if (q >= mesh.layout.eq_mom0 && q < mesh.layout.eq_mom0 + dim)
              {
                expected -= .5;
              }
            if (q == mesh.layout.eq_energy) { expected -= .5*dim; }
            for (int point = 0; point < mesh.dofs; ++point)
              {
                Near(actual[q*mesh.dofs + point], expected);
              }
          }
      }
  }
  void TestAffine()
  {
    for (int dimension = 1; dimension <= 3; ++dimension)
      {
        MeshView mesh;
        mesh.elements = 2;
        mesh.dofs = 7;
        mesh.dimension = dimension;
        mesh.equations = dimension < 3 ? dimension + 3 : 5;
        mesh.layout = Theseus::StateLayout(dimension, 1, dimension < 3 ? 1 : 0);
        mesh.communicator = MPI_COMM_WORLD;
        const int size = mesh.elements*mesh.equations*mesh.dofs;
        mfem::Vector input(size), output(size + 2), coordinates(14*dimension);
        input.UseDevice();
        output.UseDevice();
        coordinates.UseDevice();
        input = 2.0;
        coordinates = .3;
        const mfem::Vector saved(input);
        mesh.coordinates = coordinates.Read();
        nlohmann::json runtime;
        nlohmann::json configuration = {{"type", "affine"}};
        for (const auto *key : {"constant", "time", "state"})
          {
            configuration[key] = std::vector<double>(mesh.equations, 0.0);
          }
        configuration["gradient"] = std::vector<std::vector<double>>(
          dimension, std::vector<double>(mesh.equations, 0.0));
        for (int q = 0; q < mesh.equations; ++q)
          {
            configuration["constant"][q] = .5 + q;
            configuration["time"][q] = -.4*(q + 1);
            configuration["state"][q] = .02;
            for (int d = 0; d < dimension; ++d)
              {
                configuration["gradient"][d][q] = .1*(d + 1);
              }
          }
        const BuildContext context{dimension, mesh.equations, false, runtime};
        const auto registry = BuiltinRegistry();
        System system;
        system.Add(registry.Create(configuration, context));
        system.Add(registry.Create(configuration, context));
        system.Initialize(mesh);
        for (int call = 0; call < 3; ++call)
          {
            output = .25;
            Stage stage{.2*call, .001, input.Read()};
            system.AddStage(stage, output.ReadWrite() + 1);
            const auto *actual = output.HostRead();
            Near(actual[0], .25);
            Near(actual[size + 1], .25);
            for (int e = 0; e < 2; ++e)
              {
                for (int q = 0; q < mesh.equations; ++q)
                  {
                    const auto expected = .25 + 2.*(.5 + q - .4*(q + 1)*stage.time
                      + .04 + .03*dimension*(dimension + 1)/2.);
                    for (int node = 0; node < mesh.dofs; ++node)
                      {
                        Near(actual[1 + (e*mesh.equations + q)*7 + node], expected);
                      }
                  }
              }
          }
        const auto *input_host = input.HostRead();
        const auto *saved_host = saved.HostRead();
        for (int i = 0; i < size; ++i) { Near(input_host[i], saved_host[i]); }
      }
  }

  void TestConfigurationAndUnusedPath()
  {
    using nlohmann::json;
    Near(CreateConfigured(json::object(), 4, 2, false) == nullptr, 1);
    Near(CreateConfigured(json{{"sources", json::array()}}, 4, 2, false) == nullptr, 1);
    Reject([&] { CreateConfigured(json::parse(R"({"sources":{}})"), 4, 2, false); });
    Reject([&] { CreateConfigured(json::parse(R"({"sources":[{"type":"unknown"}]})"), 4, 2, false); });
    auto mms = json::parse(R"({"sources":[{"type":"periodic-euler-mms"}]})");
    Near(CreateConfigured(mms, 4, 2, false)->Size(), 1);
    Reject([&] { CreateConfigured(mms, 4, 2, true); });
    Reject([&] { CreateConfigured(mms, 5, 3, false); });
    mms["gas_model"] = "lte";
    Reject([&] { CreateConfigured(mms, 4, 2, false); });
    mms["gas_model"] = "cpg";
    mms["gamma"] = 1.3;
    Reject([&] { CreateConfigured(mms, 4, 2, false); });
    auto invalid = json::parse(R"({"sources":[{"type":"affine","constant":[1]}]})");
    Reject([&] { CreateConfigured(invalid, 4, 2, false); });
    invalid["sources"][0]["constant"] = std::vector<double>(4, 0.0);
    invalid["sources"][0]["constant"][0] = std::numeric_limits<double>::infinity();
    Reject([&] { CreateConfigured(invalid, 4, 2, false); });

    // Empty systems accept null arrays and uninitialized stage metadata: no
    // source initialization, state access, model preparation or backend launch.
    System empty;
    Near(empty.Empty(), 1);
    Near(empty.Needs().coordinates, 0);
    Stage unused;
    unused.time = std::numeric_limits<double>::quiet_NaN();
    unused.step_size = unused.time;
    for (int call = 0; call < 5; ++call) { empty.AddStage(unused, nullptr); }

    // A zero-element rank still participates in a source's preparation, but
    // never launches its point evaluator or dereferences its arrays.
    MeshView mesh;
    mesh.elements = 0;
    mesh.dofs = 3;
    mesh.equations = 4;
    mesh.dimension = 2;
    mesh.layout = Theseus::StateLayout(2, 1);
    mesh.communicator = MPI_COMM_WORLD;
    System zero;
    zero.Add(std::make_unique<PointSource<MMSModel>>(MMSModel{}));
    zero.Initialize(mesh);
    zero.AddStage(Stage{}, nullptr);
  }

}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA)) { return 1; }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  TestSources(rank);
  TestDragDimensions();
  TestAffine();
  TestConfigurationAndUnusedPath();
  int total;
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) { std::cout << checks << " checks/rank; failures=" << total << '\n'; }
  return total ? 1 : 0;
}
