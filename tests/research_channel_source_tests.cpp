#include "sources/ChannelFeedback.hpp"
#include "SourceConfiguration.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
  int checks = 0, failures = 0;
  void Near(double actual, double expected)
  {
    ++checks;
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > 2e-12*(1.0 + std::abs(expected)))
      {
        if (++failures < 10) { std::cerr << actual << " expected " << expected << '\n'; }
      }
  }
  template<typename Function> void Reject(Function action)
  {
    ++checks;
    try { action(); ++failures; }
    catch (const std::exception &) {}
  }

  void CheckController(int rank)
  {
    using namespace Theseus::Sources;
    const nlohmann::json runtime = nlohmann::json::object();
    const BuildContext context{3, 5, false, runtime};
    for (int base : {0, 1, 257, 65537})
      {
        MeshView mesh;
        mesh.elements = base == 0 ? rank : base + rank;
        mesh.dofs = 7;
        mesh.dimension = 3;
        mesh.equations = 5;
        mesh.layout = Theseus::StateLayout(3, 1);
        mesh.communicator = MPI_COMM_WORLD;
        const int size = mesh.elements*35;
        mfem::Vector input(size), weights(mesh.elements*7), output(size + 2);
        for (auto *v : {&input, &weights, &output}) { v->UseDevice(); }
        auto *u = input.HostWrite();
        auto *w = weights.HostWrite();
        double local[4] = {}, compensation[4] = {};
        auto accumulate = [&](int component, double value)
        {
          const auto corrected = value - compensation[component];
          const auto sum = local[component] + corrected;
          compensation[component] = (sum - local[component]) - corrected;
          local[component] = sum;
        };
        for (int e = 0; e < mesh.elements; ++e)
          {
            for (int point = 0; point < 7; ++point)
              {
                const auto rho = 1.0 + .1*rank + .001*(e % 5) + .005*point;
                const auto ux = 5.0 + rank + point;
                const auto internal = 1000.0 + 2*point;
                const auto weight = 1.0 + .1*rank + .01*point;
                const int offset = e*35 + point;
                u[offset] = rho;
                u[offset + 7] = rho*ux;
                u[offset + 14] = -.2*rho;
                u[offset + 21] = .3*rho;
                u[offset + 28] = rho*(internal + .5*(ux*ux + .04 + .09));
                w[e*7 + point] = weight;
                accumulate(0, rho*weight);
                accumulate(1, rho*ux*weight);
                accumulate(2, rho*internal*weight);
                accumulate(3, weight);
              }
          }
        double global[4];
        MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, mesh.communicator);
        if (global[3] == 0.0) { continue; }
        mesh.weights = weights.Read();
        const mfem::Vector saved(input);
        ChannelFeedbackModel model({{"target_density", 1.2}, {"target_streamwise_momentum", 12.0}}, context);
        model.Initialize(mesh);
        Near(model.Volume(), global[3]);
        for (int call = 0; call < 3; ++call)
          {
            if (call > 0) { input *= 1.0 + .01*call; }
            const auto scale = call == 0 ? 1.0 : (call == 1 ? 1.01 : 1.01*1.02);
            const Stage stage{.17*call, call == 1 ? .002 : .001, input.Read()};
            model.Prepare(stage);
            Near(model.Averages().density, scale*global[0]/global[3]);
            Near(model.Averages().streamwise_momentum, scale*global[1]/global[3]);
            Near(model.Averages().internal_energy_density, scale*global[2]/global[3]);
            const auto density_source = (1.2 - scale*global[0]/global[3])/(.7*stage.step_size);
            const auto momentum_source = (12.0 - scale*global[1]/global[3])/(.7*stage.step_size);
            const auto source_function = model.MakeEvaluator(stage);
            Near(source_function.density_source, density_source);
            Near(source_function.streamwise_momentum_source, momentum_source);
            output = .25;
            auto *rhs = output.ReadWrite() + 1;
            const DeviceMesh device_mesh{mesh.elements, 7, 5, 3, mesh.layout, nullptr, nullptr, nullptr};
            mfem::forall(mesh.elements*7, [=] MFEM_HOST_DEVICE (int i)
            {
              const int element = i/7;
              const int node = i % 7;
              const Point point{device_mesh, stage, element, node};
              const SourceAccumulator source{rhs + element*35, 7, node};
              source_function(point, source);
            });
            const auto *actual = output.HostRead();
            Near(actual[0], .25);
            Near(actual[size + 1], .25);
            for (int e = 0; e < mesh.elements; ++e)
              {
                for (int point = 0; point < 7; ++point)
                  {
                    const int offset = 1 + e*35 + point;
                    Near(actual[offset], .25 + density_source);
                    Near(actual[offset + 7], .25 + momentum_source);
                    Near(actual[offset + 14], .25);
                    Near(actual[offset + 21], .25);
                    Near(actual[offset + 28], .25 + (5.0 + rank + point)*momentum_source);
                  }
              }
            const auto *input_host = input.HostRead();
            const auto *saved_host = saved.HostRead();
            for (int i = 0; i < size; ++i) { Near(input_host[i], saved_host[i]*scale); }
          }
        const Stage stage{.123, .001, input.Read()};
        ChannelFeedbackModel equilibrium({{"target_density", model.Averages().density},
          {"target_streamwise_momentum", model.Averages().streamwise_momentum}}, context);
        equilibrium.Initialize(mesh);
        equilibrium.Prepare(stage);
        Near(equilibrium.MakeEvaluator(stage).density_source, 0.0);
        Near(equilibrium.MakeEvaluator(stage).streamwise_momentum_source, 0.0);
        ChannelFeedbackModel defaults(nlohmann::json::object(), context);
        defaults.Initialize(mesh);
        defaults.Prepare(stage);
        Near(defaults.MakeEvaluator(stage).density_source, (1.0 - model.Averages().density)/(.7*.001));
        Near(defaults.MakeEvaluator(stage).streamwise_momentum_source,
          (515.68342221905391 - model.Averages().streamwise_momentum)/(.7*.001));
        Reject([&] { model.Prepare(Stage{0., 0., input.Read()}); });
        Reject([&] { model.Prepare(Stage{0., std::numeric_limits<double>::quiet_NaN(), input.Read()}); });
      }
  }

  void CheckConfiguration()
  {
    using namespace Theseus::Sources;
    using nlohmann::json;
    auto runtime = json::parse(R"({"conditions":{"initial_conditions":{"function":"StreamwiseParabolicIC"}}})");
    Near(CreateConfigured(runtime, 5, 3, false)->Size(), 1);
    runtime["channel_forcing"] = false;
    Near(CreateConfigured(runtime, 5, 3, false) == nullptr, 1);
    runtime["channel_forcing"] = true;
    runtime["sources"] = json::array();
    Near(CreateConfigured(runtime, 5, 3, false) == nullptr, 1);
    runtime["channel_forcing"] = false;
    runtime["sources"] = json::array({{{"type", "channel-feedback"}}});
    Near(CreateConfigured(runtime, 5, 3, false)->Size(), 1);
    Reject([&] { CreateConfigured(runtime, 4, 2, false); });
    Reject([&] { CreateConfigured(runtime, 5, 3, true); });
    runtime["sources"][0]["relaxation_factor"] = 0.;
    Reject([&] { CreateConfigured(runtime, 5, 3, false); });
    runtime["sources"][0]["relaxation_factor"] = .7;
    runtime["sources"][0]["target_density"] = -1.;
    Reject([&] { CreateConfigured(runtime, 5, 3, false); });
  }
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA)) { return 1; }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CheckController(rank);
  CheckConfiguration();
  int total = 0;
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) { std::cout << checks << " channel checks per rank, failures=" << total << '\n'; }
  return total ? 1 : 0;
}
