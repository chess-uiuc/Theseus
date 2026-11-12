#include "Source.hpp"
#include "PirozzoliFlux.hpp"
#include <cmath>
#include <iostream>

namespace
{
  int failures = 0, checks = 0;
  void Near(mfem::real_t actual, mfem::real_t expected)
  {
    ++checks;
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > 2e-12*(1 + std::abs(expected)))
      {
        if (++failures < 10) std::cerr << actual << " expected " << expected << '\n';
      }
  }
  struct Gas
  {
    Theseus::StateLayout L{3, 1};
    int dim() const { return 3; }
  };
  struct Context
  {
    int num_elements = 2, ndof_scalar_el = 7, num_equations = 5, dim = 3;
    Gas gas;
    const mfem::real_t *elQWgts_d = nullptr;
    bool use_target_state_source = true;
    const mfem::real_t *target_state_d = nullptr, *volume_avg_state_d = nullptr;
    mfem::real_t tau_inv = 0, total_volume = 0;
  };
  struct Owner
  {
    mfem::Vector volume_avg_state = mfem::Vector(5);
    mfem::real_t current_dt = 1e-3;
  };
}
int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA)) return 1;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Context dc;
  Owner owner;
  owner.volume_avg_state = 0.0;
  mfem::Vector input(70), weights(14), target(5), result(72);
  for (auto *v : {&input, &weights, &target, &result}) v->UseDevice();
  mfem::real_t local[4] = {};
  for (int element = 0; element < 2; ++element)
    for (int point = 0; point < 7; ++point)
      {
        const int offset = element*35 + point;
        const auto rho = 1.0 + .1*rank + .02*element + .005*point;
        const auto ux = 5.0 + rank + point;
        const auto uy = -.2, uz = .3;
        const auto e = 1000.0 + 2*point;
        input.HostReadWrite()[offset] = rho;
        input.HostReadWrite()[offset + 7] = rho*ux;
        input.HostReadWrite()[offset + 14] = rho*uy;
        input.HostReadWrite()[offset + 21] = rho*uz;
        input.HostReadWrite()[offset + 28] = rho*(e + .5*(ux*ux + uy*uy + uz*uz));
        const auto w = 1.0 + .1*rank + .01*point;
        weights.HostReadWrite()[element*7 + point] = w;
        local[0] += rho*w; local[1] += rho*ux*w; local[2] += rho*e*w; local[3] += w;
      }
  mfem::real_t global[4] = {};
  MPI_Allreduce(local, global, 4, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
  dc.total_volume = global[3];
  dc.elQWgts_d = weights.Read();
  target = 0.0; target.HostReadWrite()[0] = 1.2; target.HostReadWrite()[1] = 12;
  dc.target_state_d = target.Read();
  const auto *u = input.Read();
  const mfem::Vector saved(input);
  for (int scenario = 0; scenario < 4; ++scenario)
    {
      owner.current_dt = scenario == 1 ? 2e-3 : 1e-3;
      dc.use_target_state_source = scenario != 3;
      Theseus::Source::PreProcessSourceTerms(dc, owner, u, MPI_COMM_WORLD);
      if (scenario != 3)
        {
          const auto *mean = owner.volume_avg_state.HostRead();
          Near(mean[0], global[0]/global[3]);
          Near(mean[1], global[1]/global[3]);
          Near(mean[4], global[2]/global[3]);
          Near(dc.tau_inv, 1/(.7*owner.current_dt));
        }
      if (scenario == 2)
        {
          target.HostReadWrite()[0] = owner.volume_avg_state.HostRead()[0];
          target.HostReadWrite()[1] = owner.volume_avg_state.HostRead()[1];
          dc.target_state_d = target.Read();
        }
      result = .25; result.HostReadWrite()[0] = -12345; result.HostReadWrite()[71] = -12345;
      auto *out = result.ReadWrite() + 1;
      mfem::forall(2, [=] MFEM_HOST_DEVICE (int element)
      {
        Theseus::Source::AddSourceTerms(dc, u + element*35, out + element*35);
      });
      const auto *actual = result.HostRead() + 1;
      const auto rho_source = scenario >= 2 ? 0 : (1.2 - global[0]/global[3])/(.7*owner.current_dt);
      const auto mom_source = scenario >= 2 ? 0 : (12 - global[1]/global[3])/(.7*owner.current_dt);
      for (int element = 0; element < 2; ++element)
        for (int point = 0; point < 7; ++point)
          {
            const int offset = element*35 + point;
            Near(actual[offset], .25 + rho_source);
            Near(actual[offset + 7], .25 + mom_source);
            Near(actual[offset + 14], .25); Near(actual[offset + 21], .25);
            Near(actual[offset + 28], .25 + (5.0 + rank + point)*mom_source);
          }
      Near(result.HostRead()[0], -12345); Near(result.HostRead()[71], -12345);
    }
  // Host verification must synchronize both vectors before accessing their data.
  const auto *input_host = input.HostRead();
  const auto *saved_host = saved.HostRead();
  for (int i = 0; i < input.Size(); ++i) Near(input_host[i], saved_host[i]);
  int all_failures;
  MPI_Allreduce(&failures, &all_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) std::cout << checks << " source checks per rank; failures=" << all_failures << '\n';
  return all_failures ? 1 : 0;
}
