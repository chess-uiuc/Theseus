// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "LTEHelmholtzEOS.hpp"
#ifdef THESEUS_STRESS_CHANNEL_CASES
#include "helmholtz_channel_difference_cases.hpp"
namespace MeanReference = ChannelMeanReference;
#else
#include "helmholtz_difference_cases.hpp"
namespace MeanReference = HelmholtzDifferenceReference;
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  const std::string workload = argc > 2 ? argv[2] : "full";
  if ((backend != "cpu" && backend != "cuda") ||
      (workload != "quick" && workload != "full") || argc > 3)
    {
      std::cerr << "Usage: helmholtz_mean_stress_tests [cpu|cuda] [quick|full]\n";
      return 2;
    }
  const bool quick = workload == "quick";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA))
    {
      std::cerr << "CUDA backend was not selected\n";
      return 1;
    }
  namespace HT = Theseus::HelmholtzTable;
  namespace EOS = Theseus::HelmholtzEOS;
  const int cases = sizeof(MeanReference::cases)/sizeof(MeanReference::cases[0]);
  mfem::Vector table(cases*24), coordinates(cases*10), states(cases*4);
  table = 0.0;
  for (int c = 0; c < cases; ++c)
    {
      const auto &input = MeanReference::cases[c];
      for (int j = 0; j < 16; ++j)
        {
          table[24*c + j] = input.jets[j];
        }
      for (int j = 0; j < 4; ++j)
        {
          coordinates[10*c + j] = input.bounds[j];
          states[4*c + j] = input.states[j];
        }
      coordinates[10*c + 4] = -1e9;
      coordinates[10*c + 5] = 1e9;
      for (int j = 6; j < 10; ++j)
        {
          coordinates[10*c + j] = 0.5*(input.bounds[2] + input.bounds[3]);
        }
    }
  // One ordinary host evaluation per case/orientation is an agreement baseline,
  // separate from the independent high-precision accuracy reference.
  std::vector<mfem::real_t> host_means(cases*4);
  for (int c = 0; c < cases; ++c)
    {
      const auto *grid = coordinates.HostRead() + c*10;
      const HT::Tables tables{HT::Layout{2, 2},
        {table.HostRead() + c*24, grid + 6, grid, grid + 2, grid + 4}};
      const auto *state = states.HostRead() + c*4;
      for (int reverse = 0; reverse < 2; ++reverse)
        {
          const auto left = EOS::At(tables, state[reverse ? 2 : 0], state[reverse ? 3 : 1]);
          const auto right = EOS::At(tables, state[reverse ? 0 : 2], state[reverse ? 1 : 3]);
          const auto means = EOS::ThermodynamicMeans(tables, left, right);
          host_means[4*c + 2*reverse] = means.density;
          host_means[4*c + 2*reverse + 1] = means.energy;
        }
    }
  const auto *values = table.Read();
  const auto *grids = coordinates.Read();
  const auto *input = states.Read();
  long long checked = 0;
  long long failures = 0;
  double max_scaled_error = 0;
  double max_agreement_error = 0;
  for (int round = 0; round < (quick ? 2 : 8); ++round)
    {
      for (int count : {1, 127, 128, 129, 4097, 65537, 1048579})
        {
          if (quick && count > 4097)
            {
              continue;
            }
          mfem::Vector output(2*count + 2);
          output = -12345.0;
          auto *out = output.ReadWrite();
          mfem::forall(count, [=] MFEM_HOST_DEVICE (int i)
            {
              const int c = (i*37 + round)%cases;
              const auto *grid = grids + c*10;
              const HT::Tables tables{HT::Layout{2, 2}, {values + c*24, grid + 6, grid, grid + 2, grid + 4}};
              const auto *s = input + c*4;
              const bool reverse = (i & 1) != 0;
              const auto left = EOS::At(tables, s[reverse ? 2 : 0], s[reverse ? 3 : 1]);
              const auto right = EOS::At(tables, s[reverse ? 0 : 2], s[reverse ? 1 : 3]);
              const auto means = EOS::ThermodynamicMeans(tables, left, right);
              out[1 + 2*i] = means.density;
              out[2 + 2*i] = means.energy;
            });
          const auto *actual = output.HostRead();
          failures += actual[0] != -12345.0 || actual[2*count + 1] != -12345.0;
          checked += 2;
          for (int i = 0; i < count; ++i)
            {
              const int c = (i*37 + round)%cases;
              for (int q = 0; q < 2; ++q)
                {
                  const auto expected = MeanReference::cases[c].expected[q];
                  const auto error = std::abs(actual[1 + 2*i + q] - expected)/std::max(1.0, std::abs(expected));
                  const auto bound = 4096*std::numeric_limits<mfem::real_t>::epsilon();
                  if (!std::isfinite(error) || error > bound)
                    {
                      ++failures;
                      if (failures < 16)
                        {
                          std::cerr << "Mean reference mismatch: case=" << c << ", component=" << q
                                    << ", value=" << actual[1 + 2*i + q]
                                    << ", expected=" << expected << '\n';
                        }
                    }
                  const auto host_value = host_means[4*c + 2*(i & 1) + q];
                  const auto agreement_error = std::abs(actual[1 + 2*i + q] - host_value)
                                               /std::max(mfem::real_t(1), std::abs(host_value));
                  if (!std::isfinite(agreement_error) || agreement_error > bound)
                    {
                      ++failures;
                      if (failures < 16)
                        {
                          std::cerr << "Host/backend mean mismatch: case=" << c
                                    << ", component=" << q << ", error=" << agreement_error << '\n';
                        }
                    }
                  max_agreement_error = std::max(max_agreement_error, agreement_error);
                  ++checked;
                  max_scaled_error = std::max(max_scaled_error, error);
                  ++checked;
                }
            }
        }
    }
  std::cout << checked << " 80-digit-reference/host-agreement/guard checks, " << failures
            << " failures; max scaled error=" << max_scaled_error << "; max host/backend scaled difference=" << max_agreement_error << "; backend=" << backend << '\n';
  return failures ? 1 : 0;
}
