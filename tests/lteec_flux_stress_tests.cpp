// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "lteec_flux_fixture.hpp"
using namespace TheseusTest::LTEECFluxFixture;

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  const std::string workload = argc > 2 ? argv[2] : "full";
  if ((backend != "cpu" && backend != "cuda") ||
      (workload != "quick" && workload != "full") || argc > 3)
    {
      std::cerr << "Usage: lteec_flux_stress_tests [cpu|cuda] [quick|full]\n";
      return 2;
    }
  const bool quick = workload == "quick";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA))
    {
      std::cerr << "CUDA backend was not selected\n";
      return 1;
    }
  for (int repetition = 0; repetition < (quick ? 1 : 4); ++repetition)
    {
      for (int dim = 1; dim <= 3; ++dim)
        {
          for (int order : {2, 7, 15, 31, 63, 127})
            {
              if (quick && order > 31)
                {
                  continue;
                }
              Run(dim, order, false, true);
              Run(dim, order, true, true);
            }
        }
    }
  std::cout << checks << " analytic flux/helper/assembly/host-agreement/guard checks, " << failures
            << " failures; max host/backend scaled difference=" << max_host_scaled_difference
            << "; backend=" << backend << '\n';
  return failures ? 1 : 0;
}
