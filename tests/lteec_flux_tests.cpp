// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "lteec_flux_fixture.hpp"
using namespace TheseusTest::LTEECFluxFixture;

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA))
    {
      return 1;
    }
  for (int dim = 1; dim <= 3; ++dim)
    {
      for (int order : {2, 4, 7})
        {
          Run(dim, order, false);
          Run(dim, order, true);
        }
    }
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
