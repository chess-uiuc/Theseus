// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "LTEEOS.hpp"
#include <cstdlib>
#include <iostream>
#ifdef MFEM_USE_CUDA
#include <cuda_runtime.h>
#endif

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
__global__ void AbortPressureInversionKernel()
{
  Theseus::detail::AbortLTEPressureInversion();
}
#endif

int main(int argc, char **argv)
{
  // Do not join an MPI world: intentional failure must stay in this child.
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA)) { return 2; }
  std::cout << "INVOKING_LTE_PRESSURE_ABORT backend=" << backend << std::endl;
  if (backend == "cpu")
    {
      Theseus::detail::AbortLTEPressureInversion();
    }
  else if (backend == "cuda")
    {
#ifdef MFEM_USE_CUDA
      AbortPressureInversionKernel<<<1, 1>>>();
      const auto status = cudaDeviceSynchronize();
      if (status != cudaSuccess)
        {
          std::cout << "DEVICE_ABORT_OBSERVED: " << cudaGetErrorString(status) << std::endl;
          // A deliberate trap invalidates this child's device context. Exit
          // without invoking automatic destructors against that failed context.
          std::_Exit(1);
        }
#endif
    }
  // The failure path must never return normally.
  return 0;
}
