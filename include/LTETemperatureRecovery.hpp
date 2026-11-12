// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "theseus_kernels.hpp"

namespace Theseus
{
namespace detail
{
// Shared numerical iteration only. Callers retain their lookup/error policy:
// host startup guards table coordinates; runtime uses device-ready table views.
struct LTETemperatureRecovery
{
  mfem::real_t temperature;
  mfem::real_t residual = 1;
  int iterations = 0;

  MFEM_HOST_DEVICE explicit LTETemperatureRecovery(mfem::real_t initial)
    : temperature(initial) {}
  MFEM_HOST_DEVICE bool NeedsIteration() const { return residual > 1e-12; }
  MFEM_HOST_DEVICE bool ExceededIterations() const { return iterations > 100; }
  MFEM_HOST_DEVICE void Update(mfem::real_t energy, mfem::real_t guess,
                               mfem::real_t cv)
  {
    const auto step = (energy - guess) / cv;
    temperature = temperature + step;
    residual = Kernels::rabs(step) / temperature;
    ++iterations;
  }
};
}
}
