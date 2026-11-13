// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceRegistry.hpp"

namespace Theseus::Sources
{
  // Application composition point; adding source types never changes RHS/cache.
  Registry BuiltinRegistry();
  std::unique_ptr<System> CreateConfigured(const nlohmann::json &runtime,
    int equations, int dimension, bool axisymmetric);
}
