// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceRegistry.hpp"

namespace Theseus::Sources
{
inline mfem::real_t
ManufacturedMeanVelocityScale (const nlohmann::json &configuration)
{
  const auto regime = configuration.value ("regime", std::string ("transonic"));
  if (regime == "subsonic")
    {
      return .25;
    }
  if (regime == "transonic")
    {
      return 1.;
    }
  if (regime == "supersonic")
    {
      return 5.;
    }
  throw std::invalid_argument ("MMS regime must be subsonic, transonic or supersonic");
}
}
