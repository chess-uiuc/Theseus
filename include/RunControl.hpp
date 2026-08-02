// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace Theseus
{

  inline bool StepLimitReached(int cycle, int maximum_cycle)
  {
    return maximum_cycle > 0 && cycle >= maximum_cycle;
  }

}
