// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

#include "mfem.hpp"

namespace Theseus
{
  inline mfem::real_t TimeEventTolerance(const mfem::real_t time)
  {
    return 16.0 * std::numeric_limits<mfem::real_t>::epsilon()
      * std::max(mfem::real_t(1.0), std::abs(time));
  }

  inline bool TimeEventReached(const mfem::real_t time,
                               const mfem::real_t event_time)
  {
    return time + TimeEventTolerance(time) >= event_time;
  }

  inline mfem::real_t TimeToNextEvent(
    const mfem::real_t time,
    const std::initializer_list<mfem::real_t> event_times)
  {
    mfem::real_t result = mfem::infinity();
    const mfem::real_t tolerance = TimeEventTolerance(time);
    for (const mfem::real_t event_time : event_times)
      {
        const mfem::real_t delta = event_time - time;
        if (std::isfinite(event_time) && delta > tolerance)
          result = std::min(result, delta);
      }
    return result;
  }

  inline mfem::real_t LimitTimeStepToEvents(
    const mfem::real_t nominal_step,
    const mfem::real_t time,
    const std::initializer_list<mfem::real_t> event_times)
  {
    return std::min(nominal_step, TimeToNextEvent(time, event_times));
  }
}
