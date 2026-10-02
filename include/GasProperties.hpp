// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"

namespace Theseus
{

  struct GasPropertyRequest
  {
    bool pressure = false;
    bool temperature = false;
    bool sound_speed = false;
    bool viscosity = false;
    bool thermal_conductivity = false;

    bool Empty() const
    {
      return !pressure && !temperature && !sound_speed &&
             !viscosity && !thermal_conductivity;
    }
  };

  // Only requested properties are populated. Unrequested values remain zero.
  struct GasProperties
  {
    mfem::real_t pressure = 0;
    mfem::real_t temperature = 0;
    mfem::real_t sound_speed = 0;
    mfem::real_t viscosity = 0;
    mfem::real_t thermal_conductivity = 0;
  };

}
