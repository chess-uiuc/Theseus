// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "theseus_kernels.hpp"

namespace Theseus
{

  struct PhysicsConstants
  {
    mfem::real_t gamma;
    mfem::real_t gammaInverse;
    mfem::real_t gammaP1;
    mfem::real_t gammaM1;
    mfem::real_t gammaP1Inverse;
    mfem::real_t gammaM1Inverse;
    mfem::real_t gamma_gammaM1Inverse; // gamma * gammaM1Inverse;
    mfem::real_t gammaM1_gammaInverse; // gammaM1 * gammaInverse;

    mfem::real_t Pr;
    mfem::real_t PrInverse;
    mfem::real_t R_gas;
    mfem::real_t cp;
    mfem::real_t mu;

    MFEM_HOST_DEVICE PhysicsConstants() = default;

    PhysicsConstants(mfem::real_t gamma, mfem::real_t Pr, mfem::real_t R_gas, mfem::real_t mu)
      : gamma(gamma), Pr(Pr), R_gas(R_gas), mu(mu),
        gammaInverse(1.0 / gamma), gammaM1(gamma - 1.0), gammaP1(gamma + 1.0),
        gammaM1Inverse(1.0 / gammaM1), gammaP1Inverse(1.0 / gammaP1),
        gammaM1_gammaInverse(gammaM1 * gammaInverse), gamma_gammaM1Inverse(gamma * gammaM1Inverse),
        PrInverse(1.0 / Pr), cp(gamma_gammaM1Inverse * R_gas) {}

    mfem::real_t mu_bulk = 2.0 / 3.0;
    mfem::real_t mu0 = 1.716e-5;
    mfem::real_t T0 = 273.15;
    mfem::real_t Ts = 110.4;
  };

}
