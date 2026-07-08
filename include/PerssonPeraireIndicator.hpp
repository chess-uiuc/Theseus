// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "Indicator.hpp"
#include "ModalBasis.hpp"

namespace Prandtl
{

  class PerssonPeraireIndicator : public Prandtl::Indicator
  {
  private:
    std::shared_ptr<Prandtl::ModalBasis> modalBasis;
    mfem::Vector rho_p, modes, modesM1, modesM2;
    mfem::Array2D<int> ubdegs;
    mfem::Array<int> ubdegs_row;
  
  public:
    PerssonPeraireIndicator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
                            std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
                            std::shared_ptr<mfem::ParGridFunction> eta,
                            std::shared_ptr<Prandtl::ModalBasis> modalBasis);
    virtual void CheckIndicatorSmoothness(const mfem::Vector &indicator) override;
  };

}
