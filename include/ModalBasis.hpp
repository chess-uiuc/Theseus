// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"

namespace Prandtl
{

  class ModalBasis
  {
  private:
    int dim;
    mfem::Array2D<int> ubdegs; // Array of modal basis degrees along each dimension
    mfem::Vector umc; // Vector of solution modal basis coefficients
    mfem::DenseMatrix V, V_inv; // (Inverse) Vandermonde matrix
    mfem::real_t *x, *L, *Li, *Di;

    void ComputeUBDegs(mfem::Geometry::Type &gtype);
    void ComputeVDM(mfem::IntegrationRule &solpts);

  public:
    int order, npts;

    ModalBasis(mfem::DG_FECollection &fec, mfem::Geometry::Type &gtype, int order, int dim);
    ~ModalBasis();

    void ComputeModes(const mfem::Vector &nodes);
    void ComputeModes(const mfem::Vector &nodes, mfem::Vector &modes);
    void SetModes(const mfem::Vector &modes);
    void GetModes(mfem::Vector &modes);
    mfem::real_t Eval(mfem::Vector &x);
    mfem::DenseMatrix ComputeVDM(const mfem::IntegrationRule *ir);
    mfem::Vector EvalGrad(mfem::Vector &x);
    void ComputeNodes(mfem::Vector &nodes);
    void ComputeNodes(const mfem::Vector& modes, mfem::Vector& nodes);
    const mfem::DenseMatrix& GetVandermonde();
    mfem::Array2D<int> GetPolyDegs();
  };

}
