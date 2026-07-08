// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "ModalBasis.hpp"

namespace Prandtl
{
  ModalBasis::ModalBasis(mfem::DG_FECollection &fec_, mfem::Geometry::Type &gtype_, int order_, int dim_)
    : order(order_), dim(dim_)
  {
    // Transformation requires nodal basis
    mfem::BasisType::CheckNodal(fec_.GetBasisType());
   
    x = new mfem::real_t[dim];
    L = new mfem::real_t[order + 1];
    Li = new mfem::real_t[order + 1];
    Di = new mfem::real_t[order + 1];

    mfem::IntegrationRule solpts(fec_.FiniteElementForGeometry(gtype_)->GetNodes());
    npts = solpts.GetNPoints();
    umc = mfem::Vector(npts);

    ComputeUBDegs(gtype_);
    ComputeVDM(solpts);
  }

  ModalBasis::~ModalBasis()
  {
    delete[] x;
    delete[] L;
    delete[] Li;
    delete[] Di;
  }

  mfem::Array2D<int> ModalBasis::GetPolyDegs()
  {
    return ubdegs;
  }

  const mfem::DenseMatrix& ModalBasis::GetVandermonde()
  {
    return V;
  }

  void ModalBasis::ComputeUBDegs(mfem::Geometry::Type &gtype)
  {
    ubdegs = mfem::Array2D<int>(npts, dim);
    switch (gtype)
      {
        // Segment: [0,1]
      case 1:
        {
          for (int i = 0; i < order + 1; i++)
            {
              ubdegs(i, 0) = i;
            }
          break;
        }
        // Triangle with vertices (0,0), (1,0), (0,1)
      case 2:
        {
          int n = 0;
          for (int i = 0; i < order + 1; i++)
            {
              for (int j = 0; j < order + 1 - i; j++)
                {
                  ubdegs(n, 0) = i;
                  ubdegs(n, 1) = j;
                  n++;
                }
            }
          break;
        }
        // Quad: [0,1]^2
      case 3:
        {
          int n = 0;
          for (int i = 0; i < order + 1; i++)
            {
              for (int j = 0; j < order + 1; j++)
                {
                  ubdegs(n, 0) = i;
                  ubdegs(n, 1) = j;
                  n++;
                }
            }
          break;
        }
        // Hex: [0, 1]^3
      case 5:
        {
          int n = 0;
          for (int i = 0; i < order + 1; i++)
            {
              for (int j = 0; j < order + 1; j++)
                {
                  for (int k = 0; k < order + 1; k++)
                    {
                      ubdegs(n, 0) = i;
                      ubdegs(n, 1) = j;
                      ubdegs(n, 2) = k;
                      n++;
                    }
                }
            }
          break;
        }
      default:
        MFEM_ABORT("Element type not currently supported for modal basis.")
          }
  }

  void ModalBasis::ComputeVDM(mfem::IntegrationRule &solpts)
  {
    V = mfem::DenseMatrix(npts);
    // Loop through solution nodes
    for (int i = 0; i < npts; i++)
      {
        // Compute nodal location in reference space
        solpts.IntPoint(i).Get(x, dim);

        // Compute L_k(x_i)*L_k(y_i)*L_k(z_i) for each modal basis function k corresponding to the
        // polynomial degree in ubdegs.
        for (int j = 0; j < dim; j++)
          {
            mfem::Poly_1D::CalcLegendre(order, x[j], L);
            for (int k = 0; k < npts; k++)
              {
                double v = L[ubdegs(k, j)];
                V(i,k) = (j == 0) ? v : V(i,k)*v;
              }
          }
      }

    // Invert and store Vandermonde matrix
    V_inv = V;
    V_inv.Invert();
  }

  void ModalBasis::ComputeModes(const mfem::Vector &nodes)
  {
    MFEM_ASSERT(nodes.Size() == npts,
                "Element-wise solution vector must be of same size as the modal basis.")

      V_inv.Mult(nodes, umc);
  }

  void ModalBasis::ComputeModes(const mfem::Vector &nodes, mfem::Vector &modes)
  {
    MFEM_ASSERT(nodes.Size() == npts,
                "Element-wise solution vector must be of same size as the modal basis.")
      V_inv.Mult(nodes, modes); 
  }

  void ModalBasis::SetModes(const mfem::Vector& modes)
  {
    umc = modes;
  }

  void ModalBasis::GetModes(mfem::Vector &modes)
  {
    modes = umc;
  }

  void ModalBasis::ComputeNodes(mfem::Vector &nodes)
  {
    MFEM_ASSERT(nodes.Size() == npts,
                "Element-wise solution vector must be of same size as the modal basis.")

      V.Mult(umc, nodes);  
  }

  void ModalBasis::ComputeNodes(const mfem::Vector& modes, mfem::Vector& nodes)
  {
    V.Mult(modes, nodes);
  }

  // Evalutes solution at arbitrary point x using modal basis
  mfem::real_t ModalBasis::Eval(mfem::Vector &x)
  {
    MFEM_ASSERT(x.Size() == dim,
                "Modal basis can only be evaluated at one location at a time.")

      // Pre-compute L_i(x), L_i(y), L_i(z) for all degrees up to max polynomial order.
      mfem::Array2D<double> L(order + 1, dim);
    for (int i = 0; i < dim; i++)
      {
        mfem::Poly_1D::CalcLegendre(order, x(i), Li);
        for (int j = 0; j < order + 1; j++)
          {
            L(j, i) = Li[j];
          }
      }

    // Compute u(x,y,z) as \sum L_i(x)*L_i(y)*L_i(z)
    double ux = 0;
    for (int i = 0; i < npts; i++)
      {
        double v = umc(i);
        for (int j = 0; j < dim; j++)
          {
            v *= L(ubdegs(i, j), j);
          }
        ux += v;
      }

    return ux;
  }

  mfem::DenseMatrix ModalBasis::ComputeVDM(const mfem::IntegrationRule *ir)
  {
    mfem::DenseMatrix VDM(ir->GetNPoints(), npts);
    int indx = 0;
    for (const mfem::IntegrationPoint *ip = ir->begin(); ip != ir->end(); ++ip, ++indx)
      {
        ip->Get(x, dim);
        for (int j = 0; j < dim; j++)
          {
            mfem::Poly_1D::CalcLegendre(order, x[j], L);

            for (int k = 0; k < npts; k++)
              {
                double v = L[ubdegs(k, j)];

                VDM(indx,k) = (j == 0) ? v : VDM(indx,k) * v;
              }
          }
      }
    return VDM;
  }

  // Evalutes solution gradient at arbitrary point x using modal basis
  mfem::Vector ModalBasis::EvalGrad(mfem::Vector &x)
  {
    MFEM_ASSERT(x.Size() == dim,
                "Modal basis can only be evaluated at one location at a time.")

      // Pre-compute L_i(x), L_i(y), L_i(z) (and dL_i(x)/dx, etc.) for all degrees up to max polynomial order.
      mfem::Array2D<double> L(order + 1, dim);
    mfem::Array2D<double> D(order + 1, dim);
    for (int i = 0; i < dim; i++)
      {
        mfem::Poly_1D::CalcLegendre(order, x(i), Li, Di);
        for (int j = 0; j < order + 1; j++)
          {
            L(j, i) = Li[j];
            D(j, i) = Di[j];
          }
      }

    // Compute du(x,y,z)/dx as \sum dL_i(x)/dx*L_i(y)*L_i(z)
    //         du(x,y,z)/dy as \sum L_i(x)*dL_i(y)/dy*L_i(z)
    //         du(x,y,z)/dz as \sum L_i(x)*L_i(y)*dL_i(z)/dz
    mfem::Vector gradu(dim);
    for (int d = 0; d < dim; d++)
      {
        double du = 0;
        for (int i = 0; i < npts; i++)
          {
            double v = umc(i);
            for (int j = 0; j < dim; j++)
              {
                v *= j == d ? D(ubdegs(i, j), j) : L(ubdegs(i, j), j);
              }
            du += v;
          }
        gradu(d) = du;
      }

    return gradu;
  }
}
