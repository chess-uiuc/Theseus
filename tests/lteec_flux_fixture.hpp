// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "LTEGasModel.hpp"
#include "LTEECFlux.hpp"
#include "DGSEMIntegrator.hpp"
#include "helmholtz_fixture.hpp"
#include <cmath>
#include <iostream>
#include <string>

namespace TheseusTest::LTEECFluxFixture
{
  using namespace Theseus;
  using real = mfem::real_t;
  inline int checks = 0;
  inline int failures = 0;
  inline real max_host_scaled_difference = 0;

  inline void Near(real value, real expected, const char *name)
  {
    ++checks;
    if (!std::isfinite(value) || std::abs(value - expected) > 2e-10*std::max(real(1), std::abs(expected)))
      {
        ++failures;
        if (failures < 16)
          {
            std::cerr << name << ": " << value << " expected " << expected << '\n';
          }
      }
  }

  struct Context
  {
    int dim;
    int num_equations;
    int num_face_points;
    bool axisymmetric = false;
    LTEHelmholtzGas gas;
    LTEECFlux::InviscidFlux iflux;
    MFEM_HOST_DEVICE int iface_idx(int side, int point, int equation) const
    {
      return (side*num_equations + equation)*num_face_points + point;
    }
  };

  MFEM_HOST_DEVICE void State(int dim, real r, real t, real velocity_shift, real *q)
  {
    q[0] = r;
    real kinetic = 0;
    for (int d = 0; d < dim; ++d)
      {
        const real u = 0.5*(d + 1) + velocity_shift;
        q[d + 1] = r*u;
        kinetic += 0.5*u*u;
      }
    q[dim + 1] = r*(2*r + r*r/2 + t*t + kinetic);
  }

  // Independent closed-form divided differences for H=2r+r²/2+rT/4-T².
  // No production interpolation/EOS/discrete-gradient routine supplies this oracle.
  inline void Oracle(int dim, real r1, real t1, real r2, real t2,
              const real *q1, const real *q2, const real *normal, bool face, real *out)
  {
    const auto b1 = 1/t1;
    const auto b2 = 1/t2;
    const auto dr = r2 - r1;
    const auto db = b2 - b1;
    const bool diff_r = std::abs(dr) >= 2.22e-15 + 1.49e-8*std::max(r1,r2);
    const bool diff_b = std::abs(db) >= 2.22e-15 + 1.49e-8*std::max(b1,b2);
    const auto pr = diff_r
      ? 0.5*(b1 + b2)*(2*(r1 + r2) + r1*r1 + r1*r2 + r2*r2) + (r1 + r2)/4
      : 0.5*((4*r1 + 3*r1*r1 + r1*t1/2)*b1 + (4*r2 + 3*r2*r2 + r2*t2/2)*b2);
    const auto gr = diff_r
      ? 0.5*(b1 + b2)*(4 + 1.5*(r1 + r2)) + 0.5
      : 0.5*((4 + 3*r1 + t1/2)*b1 + (4 + 3*r2 + t2/2)*b2);
    const auto pb = 0.5*(2*r1*r1 + r1*r1*r1 + 2*r2*r2 + r2*r2*r2);
    const auto gb = 0.5*(4*r1 + 1.5*r1*r1 + 4*r2 + 1.5*r2*r2)
                    + (diff_b ? t1*t2 : 0.5*(t1*t1 + t2*t2));
    const auto rho = pr/gr;
    const auto energy = gb - pb/rho;
    const auto p1 = r1*r1*(2 + r1 + t1/4);
    const auto p2 = r2*r2*(2 + r2 + t2/4);
    real un = 0, u1n = 0, u2n = 0, kinetic = 0, work = 0, norm2 = 0;
    for (int d = 0; d < dim; ++d)
      {
        const auto u1 = q1[d + 1]/r1;
        const auto u2 = q2[d + 1]/r2;
        un += 0.5*(u1 + u2)*normal[d];
        u1n += u1*normal[d];
        u2n += u2*normal[d];
        kinetic += 0.5*u1*u2;
        work += 0.5*(p1*u2 + p2*u1)*normal[d];
        norm2 += normal[d]*normal[d];
      }
    const auto c1 = std::sqrt(4*r1 + 3*r1*r1 + r1*t1/2 + r1*r1/32);
    const auto c2 = std::sqrt(4*r2 + 3*r2*r2 + r2*t2/2 + r2*r2/32);
    const auto lambda = face ? 0.5*std::max(std::abs(u1n) + c1*std::sqrt(norm2),
                                           std::abs(u2n) + c2*std::sqrt(norm2)) : 0;
    out[0] = rho*un - lambda*(q2[0] - q1[0]);
    for (int d = 0; d < dim; ++d)
      {
        out[d + 1] = rho*un*0.5*(q1[d + 1]/r1 + q2[d + 1]/r2)
                     + 0.5*(p1 + p2)*normal[d] - lambda*(q2[d + 1] - q1[d + 1]);
      }
    out[dim + 1] = rho*un*(energy + kinetic) + work - lambda*(q2[dim + 1] - q1[dim + 1]);
  }

  inline void HostAgreement(real actual, real expected, const char *name)
  {
    ++checks;
    const auto scale = std::max(real(1), std::abs(expected));
    const auto tolerance = 1e-10*scale;
    max_host_scaled_difference = std::max(max_host_scaled_difference, std::abs(actual - expected)/scale);
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
      {
        ++failures;
        if (failures < 16)
          {
            std::cerr << name << ": " << actual << " expected " << expected << '\n';
          }
      }
  }

  inline void Run(int dim, int order, bool point_parallel, bool compare_host = false)
  {
    HelmholtzTable::Data data;
    const auto tables = TheseusTest::AnalyticHelmholtz(data);
    PhysicsConstants physics(1.4, 0.72, 287, 0.02, 1);
    physics.T0 = 1;
    physics.Ts = 1;
    const StateLayout layout(dim, 1);
    struct Owner { HelmholtzTable::Data *lteTableData; } owner{&data};
    LTEHelmholtzGas host(physics, layout, tables);
    const auto gas = host.to_device(owner);
    const int nfp = dim == 3 ? (order + 1)*(order + 1) : order + 1;
    const int neq = dim + 2;
    const Context context{dim, neq, nfp, false, gas, {}};
    mfem::Vector u(2*neq*nfp), gx(u.Size()), gy(u.Size()), gz(u.Size());
    gx = 0.0;
    gy = 0.0;
    gz = 0.0;
    mfem::Vector normals(dim*nfp), wm(nfp), wp(nfp);
    for (int fp = 0; fp < nfp; ++fp)
      {
        const int pattern = fp%6;
        const real r1 = pattern == 5 ? 1.25 : 0.875;
        const real t1 = pattern == 5 ? 3.5 : 2.5;
        const real r2 = pattern == 1 || pattern == 2 ? r1 : pattern == 4 ? r1 + 1e-10 : 1.625;
        const real t2 = pattern == 1 || pattern == 3 ? t1 : pattern == 4 ? t1 + 1e-10 : 4.25;
        real left[MAXEQ]{}, right[MAXEQ]{};
        State(dim, r1, t1, 0, left);
        State(dim, r2, t2, pattern == 1 ? 0 : 0.25, right);
        for (int q = 0; q < neq; ++q)
          {
            u[context.iface_idx(0, fp, q)] = left[q];
            u[context.iface_idx(1, fp, q)] = right[q];
          }
        for (int d = 0; d < dim; ++d)
          {
            normals[fp*dim + d] = (fp%3 == 0 ? 0 : 1)*(d == 1 ? -0.7 : 0.3*(d + 1));
          }
        wm[fp] = 1.25;
        wp[fp] = 0.75;
      }
    // Nonzero temperature gradients force later viscosity/conductivity/EOS
    // calls. Velocity gradients are zero, so the independent viscous oracle is
    // pure heat conduction, with zero mass/momentum contribution.
    for (int side = 0; side < 2; ++side)
      {
        for (int fp = 0; fp < nfp; ++fp)
          {
            gx[context.iface_idx(side, fp, dim + 1)] = 0.25;
            gy[context.iface_idx(side, fp, dim + 1)] = -0.5;
            gz[context.iface_idx(side, fp, dim + 1)] = 0.75;
          }
      }
    mfem::Vector result(2*neq*nfp + 2), direct(neq*nfp + 2), volume(neq*nfp + 2), reverse(neq*nfp);
    mfem::Vector damping(neq*nfp + 2);
    damping = -12345.0;
    result = -12345.0;
    direct = -12345.0;
    volume = -12345.0;
    const auto *states = u.Read();
    const auto *norm = normals.Read();
    const auto *x = gx.Read();
    const auto *y = gy.Read();
    const auto *z = gz.Read();
    const auto *minus_weight = wm.Read();
    const auto *plus_weight = wp.Read();
    // Kernels leave guard entries untouched: preserve their initialized values
    // when synchronizing to device memory. Write() would discard that content.
    auto *rhs = result.ReadWrite();
    auto *face = direct.ReadWrite();
    auto *vol = volume.ReadWrite();
    auto *rev = reverse.Write();
    auto *damp = damping.ReadWrite();
    mfem::forall(point_parallel ? nfp : 1, [=] MFEM_HOST_DEVICE (int thread)
      {
        const int begin = point_parallel ? thread : 0;
        const int end = point_parallel ? thread + 1 : nfp;
        for (int fp = begin; fp < end; ++fp)
          {
            real left[MAXEQ]{}, right[MAXEQ]{}, opposite[MAXDIM]{}, metric2[MAXDIM]{};
            for (int q = 0; q < neq; ++q)
              {
                left[q] = states[context.iface_idx(0, fp, q)];
                right[q] = states[context.iface_idx(1, fp, q)];
              }
            for (int d = 0; d < dim; ++d)
              {
                opposite[d] = -norm[fp*dim + d];
                metric2[d] = 2*norm[fp*dim + d];
              }
            Lax_dissipation(gas, PointStateView(left), PointStateView(right),
              norm + fp*dim, damp + 1 + fp*neq);
            LTEECFlux::ComputeFaceFluxKernel(gas, left, right, norm + fp*dim, face + 1 + fp*neq);
            LTEECFlux::ComputeVolumeFluxKernel(gas, left, right, norm + fp*dim, metric2, vol + 1 + fp*neq);
            LTEECFlux::ComputeFaceFluxKernel(gas, right, left, opposite, rev + fp*neq);
            DGSEMIntegrator::AssembleViscousFacePointKernel(context, states, norm + fp*dim,
              minus_weight[fp], plus_weight[fp], x, y, z, 0.0, fp, rhs + 1);
          }
      });
    const auto *dissipation = damping.HostRead();
    const auto *assembled = result.HostRead();
    const auto *actual = direct.HostRead();
    const auto *ec = volume.HostRead();
    const auto *reversed = reverse.HostRead();
    mfem::Vector host_result;
    if (compare_host)
      {
        host_result.SetSize(2*neq*nfp);
      }
    const Context host_context{dim, neq, nfp, false, host, {}};

    for (int fp = 0; fp < nfp; ++fp)
      {
        const int pattern = fp%6;
        const real r1 = pattern == 5 ? 1.25 : 0.875;
        const real t1 = pattern == 5 ? 3.5 : 2.5;
        const real r2 = pattern == 1 || pattern == 2 ? r1 : pattern == 4 ? r1 + 1e-10 : 1.625;
        const real t2 = pattern == 1 || pattern == 3 ? t1 : pattern == 4 ? t1 + 1e-10 : 4.25;
        real left[MAXEQ]{}, right[MAXEQ]{}, expected[MAXEQ]{}, central[MAXEQ]{};
        State(dim, r1, t1, 0, left);
        State(dim, r2, t2, pattern == 1 ? 0 : 0.25, right);
        Oracle(dim, r1, t1, r2, t2, left, right, normals.HostRead() + fp*dim, true, expected);
        real nondissipative[MAXEQ]{};
        Oracle(dim, r1, t1, r2, t2, left, right, normals.HostRead() + fp*dim, false, nondissipative);
        real metric[MAXDIM]{};
        for (int d = 0; d < dim; ++d)
          {
            metric[d] = 1.5*normals[fp*dim + d];
          }
        Oracle(dim, r1, t1, r2, t2, left, right, metric, false, central);
        real host_face[MAXEQ]{}, host_volume[MAXEQ]{}, host_dissipation[MAXEQ]{};
        if (compare_host)
          {
            const auto *normal = normals.HostRead() + fp*dim;
            real second_metric[MAXDIM]{};
            for (int d = 0; d < dim; ++d)
              {
                second_metric[d] = 2*normal[d];
              }
            LTEECFlux::ComputeFaceFluxKernel(host, left, right, normal, host_face);
            LTEECFlux::ComputeVolumeFluxKernel(host, left, right, normal, second_metric, host_volume);
            Lax_dissipation(host, PointStateView(left), PointStateView(right), normal, host_dissipation);
            DGSEMIntegrator::AssembleViscousFacePointKernel(host_context, u.HostRead(), normal,
              wm[fp], wp[fp], gx.HostRead(), gy.HostRead(), gz.HostRead(), 0.0, fp, host_result.HostWrite());
          }
        real entropy_jump_flux = 0;
        real potential_jump = 0;
        real kinetic1 = 0;
        real kinetic2 = 0;
        for (int d = 0; d < dim; ++d)
          {
            const auto u1 = left[d + 1]/r1;
            const auto u2 = right[d + 1]/r2;
            kinetic1 += 0.5*u1*u1;
            kinetic2 += 0.5*u2*u2;
            entropy_jump_flux += (u2/t2 - u1/t1)*ec[1 + fp*neq + d + 1];
            potential_jump += (r2*r2*(2 + r2 + t2/4)*u2/t2
                               - r1*r1*(2 + r1 + t1/4)*u1/t1)*metric[d];
          }
        const auto g1 = 4*r1 + 1.5*r1*r1 + r1*t1/2 - t1*t1;
        const auto g2 = 4*r2 + 1.5*r2*r2 + r2*t2/2 - t2*t2;
        entropy_jump_flux += ((g2 - kinetic2)/t2 - (g1 - kinetic1)/t1)*ec[1 + fp*neq];
        entropy_jump_flux += (1/t1 - 1/t2)*ec[1 + fp*neq + dim + 1];
        Near(entropy_jump_flux, potential_jump, "Tadmor entropy condition");
        real thermal_normal = 0;
        const real gradient[]{0.25, -0.5, 0.75};
        for (int d = 0; d < dim; ++d)
          {
            thermal_normal += gradient[d]*normals[fp*dim + d];
          }
        const real conductivity = 0.5*((4 + 2*r1 + t1) + (4 + 2*r2 + t2));
        for (int q = 0; q < neq; ++q)
          {
            if (compare_host)
              {
                HostAgreement(actual[1 + fp*neq + q], host_face[q], "face host/backend agreement");
                HostAgreement(ec[1 + fp*neq + q], host_volume[q], "volume host/backend agreement");
                HostAgreement(dissipation[1 + fp*neq + q], host_dissipation[q], "Lax host/backend agreement");
                HostAgreement(assembled[1 + context.iface_idx(0, fp, q)],
                  host_result[context.iface_idx(0, fp, q)], "minus assembly host/backend agreement");
                HostAgreement(assembled[1 + context.iface_idx(1, fp, q)],
                  host_result[context.iface_idx(1, fp, q)], "plus assembly host/backend agreement");
              }
            Near(dissipation[1 + fp*neq + q], nondissipative[q] - expected[q], "shared Lax analytic oracle");
            Near(actual[1 + fp*neq + q], expected[q], "face analytic oracle");
            Near(ec[1 + fp*neq + q], central[q], "volume analytic oracle");
            Near(reversed[fp*neq + q], -actual[1 + fp*neq + q], "reversed orientation");
            const auto total = expected[q] - (q == dim + 1 ? conductivity*thermal_normal : 0);
            Near(assembled[1 + context.iface_idx(0, fp, q)], -1.25*total, "minus viscous-face assembly");
            Near(assembled[1 + context.iface_idx(1, fp, q)], 0.75*total, "plus viscous-face assembly");
          }
      }
    Near(dissipation[0], -12345, "Lax leading guard");
    Near(dissipation[damping.Size() - 1], -12345, "Lax trailing guard");
    Near(assembled[0], -12345, "assembly leading guard");
    Near(assembled[result.Size() - 1], -12345, "assembly trailing guard");
    Near(actual[0], -12345, "face leading guard");
    Near(actual[direct.Size() - 1], -12345, "face trailing guard");
    Near(ec[0], -12345, "volume leading guard");
    Near(ec[volume.Size() - 1], -12345, "volume trailing guard");
  }
}
