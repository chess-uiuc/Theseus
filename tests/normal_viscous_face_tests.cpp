// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "DGSEMIntegrator.hpp"
#include "LaxFriedrichsFlux.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
  template<typename Gas>
  struct FaceContext
  {
    int dim;
    int num_equations;
    int num_face_points;
    bool axisymmetric;
    Gas gas;
    Theseus::LaxFriedrichsFlux::InviscidFlux iflux;

    MFEM_HOST_DEVICE int iface_idx(int side, int fp, int q) const
    {
      return (side * num_equations + q) * num_face_points + fp;
    }
  };

  // Independent closed-form stress/heat-flux oracle. It deliberately does not
  // call either physical viscous kernel or the temperature-gradient helper.
  template<typename Gas>
  void Reference(const Gas &gas, bool lte, bool axisymmetric, mfem::real_t radius,
                 const mfem::real_t *state, const mfem::real_t *gx,
                 const mfem::real_t *gy, const mfem::real_t *gz,
                 int stride, int offset, const mfem::real_t *normal,
                 mfem::real_t *flux)
  {
    Theseus::PointStateView S{state};
    const int dim = gas.dim();
    const int energy = dim + 1;
    const mfem::real_t rho = state[0];
    const mfem::real_t temperature = gas.temperature(S);
    const mfem::real_t mu = gas.viscosity(S);
    const mfem::real_t kappa = gas.thermal_conductivity(S);
    const mfem::real_t *gradient[] = {gx, gy, gz};
    mfem::real_t velocity[3] = {};
    mfem::real_t derivative[3][3] = {};
    mfem::real_t div = 0.0;
    for (int i = 0; i < dim; ++i)
      {
        velocity[i] = state[1 + i] / rho;
        for (int j = 0; j < dim; ++j)
          {
            derivative[i][j] = gradient[j][offset + (1 + i) * stride];
          }
        div += derivative[i][i];
      }
    if (axisymmetric)
      {
        if (radius <= Theseus::AxisymmetricGeometry::radius_tolerance)
          {
            div += derivative[1][1];
            velocity[1] = 0.0;
          }
        else
          {
            div += velocity[1] / radius;
          }
      }
    for (int q = 0; q < dim + 2; ++q)
      {
        flux[q] = 0.0;
      }
    for (int i = 0; i < dim; ++i)
      {
        for (int j = 0; j < dim; ++j)
          {
            const mfem::real_t tau = mu * (derivative[i][j] + derivative[j][i]
              - (i == j ? gas.phys.mu_bulk * div : 0.0));
            flux[1 + i] += tau * normal[j];
            flux[energy] += velocity[i] * tau * normal[j];
          }
      }
    for (int j = 0; j < dim; ++j)
      {
        const mfem::real_t thermal_derivative = gradient[j][offset + energy * stride];
        const mfem::real_t density_derivative = gradient[j][offset];
        const mfem::real_t dT = lte ? thermal_derivative
          : (thermal_derivative - gas.phys.R_gas * temperature * density_derivative)
            / (rho * gas.phys.R_gas);
        flux[energy] += kappa * dT * normal[j];
      }
  }

  template<typename Gas>
  bool CheckFaces(int dim, int nfp, bool axisymmetric, bool lte,
                  const Gas &host_gas, const Gas &device_gas, bool face_parallel,
                  int generation = 0)
  {
    const int faces = 7;
    const int points = faces * nfp;
    const int equations = dim + 2;
    const int face_size = 2 * equations * nfp;
    mfem::Vector state(faces * face_size), gx(state.Size()), gy(state.Size()), gz(state.Size());
    mfem::Vector normal(points * dim), radius(points), minus_weight(points), plus_weight(points);
    mfem::Vector actual(state.Size() + 2), expected(state.Size());
    mfem::Vector *buffers[] = {&state, &gx, &gy, &gz, &normal, &radius,
                              &minus_weight, &plus_weight, &actual};
    for (auto *buffer : buffers)
      {
        buffer->UseDevice(true);
      }
    actual = -987654.25;
    FaceContext<Gas> host{dim, equations, nfp, axisymmetric, host_gas, {}};
    for (int f = 0; f < faces; ++f)
      {
        for (int fp = 0; fp < nfp; ++fp)
          {
            const int point = f * nfp + fp;
            radius.HostReadWrite()[point] = point % 3 == 0 ? 0.0 : 0.17 + 0.001 * point;
            minus_weight.HostReadWrite()[point] = 0.7 + 0.003 * point;
            plus_weight.HostReadWrite()[point] = 1.2 + 0.005 * point;
            for (int d = 0; d < dim; ++d)
              {
                normal.HostReadWrite()[point * dim + d] =
                  (point % (dim + 1) == d ? 0.0 : (d % 2 ? -0.8 : 0.3)) * (1.0 + 0.01 * point);
              }
            mfem::real_t minus[5] = {}, plus[5] = {}, numerical[5] = {}, viscous[5] = {};
            for (int side = 0; side < 2; ++side)
              {
                mfem::real_t *q = side == 0 ? minus : plus;
                q[0] = 1.0 + 0.003 * point + 0.04 * side + 0.01 * generation;
                mfem::real_t kinetic = 0.0;
                for (int d = 0; d < dim; ++d)
                  {
                    const mfem::real_t velocity = 0.2 + 0.07 * d + 0.002 * point + 0.03 * side + 0.02 * generation;
                    q[1 + d] = q[0] * velocity;
                    kinetic += 0.5 * q[0] * velocity * velocity;
                  }
                q[dim + 1] = q[0] * 287.05 * (350.0 + 0.1 * point + side + generation) / 0.4 + kinetic;
                for (int eq = 0; eq < equations; ++eq)
                  {
                    const int index = f * face_size + host.iface_idx(side, fp, eq);
                    state.HostReadWrite()[index] = q[eq];
                    gx.HostReadWrite()[index] = (lte && eq == 0) ? 0.0 : 0.13 * (eq + 1) + 0.002 * point + side;
                    gy.HostReadWrite()[index] = (lte && eq == 0) ? 0.0 : -0.09 * (eq + 1) + 0.003 * point - side;
                    gz.HostReadWrite()[index] = (lte && eq == 0) ? 0.0 : 0.05 * (eq + 1) - 0.004 * point + side;
                  }
              }
            host.iflux.ComputeFaceFlux(host.gas, minus, plus, normal.HostRead() + point * dim, numerical);
            for (int side = 0; side < 2; ++side)
              {
                Reference(host.gas, lte, axisymmetric, radius.HostRead()[point],
                          side == 0 ? minus : plus, gx.HostRead() + f * face_size,
                          gy.HostRead() + f * face_size, gz.HostRead() + f * face_size,
                          nfp, host.iface_idx(side, fp, 0), normal.HostRead() + point * dim, viscous);
                for (int eq = 0; eq < equations; ++eq)
                  {
                    numerical[eq] -= 0.5 * viscous[eq];
                  }
              }
            for (int eq = 0; eq < equations; ++eq)
              {
                expected[f * face_size + host.iface_idx(0, fp, eq)] = -minus_weight.HostRead()[point] * numerical[eq];
                expected[f * face_size + host.iface_idx(1, fp, eq)] = plus_weight.HostRead()[point] * numerical[eq];
              }
          }
      }
    auto context = host;
    context.gas = device_gas;
    const auto *u = state.Read();
    const auto *x = gx.Read();
    const auto *y = gy.Read();
    const auto *z = gz.Read();
    const auto *n = normal.Read();
    const auto *r = radius.Read();
    const auto *wm = minus_weight.Read();
    const auto *wp = plus_weight.Read();
    auto *rhs = actual.ReadWrite() + 1;
    mfem::forall(face_parallel ? faces : points, [=] MFEM_HOST_DEVICE (int index)
    {
      if (face_parallel)
        {
          Theseus::DGSEMIntegrator::AssembleViscousElementFaceKernel(
            context, u + index * face_size, n + index * nfp * dim,
            wm + index * nfp, wp + index * nfp,
            x + index * face_size, y + index * face_size, z + index * face_size,
            r + index * nfp, rhs + index * face_size);
        }
      else
        {
          const int f = index / nfp;
          const int fp = index % nfp;
          Theseus::DGSEMIntegrator::AssembleViscousFacePointKernel(
            context, u + f * face_size, n + index * dim, wm[index], wp[index],
            x + f * face_size, y + f * face_size, z + f * face_size,
            r[index], fp, rhs + f * face_size);
        }
    });
    const auto *result = actual.HostRead() + 1;
    const mfem::real_t tolerance = 2e-12 + 128 * std::numeric_limits<mfem::real_t>::epsilon();
    for (int i = 0; i < expected.Size(); ++i)
      {
        if (!std::isfinite(result[i]) || std::abs(result[i] - expected[i]) > tolerance * (1.0 + std::abs(expected[i])))
          {
            std::cerr << "Normal face mismatch dim=" << dim << " nfp=" << nfp
                      << " LTE=" << lte << " axis=" << axisymmetric << " face_parallel=" << face_parallel
                      << " index=" << i << " expected=" << expected[i] << " actual=" << result[i] << '\n';
            return false;
          }
      }
    const auto *storage = actual.HostRead();
    if (storage[0] != -987654.25 || storage[actual.Size() - 1] != -987654.25)
      {
        std::cerr << "Face RHS boundary was overwritten\n";
        return false;
      }
    return true;
  }
}

int main(int argc, char **argv)
{
  mfem::Device device(argc > 1 ? argv[1] : "cpu");
  // Analytic LTE fixture with variable transport and exact linear caloric inversion.
  // Owned MFEM buffers exercise the real LTE device-table pointer path.
  struct TableOwner
  {
    std::unique_ptr<Theseus::LTETable::Data> lteTableData =
      std::make_unique<Theseus::LTETable::Data>();
  } owner;
  auto &data = *owner.lteTableData;
  Theseus::LTETable::LTETables tables(3, 3);
  data.rho_grid.SetSize(3);
  data.T_grid.SetSize(3);
  data.e_grid.SetSize(3);
  data.lte_table.SetSize(81);
  data.inv_table.SetSize(9);
  for (int i = 0; i < 3; ++i)
    {
      data.rho_grid[i] = 0.5 + i;
      data.T_grid[i] = 200.0 + 400.0 * i;
      data.e_grid[i] = 287.05 * data.T_grid[i] / 0.4;
    }
  for (int j = 0; j < 3; ++j)
    {
      for (int i = 0; i < 3; ++i)
        {
          const auto rho = data.rho_grid[i];
          const auto temperature = data.T_grid[j];
          const mfem::real_t properties[] = {
            rho * 287.05 * temperature, 287.05 * temperature / 0.4,
            287.05 / 0.4, 1.4 * 287.05 / 0.4, 287.05, 1.4,
            std::sqrt(1.4 * 287.05 * temperature),
            1e-5 * (1.0 + temperature / 300.0 + rho / 10.0),
            0.02 * (1.0 + temperature / 300.0 + rho / 10.0)};
          for (int q = 0; q < 9; ++q)
            {
              data.lte_table[tables.L.property_index(q, i, j)] = properties[q];
            }
          data.inv_table[j * 3 + i] = temperature;
        }
    }
  tables.tables = {data.lte_table.HostRead(), data.inv_table.HostRead(),
                   data.rho_grid.HostRead(), data.T_grid.HostRead(), data.e_grid.HostRead()};

  for (int dim : {1, 2, 3})
    {
      for (int order : {1, 2, 3})
        {
          const int nfp = dim == 1 ? 1 : (dim == 2 ? order + 1 : (order + 1) * (order + 1));
          Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 1.8e-5);
          physics.mu_bulk = 0.41;
          Theseus::StateLayout layout(dim, 1);
          Theseus::IdealGasModel cpg(physics, layout);
          Theseus::LTEGas lte(physics, layout, tables);
          const auto device_lte = lte.to_device(owner);
          for (bool axisymmetric : {false, true})
            {
              if (axisymmetric && dim != 2)
                {
                  continue;
                }
              for (bool face_parallel : {false, true})
                {
                  if (!CheckFaces(dim, nfp, axisymmetric, false, cpg, cpg, face_parallel) ||
                      !CheckFaces(dim, nfp, axisymmetric, true, lte, device_lte, face_parallel))
                    {
                      return 1;
                    }
                }
            }
        }
    }
  // Exercise changing inputs and point counts through the original assembly API.
  Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 1.8e-5);
  Theseus::StateLayout layout(3, 1);
  Theseus::LTEGas lte(physics, layout, tables);
  const auto device_lte = lte.to_device(owner);
  int generation = 0;
  for (int nfp : {4, 4, 9, 1, 16, 4})
    {
      for (bool face_parallel : {false, true})
        {
          if (!CheckFaces(3, nfp, false, true, lte, device_lte, face_parallel,
                          ++generation))
            {
              return 1;
            }
        }
    }
  std::cout << "Viscous face assembly matches independent stress/heat-flux formulas.\n";
  return 0;
}
