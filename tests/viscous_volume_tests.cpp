// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "DGSEMIntegrator.hpp"
#include <cmath>
#include <iostream>

namespace
{
  template<typename Gas>
  struct Context
  {
    int dim, num_equations, ndof_scalar_el, Np_x, Np_y, Np_z;
    bool axisymmetric;
    Gas gas;
    const mfem::real_t *Dhat_d;
    const mfem::real_t *D_d;
  };

  template<typename Gas>
  bool CheckCase(int dim, int order, bool axisymmetric, const Gas &host_gas, const Gas &device_gas)
  {
    const int n = order + 1;
    const int ny = dim > 1 ? n : 1;
    const int nz = dim > 2 ? n : 1;
    const int points = n * ny * nz;
    const int equations = dim + 2;
    const int size = points * equations;
    const int elements = 3;
    mfem::Vector state(elements * size), gx(elements * size), gy(elements * size);
    mfem::Vector gz(elements * size), jac(elements * points);
    mfem::Vector metric(elements * points * dim * dim), radius(elements * points);
    mfem::Vector derivative(n * n), flux(elements * size * dim);
    mfem::Vector actual(elements * size), expected(elements * size);
    mfem::Vector *vectors[] = {&state, &gx, &gy, &gz, &jac, &metric, &radius,
                              &derivative, &flux, &actual};
    for (auto *vector : vectors)
      {
        vector->UseDevice();
      }
    // Dense, nonsymmetric weights exercise every sampled flux and its ordering.
    // This tests algebraic equivalence, not a particular differentiation formula.
    for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < n; ++j)
          {
            derivative.HostWrite()[j + n * i] = (i == j ? -0.7 : 0.2) + 0.03 * (i - j);
          }
      }
    for (int e = 0; e < elements; ++e)
      {
        for (int point = 0; point < points; ++point)
          {
            const int offset = e * size;
            const mfem::real_t rho = 1.0 + 0.001 * point + 0.1 * e;
            mfem::real_t kinetic = 0.0;
            state.HostWrite()[offset + point] = rho;
            for (int d = 0; d < dim; ++d)
              {
                const mfem::real_t velocity = 0.2 * (d + 1) + 0.001 * point;
                state.HostWrite()[offset + (d + 1) * points + point] = rho * velocity;
                kinetic += 0.5 * rho * velocity * velocity;
              }
            state.HostWrite()[offset + (equations - 1) * points + point] =
              rho * 287.05 * (300.0 + point) / 0.4 + kinetic;
            jac.HostWrite()[e * points + point] = 0.8 + 0.001 * point;
            radius.HostWrite()[e * points + point] = point / n == 0 ? 0.0 : 0.3 + 0.001 * point;
            for (int row = 0; row < dim; ++row)
              {
                for (int col = 0; col < dim; ++col)
                  {
                    metric.HostWrite()[((e * points + point) * dim + row) * dim + col] =
                      (row == col ? 1.1 : 0.05) + 0.0001 * point * (row + 1);
                  }
              }
            for (int q = 0; q < equations; ++q)
              {
                const int index = offset + q * points + point;
                gx.HostWrite()[index] = 0.01 * (q + 1) + 0.0001 * point;
                gy.HostWrite()[index] = -0.02 * (q + 1) + 0.0002 * point;
                gz.HostWrite()[index] = 0.03 * (q + 1) - 0.0001 * point;
                actual.HostWrite()[index] = expected[index] = 0.25 + 0.001 * index;
              }
          }
      }
    Context<Gas> host{dim, equations, points, n, ny, nz, axisymmetric, host_gas,
                 derivative.HostRead(), derivative.HostRead()};
    // Retained element implementation is independent of the new flux cache.
    for (int e = 0; e < elements; ++e)
      {
        Theseus::DGSEMIntegrator::AssembleViscousElementVolumeKernel(
          host, state.HostRead() + e * size, jac.HostRead() + e * points,
          metric.HostRead() + e * points * dim * dim, radius.HostRead() + e * points,
          gx.HostRead() + e * size, gy.HostRead() + e * size,
          gz.HostRead() + e * size, expected.HostReadWrite() + e * size);
      }
    Context<Gas> device = host;
    device.gas = device_gas;
    device.Dhat_d = derivative.Read();
    device.D_d = derivative.Read();
    const auto *u = state.Read();
    const auto *x = gx.Read();
    const auto *y = gy.Read();
    const auto *z = gz.Read();
    const auto *j = jac.Read();
    const auto *m = metric.Read();
    const auto *r = radius.Read();
    auto *f = flux.Write();
    mfem::forall(elements * points, [=] MFEM_HOST_DEVICE (int index)
    {
      const int e = index / points;
      Theseus::DGSEMIntegrator::ComputeViscousVolumeFluxPointKernel(
        device, u + e * size, m + e * points * dim * dim, r + e * points,
        x + e * size, y + e * size, z + e * size,
        index % points, f + e * size * dim);
    });
    const auto *cached = flux.Read();
    auto *result = actual.ReadWrite();
    mfem::forall(elements * points, [=] MFEM_HOST_DEVICE (int index)
    {
      const int e = index / points;
      Theseus::DGSEMIntegrator::AssembleViscousVolumePointKernel(
        device, u + e * size, j + e * points, m + e * points * dim * dim,
        r + e * points, x + e * size, y + e * size, z + e * size,
        cached + e * size * dim, index % points, result + e * size);
    });
    const auto *values = actual.HostRead();
    for (int i = 0; i < elements * size; ++i)
      {
        const auto error = std::abs(values[i] - expected[i]);
        if (!std::isfinite(values[i]) || error > 2e-12 * (1.0 + std::abs(expected[i])))
          {
            std::cerr << "Mismatch: dim=" << dim << " order=" << order
                      << " axis=" << axisymmetric << " index=" << i
                      << " error=" << error << '\n';
            return false;
          }
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
  for (int order : {2, 4, 6})
    {
      for (int dim : {1, 2, 3})
        {
          Theseus::PhysicsConstants physics(1.4, 0.72, 287.05, 1.8e-5);
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
              if (!CheckCase(dim, order, axisymmetric, cpg, cpg) ||
                  !CheckCase(dim, order, axisymmetric, lte, device_lte))
                {
                  return 1;
                }
            }
        }
    }
  std::cout << "Viscous flux cache matches element reference in all 24 CPG/LTE cases.\n";
  return 0;
}
