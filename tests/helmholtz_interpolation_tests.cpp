// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "LTEHelmholtzInterpolation.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
  using real_t = mfem::real_t;
  namespace H = Theseus::HelmholtzInterpolation;
  int failures = 0;
  int checks = 0;

  double Power(double value, int exponent)
  {
    double result = 1;
    for (int i = 0; i < exponent; ++i)
      {
        result *= value;
      }
    return result;
  }

  // Differentiate the physical power monomial directly, independently of
  // the Hermite interpolation/basis used by the implementation under test.
  double Monomial(double density, double temperature, int density_power,
                  int temperature_power, int density_derivative,
                  int temperature_derivative)
  {
    if (density_derivative > density_power || temperature_derivative > temperature_power)
      {
        return 0;
      }
    double factor = 1;
    for (int i = 0; i < density_derivative; ++i)
      {
        factor *= density_power - i;
      }
    for (int i = 0; i < temperature_derivative; ++i)
      {
        factor *= temperature_power - i;
      }
    return factor * Power(density, density_power - density_derivative)
           * Power(temperature, temperature_power - temperature_derivative);
  }

  double BasisBound(int derivative, bool slope)
  {
    if (derivative == 0)
      {
        return 1;
      }
    if (derivative == 1)
      {
        return slope ? 1 : 1.5;
      }
    return slope ? 4 : 6;
  }

  // Bound the absolute sum by the exact maxima of the one-dimensional basis
  // factors (or conservative bounds on them) on [0,1]. This tolerance measures
  // interpolation roundoff, not a CPU/GPU solver-agreement acceptance criterion.
  double RoundoffScale(const H::Cell &cell, int dr, int dt)
  {
    double sum = 0;
    for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 2; ++j)
          {
            const int index = i + j * cell.temperature_stride;
            sum += std::abs(cell.potential[index]) * BasisBound(dr, false) * BasisBound(dt, false);
            sum += std::abs(cell.density_derivative[index] * cell.density_width)
                   * BasisBound(dr, true) * BasisBound(dt, false);
            sum += std::abs(cell.temperature_derivative[index] * cell.temperature_width)
                   * BasisBound(dr, false) * BasisBound(dt, true);
            sum += std::abs(cell.mixed_derivative[index] * cell.density_width * cell.temperature_width)
                   * BasisBound(dr, true) * BasisBound(dt, true);
          }
      }
    return sum / (Power(cell.density_width, dr) * Power(cell.temperature_width, dt));
  }

  void Check(double actual, double expected, double scale, const char *where)
  {
    ++checks;
    const double tolerance = 128 * std::numeric_limits<real_t>::epsilon()
                             * std::max({1.0, std::abs(expected), scale});
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
      {
        if (failures < 12)
          {
            std::cerr.precision(17);
            std::cerr << where << ": actual=" << actual << " expected=" << expected
                      << " error=" << std::abs(actual - expected)
                      << " tolerance=" << tolerance << '\n';
          }
        ++failures;
      }
  }

  template<int DR, int DT>
  void CheckMonomial(const H::Cell &cell, real_t xi, real_t eta,
                     double density, double temperature, int a, int b)
  {
    Check(H::Evaluate<DR, DT>(cell, xi, eta),
          Monomial(density, temperature, a, b, DR, DT),
          RoundoffScale(cell, DR, DT), "physical monomial derivative");
  }

  template<int Axis, int DR, int DT>
  void CheckDividedMonomial(const H::Cell &cell, real_t xi, real_t eta,
      real_t density_low, real_t temperature_low, int a, int b)
  {
    for (int separation = 0; separation < 3; ++separation)
      {
        const auto start = Axis == 0 ? xi : eta;
        const auto finish = separation == 0 ? start : separation == 1
                            ? start + (start == 1 ? -1e-10 : 1e-10) : 0.5*(start + 1);
        const auto r1 = density_low + xi*cell.density_width;
        const auto r2 = density_low + (Axis == 0 ? finish : xi)*cell.density_width;
        const auto t1 = temperature_low + eta*cell.temperature_width;
        const auto t2 = temperature_low + (Axis == 1 ? finish : eta)*cell.temperature_width;
        double expected = 0;
        if (a >= DR && b >= DT)
          {
            double factor = 1;
            for (int k = 0; k < DR; ++k)
              {
                factor *= a - k;
              }
            for (int k = 0; k < DT; ++k)
              {
                factor *= b - k;
              }
            const int exponent = Axis == 0 ? a - DR : b - DT;
            for (int k = 0; k < exponent; ++k)
              {
                expected += Axis == 0 ? Power(r1, exponent - 1 - k)*Power(r2, k)
                                      : Power(t1, exponent - 1 - k)*Power(t2, k);
              }
            expected *= factor*(Axis == 0 ? Power(t1, b - DT) : Power(r1, a - DR));
          }
        const auto actual = H::DividedDifference<Axis, DR, DT>(cell, xi, eta,
                              Axis == 0 ? finish : xi, Axis == 1 ? finish : eta);
        Check(actual, expected, RoundoffScale(cell, DR + (Axis == 0), DT + (Axis == 1)),
              "physical divided monomial derivative");
      }
  }

  void PolynomialReproduction()
  {
    const real_t density_grid[3] = {0.5, 1.25, 2.5};
    const real_t temperature_grid[3] = {0.25, 1.0, 3.25};
    const real_t coordinates[5] = {0, 0.125, 0.5, 0.875, 1};
    for (int a = 0; a <= 3; ++a)
      {
        for (int b = 0; b <= 3; ++b)
          {
            // Test-only table storage: four planes, with density contiguous.
            std::vector<real_t> table(36);
            for (int i = 0; i < 3; ++i)
              {
                for (int j = 0; j < 3; ++j)
                  {
                    const int index = i + 3 * j;
                    table[index] = Monomial(density_grid[i], temperature_grid[j], a, b, 0, 0);
                    table[9 + index] = Monomial(density_grid[i], temperature_grid[j], a, b, 1, 0);
                    table[18 + index] = Monomial(density_grid[i], temperature_grid[j], a, b, 0, 1);
                    table[27 + index] = Monomial(density_grid[i], temperature_grid[j], a, b, 1, 1);
                  }
              }
            const auto original = table;
            for (int i = 0; i < 2; ++i)
              {
                for (int j = 0; j < 2; ++j)
                  {
                    const int lower = i + 3 * j;
                    const H::Cell cell = {
                      table.data() + lower, table.data() + 9 + lower,
                      table.data() + 18 + lower, table.data() + 27 + lower, 3,
                      density_grid[i+1] - density_grid[i],
                      temperature_grid[j+1] - temperature_grid[j]};
                    for (const auto xi : coordinates)
                      {
                        for (const auto eta : coordinates)
                          {
                            const double density = double(density_grid[i]) + double(xi) * cell.density_width;
                            const double temperature = double(temperature_grid[j]) + double(eta) * cell.temperature_width;
                            CheckMonomial<0, 0>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<1, 0>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<0, 1>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<2, 0>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<1, 1>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<0, 2>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<2, 1>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<1, 2>(cell, xi, eta, density, temperature, a, b);
                            CheckMonomial<2, 2>(cell, xi, eta, density, temperature, a, b);
                            CheckDividedMonomial<0, 0, 0>(cell, xi, eta, density_grid[i], temperature_grid[j], a, b);
                            CheckDividedMonomial<0, 1, 0>(cell, xi, eta, density_grid[i], temperature_grid[j], a, b);
                            CheckDividedMonomial<1, 0, 0>(cell, xi, eta, density_grid[i], temperature_grid[j], a, b);
                            CheckDividedMonomial<1, 1, 0>(cell, xi, eta, density_grid[i], temperature_grid[j], a, b);
                          }
                      }
                  }
              }
            ++checks;
            if (std::memcmp(original.data(), table.data(), table.size() * sizeof(real_t)) != 0)
              {
                ++failures;
                std::cerr << "Interpolation modified its input table\n";
              }
          }
      }
  }

  void IndependentCornerData()
  {
    // Exact-rational power-basis reference from the four arbitrary corner
    // jets below. The associated polynomial is not assumed to be a physical EOS.
    const real_t potential[6] = {17, 2, 12345, -5, 11, 12345};
    const real_t density[6] = {3, -7, 12345, 13, 19, 12345};
    const real_t temperature[6] = {23, -29, 12345, 31, -37, 12345};
    const real_t mixed[6] = {-41, 43, 12345, 47, -53, 12345};
    const H::Cell cell = {potential, density, temperature, mixed, 3, 0.375, 2.5};
    // Constants are filled from an independent rational power-basis expansion.
    const double expected[6] = {
      -2.8254470825195312, 43.547119140625, -15.3924560546875, 475.9739583333333, 120.87109375, 19.55453125
    };
    const real_t xi = 0.25;
    const real_t eta = 0.625;
    Check(H::Evaluate<0, 0>(cell, xi, eta), expected[0], RoundoffScale(cell, 0, 0), "arbitrary corners H");
    Check(H::Evaluate<1, 0>(cell, xi, eta), expected[1], RoundoffScale(cell, 1, 0), "arbitrary corners H_r");
    Check(H::Evaluate<0, 1>(cell, xi, eta), expected[2], RoundoffScale(cell, 0, 1), "arbitrary corners H_T");
    Check(H::Evaluate<2, 0>(cell, xi, eta), expected[3], RoundoffScale(cell, 2, 0), "arbitrary corners H_rr");
    Check(H::Evaluate<1, 1>(cell, xi, eta), expected[4], RoundoffScale(cell, 1, 1), "arbitrary corners H_rT");
    Check(H::Evaluate<0, 2>(cell, xi, eta), expected[5], RoundoffScale(cell, 0, 2), "arbitrary corners H_TT");
  }

  void KernelEvaluation()
  {
    // Exercise the same helper from a real forall launch. Device pointers are
    // rebound into the cell inside the lambda rather than capturing host planes.
    mfem::Vector table(24);
    auto *host = table.HostWrite();
    for (int i = 0; i < 24; ++i)
      {
        host[i] = 12345;
      }
    for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 2; ++j)
          {
            const int index = i + 3*j;
            const double r = 0.75 + i*0.5;
            const double t = 0.5 + j;
            host[index] = Monomial(r, t, 3, 3, 0, 0);
            host[6+index] = Monomial(r, t, 3, 3, 1, 0);
            host[12+index] = Monomial(r, t, 3, 3, 0, 1);
            host[18+index] = Monomial(r, t, 3, 3, 1, 1);
          }
      }
    const real_t *input = table.Read();
    mfem::Vector output(25*6);
    real_t *values = output.Write();
    mfem::forall(25, [=] MFEM_HOST_DEVICE (int p)
    {
      const H::Cell cell = {input, input+6, input+12, input+18, 3, 0.5, 1.0};
      const real_t xi = (p % 5) * real_t(0.25);
      const real_t eta = (p / 5) * real_t(0.25);
      values[6*p] = H::Evaluate<0, 0>(cell, xi, eta);
      values[6*p+1] = H::Evaluate<1, 0>(cell, xi, eta);
      values[6*p+2] = H::Evaluate<0, 1>(cell, xi, eta);
      values[6*p+3] = H::Evaluate<2, 0>(cell, xi, eta);
      values[6*p+4] = H::Evaluate<1, 1>(cell, xi, eta);
      values[6*p+5] = H::Evaluate<0, 2>(cell, xi, eta);
    });
    const real_t *results = output.HostRead();
    const int derivatives[6][2] = {{0,0}, {1,0}, {0,1}, {2,0}, {1,1}, {0,2}};
    const H::Cell cell = {host, host+6, host+12, host+18, 3, 0.5, 1.0};
    for (int p = 0; p < 25; ++p)
      {
        const double r = 0.75 + (p % 5)*0.125;
        const double t = 0.5 + (p / 5)*0.25;
        for (int q = 0; q < 6; ++q)
          {
            const int dr = derivatives[q][0];
            const int dt = derivatives[q][1];
            Check(results[6*p+q], Monomial(r,t,3,3,dr,dt),
                  RoundoffScale(cell, dr, dt), "forall polynomial derivative");
          }
      }
  }
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend);
  if (backend == "cuda" && !mfem::Device::Allows(mfem::Backend::CUDA_MASK))
    {
      std::cerr << "CUDA test did not select a CUDA backend\n";
      return 2;
    }
  PolynomialReproduction();
  IndependentCornerData();
  KernelEvaluation();
  std::cout << "Hermite interpolation: " << checks << " checks, "
            << failures << " failures, backend=" << backend << '\n';
  return failures ? 1 : 0;
}
