// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "ChannelFlowInitialCondition.hpp"
#include <iostream>
#include <limits>
#include <utility>
int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  auto profile = Prandtl::ChannelFlowIC(293.15, 1.5);
  mfem::Vector x(3), a(5), b(5);
  int failures = 0;
  for (double y : {-1.0, -0.4, 0.0, 0.6, 1.0})
    {
      x[0] = 0.7; x[1] = y; x[2] = 0.3;
      profile(x, a); profile(x, b);
      for (int q = 0; q < 5; ++q)
        {
          if (!std::isfinite(a[q]) || a[q] != b[q]) { ++failures; }
        }
      const auto internal_energy = a[4] - 0.5*(a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
      if (a[0] != 1 || std::abs(internal_energy - 287.05*293.15/0.4) > 1e-8) { ++failures; }
      if (std::abs(y) == 1 && (a[1] != 0 || a[2] != 0 || a[3] != 0)) { ++failures; }
      for (int d : {0, 2})
        {
          x[d] += d == 0 ? 4*M_PI : 4*M_PI/3;
          profile(x, b);
          for (int q = 0; q < 5; ++q)
            {
              if (std::abs(a[q] - b[q]) > 1e-12*std::max(1.0, std::abs(a[q]))) { ++failures; }
            }
          x[d] -= d == 0 ? 4*M_PI : 4*M_PI/3;
        }
    }
  for (auto values : {std::pair<double,double>{-1, 1}, {293.15, -1},
                     {std::numeric_limits<double>::quiet_NaN(), 1}})
    {
      bool rejected = false;
      try { Prandtl::ChannelFlowIC(values.first, values.second); }
      catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) { ++failures; }
    }
  std::cout << "Channel initialization failures: " << failures << '\n';
  return failures ? 1 : 0;
}
