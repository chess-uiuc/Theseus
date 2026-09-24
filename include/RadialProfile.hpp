// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Theseus {

/// Stationary meridional primitive data, sampled as a function of radius.
/// Reading and interpolation run on the host during boundary initialization;
/// the boundary kernels consume the resulting cached conservative states.
struct RadialProfile {
  // Columns: radius [m], static temperature [K], axial and radial velocity [m/s].
  // The input's flag and swirl columns are not part of the retained state.
  using Row = std::array<double, 4>;
  std::vector<Row> rows;

  /// Read the existing profile format and retain the block selected by flag:
  ///   points_per_block number_of_blocks
  ///   radius flag temperature ux ur swirl
  /// Blank data lines are allowed; each nonblank row must have six columns.
  /// Swirl is parsed but intentionally ignored in the swirl-free formulation.
  /// Invalid input, inconsistent counts, or a missing flag throws runtime_error.
  static RadialProfile Read(const std::string &path, int flag) {
    std::ifstream f(path);
    int count = 0, blocks = 0;
    std::string line, extra;

    if (!std::getline(f, line))
      throw std::runtime_error("Cannot read profile: " + path);

    std::istringstream header(line);
    if (!(header >> count >> blocks) || (header >> extra) || count < 2 || blocks < 1)
      throw std::runtime_error("Profile requires header: points blocks");

    RadialProfile result;
    int total = 0;
    while (std::getline(f, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos)
        continue;

      std::istringstream in(line);
      double r, label, T, ux, ur, swirl;
      if (!(in >> r >> label >> T >> ux >> ur >> swirl) || (in >> extra) ||
          !std::isfinite(r + label + T + ux + ur + swirl) || r < 0 || T <= 0)
        throw std::runtime_error("Invalid radial profile row");

      // Count every block, but check radial ordering only within the selected one.
      // Different blocks can share radii without creating interpolation duplicates.
      ++total;
      if (label != flag)
        continue;

      if (!result.rows.empty() && r <= result.rows.back()[0])
        throw std::runtime_error("Selected profile radii must increase strictly");
      result.rows.push_back({r, T, ux, ur});
    }

    if (total != count * blocks || result.rows.size() != static_cast<size_t>(count))
      throw std::runtime_error("Profile header/selected flag count mismatch");

    // A supplied axis sample must already satisfy the radial-velocity condition.
    // If no axis sample exists, Evaluate supplies the extension described below.
    if (result.rows.front()[0] == 0 && result.rows.front()[3] != 0)
      throw std::runtime_error("Radial velocity must vanish at axis");
    return result;
  }

  /// Return {r, T, ux, ur} by piecewise-linear primitive interpolation.
  /// Rows must be valid and strictly increasing, as guaranteed by Read.
  /// Radii outside the table are rejected, except for the axis extension and
  /// a 1e-12 m roundoff allowance at the upper endpoint.
  Row Evaluate(double r) const {
    if (rows.size() < 2 || !std::isfinite(r) || r < 0 || r > rows.back()[0] + 1e-12)
      throw std::runtime_error("Boundary radius outside profile coverage");

    const auto &a = rows.front();
    // Below the first sample: keep T and ux constant, and take ur linearly to
    // zero at r=0. These provide even T/ux and odd ur continuations at the axis.
    if (r <= a[0])
      return {r, a[1], a[2], a[0] > 0 ? a[3] * r / a[0] : 0.0};

    // Interpolate primitives before converting to a conservative state.
    // Preserve signed velocities, including any measured local reverse flow.
    for (size_t i = 1; i < rows.size(); ++i)
      if (r <= rows[i][0] + 1e-12) {
        const auto &l = rows[i-1];
        const auto &h = rows[i];
        double t = std::min(1.0, (r - l[0]) / (h[0] - l[0]));
        return {r, l[1] + t * (h[1] - l[1]), l[2] + t * (h[2] - l[2]),
                l[3] + t * (h[3] - l[3])};
      }
    throw std::runtime_error("Profile interpolation failed");
  }
};

/// Convert static primitives to the 2D CPG state [rho, rho*ux, rho*ur, rho*E].
/// Inputs use Pa, K, m/s and R in J/(kg K); gamma is dimensionless.
/// Total energy includes internal and meridional kinetic energy, with no swirl.
/// This is an ordinary physical state: no axisymmetric radius weighting is used.
/// This helper is CPG-specific and must not be used for an LTE equation of state.
inline std::array<double, 4> CPGState(double p, double T, double ux, double ur,
                                     double gamma, double R) {
  if (!(p > 0 && T > 0 && gamma > 1 && R > 0) || !std::isfinite(p + T + ux + ur + gamma + R))
    throw std::runtime_error("Invalid CPG primitive state");

  double rho = p / (R * T);
  return {rho, rho * ux, rho * ur,
          p / (gamma - 1) + 0.5 * rho * (ux * ux + ur * ur)};
}

} // namespace Theseus
