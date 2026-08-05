// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace
{
  std::vector<std::pair<mfem::real_t, mfem::real_t>> SortedPoints(
    const mfem::IntegrationRule &rule)
  {
    std::vector<std::pair<mfem::real_t, mfem::real_t>> points;
    for (int i = 0; i < rule.GetNPoints(); i++)
      {
        const auto &point = rule.IntPoint(i);
        points.emplace_back(point.x, point.y);
      }
    std::sort(points.begin(), points.end());
    return points;
  }
}

TEST(gll_subcell_points_match_dgsem_nodes)
{
  constexpr int order = 3;
  mfem::GeometryRefiner refiner(mfem::Quadrature1D::GaussLobatto);
  const auto *refined = refiner.Refine(mfem::Geometry::SQUARE, order);

  mfem::DG_FECollection collection(order, 2, mfem::BasisType::GaussLobatto);
  const auto *element = collection.FiniteElementForGeometry(mfem::Geometry::SQUARE);

  const auto refined_points = SortedPoints(refined->RefPts);
  const auto element_nodes = SortedPoints(element->GetNodes());
  EXPECT_EQ(refined_points.size(), element_nodes.size());
  for (std::size_t i = 0; i < refined_points.size(); i++)
    {
      EXPECT_CLOSE(refined_points[i].first, element_nodes[i].first, 1e-15);
      EXPECT_CLOSE(refined_points[i].second, element_nodes[i].second, 1e-15);
    }
  return 0;
}

TEST(order_three_gll_output_has_nine_linear_subcells)
{
  constexpr int order = 3;
  mfem::GeometryRefiner refiner(mfem::Quadrature1D::GaussLobatto);
  const auto *refined = refiner.Refine(mfem::Geometry::SQUARE, order);
  const int vertices_per_quad = 4;

  EXPECT_EQ(refined->RefPts.GetNPoints(), 16);
  EXPECT_EQ(refined->RefGeoms.Size() / vertices_per_quad, 9);
  return 0;
}
