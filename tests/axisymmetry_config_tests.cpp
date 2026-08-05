// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus testing suites.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"

#include "AxisymmetryConfig.hpp"

#include <stdexcept>
#include <string>

using Theseus::AxisymmetryConfig;

namespace
{
  bool ConfigRejected(const nlohmann::json &config,
                      const nlohmann::json &runtime,
                      bool axisymmetric_build,
                      const std::string &message_fragment)
  {
    try
      {
        AxisymmetryConfig::Validate(config, runtime, axisymmetric_build);
      }
    catch (const std::invalid_argument &error)
      {
        return std::string(error.what()).find(message_fragment) !=
               std::string::npos;
      }
    return false;
  }
}

TEST(axisymmetry_contract_defines_z_r_coordinates)
{
  EXPECT_EQ(AxisymmetryConfig::axial_coordinate, 0);
  EXPECT_EQ(AxisymmetryConfig::radial_coordinate, 1);
  EXPECT_EQ(AxisymmetryConfig::spatial_dimension, 2);
  EXPECT_EQ(AxisymmetryConfig::conservative_equations, 4);
  return 0;
}

TEST(axisymmetric_build_accepts_swirl_free_2d_layout)
{
  const auto config = nlohmann::json::parse(R"({
    "compileTime": {"AXISYMMETRIC": true}
  })");
  const auto runtime = nlohmann::json::parse(R"({
    "dim": 2,
    "num_equations": 4
  })");

  AxisymmetryConfig::Validate(config, runtime, true);
  return 0;
}

TEST(axisymmetry_build_and_case_must_agree)
{
  const auto axis_config = nlohmann::json::parse(R"({
    "compileTime": {"AXISYMMETRIC": true}
  })");
  const auto cartesian_config = nlohmann::json::parse(R"({
    "compileTime": {"AXISYMMETRIC": false}
  })");
  const auto runtime = nlohmann::json::object();

  EXPECT_TRUE(ConfigRejected(axis_config, runtime, false,
                             "without AXISYMMETRIC"));
  EXPECT_TRUE(ConfigRejected(cartesian_config, runtime, true,
                             "with AXISYMMETRIC"));
  return 0;
}

TEST(axisymmetric_layout_must_be_2d_and_swirl_free)
{
  const auto config = nlohmann::json::object();

  auto runtime = nlohmann::json::parse(R"({
    "dim": 3,
    "num_equations": 4
  })");
  EXPECT_TRUE(ConfigRejected(config, runtime, true, "dim = 2"));

  runtime["dim"] = 2;
  runtime["num_equations"] = 5;
  EXPECT_TRUE(ConfigRejected(config, runtime, true,
                             "num_equations = 4"));
  return 0;
}

TEST(cartesian_build_does_not_apply_axisymmetric_layout_restrictions)
{
  const auto config = nlohmann::json::object();
  const auto runtime = nlohmann::json::parse(R"({
    "dim": 3,
    "num_equations": 5
  })");

  AxisymmetryConfig::Validate(config, runtime, false);
  return 0;
}

TEST(axisymmetric_mesh_requires_two_dimensions)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(2);
  bool rejected = false;
  try
    {
      AxisymmetryConfig::ValidateMesh(mesh);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("dimension 2") !=
                 std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(axisymmetric_mesh_rejects_negative_radius)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mesh.GetVertex(0)[AxisymmetryConfig::radial_coordinate] = -0.25;

  bool rejected = false;
  try
    {
      AxisymmetryConfig::ValidateMesh(mesh);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("negative radial") !=
                 std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(axisymmetric_mesh_accepts_nonnegative_radius)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  AxisymmetryConfig::ValidateMesh(mesh);
  return 0;
}
