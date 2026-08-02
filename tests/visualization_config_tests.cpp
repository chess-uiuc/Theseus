// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "unit_test.hpp"

#include "VisualizationConfig.hpp"

#include <stdexcept>
#include <string>

using Theseus::VisualizationConfig;
using Theseus::VisualizationField;
using Theseus::VisualizationMeshMode;

TEST(visualization_fields_default_to_all_available_fields)
{
  const nlohmann::json runtime = nlohmann::json::object();
  const auto config = VisualizationConfig::FromRuntime(runtime, true);

  EXPECT_TRUE(config.Has(VisualizationField::density));
  EXPECT_TRUE(config.Has(VisualizationField::velocity));
  EXPECT_TRUE(config.Has(VisualizationField::pressure));
  EXPECT_TRUE(config.Has(VisualizationField::blending_coefficient));
  return 0;
}

TEST(visualization_fields_can_be_selected_and_duplicates_are_ignored)
{
  const auto runtime = nlohmann::json::parse(R"({
    "visualization": {"fields": ["pressure", "velocity", "pressure"]}
  })");
  const auto config = VisualizationConfig::FromRuntime(runtime, true);

  EXPECT_TRUE(!config.Has(VisualizationField::density));
  EXPECT_TRUE(config.Has(VisualizationField::velocity));
  EXPECT_TRUE(config.Has(VisualizationField::pressure));
  EXPECT_TRUE(!config.Has(VisualizationField::blending_coefficient));
  EXPECT_EQ(config.Fields().size(), std::size_t{2});
  return 0;
}

TEST(visualization_fields_reject_unknown_names)
{
  const auto runtime = nlohmann::json::parse(R"({
    "visualization": {"fields": ["temperature"]}
  })");

  bool rejected = false;
  try
    {
      VisualizationConfig::FromRuntime(runtime, true);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("temperature") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(visualization_fields_reject_unavailable_blending_field)
{
  const auto runtime = nlohmann::json::parse(R"({
    "visualization": {"fields": ["blending_coefficient"]}
  })");

  bool rejected = false;
  try
    {
      VisualizationConfig::FromRuntime(runtime, false);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("SUBCELL_FV_BLENDING") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(visualization_fields_require_a_nonempty_string_array)
{
  for (const auto &fields : {nlohmann::json::array(), nlohmann::json("density")})
    {
      nlohmann::json runtime;
      runtime["visualization"]["fields"] = fields;
      bool rejected = false;
      try
        {
          VisualizationConfig::FromRuntime(runtime, true);
        }
      catch (const std::invalid_argument &)
        {
          rejected = true;
        }
      EXPECT_TRUE(rejected);
    }
  return 0;
}

TEST(visualization_mesh_mode_defaults_to_gll_subcells)
{
  const nlohmann::json runtime = nlohmann::json::object();
  const auto config = VisualizationConfig::FromRuntime(runtime, true);

  EXPECT_TRUE(config.MeshMode() == VisualizationMeshMode::gll_subcells);
  return 0;
}

TEST(visualization_mesh_mode_can_select_gll_subcells)
{
  const auto runtime = nlohmann::json::parse(R"({
    "visualization": {"mesh_mode": "gll_subcells"}
  })");
  const auto config = VisualizationConfig::FromRuntime(runtime, true);

  EXPECT_TRUE(config.MeshMode() == VisualizationMeshMode::gll_subcells);
  return 0;
}

TEST(visualization_mesh_mode_rejects_unknown_values)
{
  const auto runtime = nlohmann::json::parse(R"({
    "visualization": {"mesh_mode": "regular_subcells"}
  })");

  bool rejected = false;
  try
    {
      VisualizationConfig::FromRuntime(runtime, true);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("regular_subcells") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}
