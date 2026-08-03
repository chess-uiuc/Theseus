// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "CheckpointConfig.hpp"
#include "RunControl.hpp"
#include "unit_test.hpp"

#include <stdexcept>
#include <string>

using Theseus::CheckpointCompatibility;
using Theseus::CheckpointConfig;

TEST(checkpoints_default_to_disabled)
{
  const auto config = CheckpointConfig::FromRuntime(nlohmann::json::object(), "output");

  EXPECT_TRUE(!config.LoadEnabled());
  EXPECT_TRUE(!config.SaveEnabled());
  EXPECT_CLOSE(config.Interval(), 0.01, 1e-15);
  EXPECT_TRUE(config.Directory() == std::filesystem::path("output/Checkpoints"));
  return 0;
}

TEST(checkpoint_paths_are_constructed_consistently)
{
  const auto runtime = nlohmann::json::parse(R"({
    "checkpoint_load": true,
    "checkpoint_save": true,
    "checkpoint_cycle": 12,
    "checkpoint_dt": 0.25,
    "checkpoints_folder": "restart-data"
  })");
  const auto config = CheckpointConfig::FromRuntime(runtime, "results");

  EXPECT_TRUE(config.LoadEnabled());
  EXPECT_TRUE(config.SaveEnabled());
  EXPECT_EQ(config.Cycle(), 12);
  EXPECT_TRUE(config.MetadataFile(12) ==
              std::filesystem::path("results/restart-data/Cycle12/checkpoint_cycle_12.json"));
  EXPECT_TRUE(config.RankFile(12, 3) ==
              std::filesystem::path(
                "results/restart-data/Cycle12/checkpoint_cycle_12.00000003.chk"));
  return 0;
}

TEST(checkpoint_loading_requires_a_positive_cycle)
{
  bool rejected = false;
  try
    {
      CheckpointConfig::FromRuntime({{"checkpoint_load", true}}, ".");
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("checkpoint_cycle") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(enabled_checkpoints_require_a_positive_interval)
{
  bool rejected = false;
  try
    {
      CheckpointConfig::FromRuntime(
        {{"checkpoint_save", true}, {"checkpoint_dt", 0.0}}, ".");
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("checkpoint_dt") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(current_checkpoint_metadata_validates_compatibility)
{
  const CheckpointCompatibility expected{2, 3, 2, 4, 8, 576, 9216, false};
  const auto metadata = CheckpointConfig::Metadata(0.1, 10, expected);

  EXPECT_EQ(metadata.at("format_version").get<int>(), 2);
  EXPECT_TRUE(metadata.at("state_representation") == "conservative");
  EXPECT_TRUE(metadata.at("geometry") == "cartesian");
  EXPECT_TRUE(CheckpointConfig::ValidateMetadata(metadata, expected));
  return 0;
}

TEST(checkpoint_metadata_rejects_an_incompatible_mpi_layout)
{
  const CheckpointCompatibility written{2, 3, 2, 4, 8, 576, 9216, false};
  const CheckpointCompatibility current{4, 3, 2, 4, 8, 576, 9216, false};
  bool rejected = false;
  try
    {
      CheckpointConfig::ValidateMetadata(
        CheckpointConfig::Metadata(0.1, 10, written), current);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("mpi_ranks") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(legacy_checkpoint_metadata_remains_loadable)
{
  const CheckpointCompatibility expected{2, 3, 2, 4, 8, 576, 9216, false};
  const nlohmann::json legacy{{"time", 0.1}, {"cycle", 10}};

  EXPECT_TRUE(!CheckpointConfig::ValidateMetadata(legacy, expected));
  return 0;
}

TEST(axisymmetric_checkpoint_records_conservative_state_semantics)
{
  const CheckpointCompatibility expected{2, 3, 2, 4, 8, 576, 9216, true};
  const auto metadata = CheckpointConfig::Metadata(0.1, 10, expected);

  EXPECT_TRUE(metadata.at("state_representation") == "conservative");
  EXPECT_TRUE(metadata.at("geometry") == "axisymmetric");
  EXPECT_TRUE(CheckpointConfig::ValidateMetadata(metadata, expected));
  return 0;
}

TEST(axisymmetric_restart_rejects_ambiguous_ru_checkpoint_metadata)
{
  const CheckpointCompatibility expected{2, 3, 2, 4, 8, 576, 9216, true};
  auto version_one = CheckpointConfig::Metadata(0.1, 10, expected);
  version_one["format_version"] = 1;
  version_one.erase("state_representation");
  version_one.erase("geometry");

  bool rejected = false;
  try
    {
      CheckpointConfig::ValidateMetadata(version_one, expected);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("U or rU") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(checkpoint_geometry_must_match_current_run)
{
  const CheckpointCompatibility cartesian{2, 3, 2, 4, 8, 576, 9216, false};
  const CheckpointCompatibility axisymmetric{2, 3, 2, 4, 8, 576, 9216, true};

  bool rejected = false;
  try
    {
      CheckpointConfig::ValidateMetadata(
        CheckpointConfig::Metadata(0.1, 10, cartesian), axisymmetric);
    }
  catch (const std::invalid_argument &error)
    {
      rejected = std::string(error.what()).find("geometry") != std::string::npos;
    }
  EXPECT_TRUE(rejected);
  return 0;
}

TEST(step_limit_is_disabled_by_a_negative_sentinel)
{
  EXPECT_TRUE(!Theseus::StepLimitReached(1, -1));
  EXPECT_TRUE(!Theseus::StepLimitReached(199, 200));
  EXPECT_TRUE(Theseus::StepLimitReached(200, 200));
  EXPECT_TRUE(Theseus::StepLimitReached(201, 200));
  return 0;
}
