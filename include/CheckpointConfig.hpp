// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "json.hpp"

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace Theseus
{

  struct CheckpointCompatibility
  {
    int mpi_ranks;
    int order;
    int dimension;
    int num_equations;
    int real_bytes;
    long long global_elements;
    long long global_dofs;
    bool axisymmetric;
  };

  class CheckpointConfig
  {
  private:
    bool load_ = false;
    bool save_ = false;
    int cycle_ = 0;
    double interval_ = 0.01;
    std::filesystem::path directory_;

  public:
    static CheckpointConfig FromRuntime(const nlohmann::json &runtime,
                                        const std::filesystem::path &output_directory)
    {
      CheckpointConfig result;
      result.load_ = runtime.value("checkpoint_load", false);
      result.save_ = runtime.value("checkpoint_save", false);
      result.cycle_ = runtime.value("checkpoint_cycle", 0);
      result.interval_ = runtime.value("checkpoint_dt", 0.01);

      const auto folder = runtime.value("checkpoints_folder", std::string("Checkpoints"));
      if (folder.empty())
        {
          throw std::invalid_argument("runTime.checkpoints_folder must not be empty");
        }
      result.directory_ = output_directory / folder;

      if ((result.load_ || result.save_) && result.interval_ <= 0.0)
        {
          throw std::invalid_argument("runTime.checkpoint_dt must be greater than zero");
        }
      if (result.load_ && result.cycle_ <= 0)
        {
          throw std::invalid_argument(
            "runTime.checkpoint_cycle must be greater than zero when checkpoint_load is true");
        }
      return result;
    }

    bool LoadEnabled() const { return load_; }
    bool SaveEnabled() const { return save_; }
    int Cycle() const { return cycle_; }
    double Interval() const { return interval_; }
    const std::filesystem::path &Directory() const { return directory_; }

    std::filesystem::path CycleDirectory(int cycle) const
    {
      return directory_ / ("Cycle" + std::to_string(cycle));
    }

    std::filesystem::path MetadataFile(int cycle) const
    {
      return CycleDirectory(cycle) /
             ("checkpoint_cycle_" + std::to_string(cycle) + ".json");
    }

    std::filesystem::path RankFile(int cycle, int rank) const
    {
      std::ostringstream filename;
      filename << "checkpoint_cycle_" << cycle << "."
               << std::setw(8) << std::setfill('0') << rank << ".chk";
      return CycleDirectory(cycle) / filename.str();
    }

    static nlohmann::json Metadata(double time, int cycle,
                                   const CheckpointCompatibility &compatibility)
    {
      return {{"format_version", 2},
              {"state_format", "raw_vector_v1"},
              {"state_representation", "conservative"},
              {"geometry", compatibility.axisymmetric ? "axisymmetric" : "cartesian"},
              {"real_bytes", compatibility.real_bytes},
              {"time", time},
              {"cycle", cycle},
              {"mpi_ranks", compatibility.mpi_ranks},
              {"order", compatibility.order},
              {"dimension", compatibility.dimension},
              {"num_equations", compatibility.num_equations},
              {"global_elements", compatibility.global_elements},
              {"global_dofs", compatibility.global_dofs}};
    }

    // Returns false for legacy metadata, which contains only time and cycle.
    static bool ValidateMetadata(const nlohmann::json &metadata,
                                 const CheckpointCompatibility &expected)
    {
      if (!metadata.contains("format_version"))
        {
          if (expected.axisymmetric)
            {
              throw std::invalid_argument(
                "Legacy checkpoint cannot establish axisymmetric conservative-state semantics");
            }
          return false;
        }
      const int format_version = metadata.value("format_version", 0);
      if (format_version != 1 && format_version != 2)
        {
          throw std::invalid_argument("Unsupported checkpoint metadata format version");
        }
      if (metadata.value("state_format", std::string()) != "raw_vector_v1")
        {
          throw std::invalid_argument("Unsupported checkpoint state format");
        }
      if (format_version == 1 && expected.axisymmetric)
        {
          throw std::invalid_argument(
            "Version 1 checkpoint cannot establish whether an axisymmetric state stores U or rU");
        }
      if (format_version == 2)
        {
          if (metadata.value("state_representation", std::string()) != "conservative")
            {
              throw std::invalid_argument("Unsupported checkpoint state representation");
            }
          const std::string expected_geometry =
            expected.axisymmetric ? "axisymmetric" : "cartesian";
          if (metadata.value("geometry", std::string()) != expected_geometry)
            {
              throw std::invalid_argument(
                "Checkpoint is incompatible with the current run: geometry differs");
            }
        }
      if (metadata.value("real_bytes", 0) != expected.real_bytes)
        {
          throw std::invalid_argument("Checkpoint floating-point representation differs");
        }

      const auto require_equal = [&metadata](const char *name, long long value)
      {
        if (!metadata.contains(name) || metadata.at(name).get<long long>() != value)
          {
            throw std::invalid_argument(
              "Checkpoint is incompatible with the current run: " + std::string(name) +
              " differs");
          }
      };
      require_equal("mpi_ranks", expected.mpi_ranks);
      require_equal("order", expected.order);
      require_equal("dimension", expected.dimension);
      require_equal("num_equations", expected.num_equations);
      require_equal("global_elements", expected.global_elements);
      require_equal("global_dofs", expected.global_dofs);
      return true;
    }
  };

}
