// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "Simulation.hpp"

int main(int argc, char* argv[])
{
  std::string config_file_path;
  std::string device_name;

  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
        
      if (arg == "-c")
        {
          if (i + 1 >= argc)
            {
              std::cerr << "Error: -c requires a configuration file path.\n";
              return 1;
            }
          config_file_path = argv[++i];  // consume next argument
        }
      else if (arg == "-d")
        {
          if (i + 1 >= argc)
            {
              std::cerr << "Error: -d requires a device name.\n";
              return 1;
            }
          device_name = argv[++i];  // consume next argument
        }
      else
        {
          std::cerr << "Error: Unknown argument: " << arg << "\n";
          return 1;
        }
    }

  if (config_file_path.empty())
    {
      std::cerr << "\nError: Configuration file not specified.\n";
      std::cerr << "Usage: " << argv[0]
                << " -c <path/to/config.json> [-d <device>]\n\n";
      return 1;
    }

  Theseus::Simulation &sim = Theseus::Simulation::SimulationCreate(device_name);
  if(sim.LoadConfig(config_file_path)){
    std::cerr << "Failed to load configuration file: " << config_file_path << std::endl;
    return 1;
  }

  sim.Run();
}
