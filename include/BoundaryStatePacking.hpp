// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "GasModel.hpp"
#include "RadialProfile.hpp"
#include "bc_cache_utilities.hpp"

#include <sstream>

namespace Theseus
{
  inline void PackBoundaryPointStates(const mfem::Array<BCDescriptor> &descriptors,
                                      const mfem::Vector &input,
                                      const std::vector<std::string> &names,
                                      const mfem::Array<int> &markers, int points_per_face,
                                      const mfem::Vector &coordinates, const GasModelInterface &gas,
                                      mfem::Array<BCDescriptor> &point_descriptors,
                                      mfem::Vector &output)
  {
    struct Profile
    {
      RadialProfile samples;
      mfem::real_t pressure = 0;
      mfem::real_t gamma = 0;
      mfem::real_t gas_constant = 0;
      bool legacy = false;
    };

    std::vector<Profile> profiles(descriptors.Size());
    for (int marker = 0; marker < descriptors.Size(); ++marker)
      {
        const auto &bc = descriptors[marker];
        const bool legacy = bc.data_kind == int(BCDataKind::RadialCPG);
        if (!legacy && bc.data_kind != int(BCDataKind::RadialPhysical))
          {
            continue;
          }

        const auto *data = input.HostRead() + bc.data_index;
        auto &profile = profiles[marker];
        profile.pressure = data[0];
        profile.legacy = legacy;
        if (legacy)
          {
            profile.gamma = data[1];
            profile.gas_constant = data[2];
          }

        const int header_size = legacy ? 4 : 2;
        const int count = int(data[header_size - 1]);
        for (int i = 0; i < count; ++i)
          {
            profile.samples.rows.push_back({data[header_size + 4 * i], data[header_size + 4 * i + 1],
                                            data[header_size + 4 * i + 2], data[header_size + 4 * i + 3]});
          }
      }

    const int dim = gas.layout().dim;
    const int points = markers.Size() * points_per_face;
    const auto *xyz = coordinates.HostRead();
    point_descriptors.SetSize(points);
    output = input;

    for (int point = 0; point < points; ++point)
      {
        const int marker = markers[point / points_per_face];
        BCDescriptor bc{};
        bc.type = int(BCType::Invalid);
        if (marker >= 0)
          {
            bc = descriptors[marker];
          }
        if (bc.data_kind == int(BCDataKind::RadialCPG) ||
            bc.data_kind == int(BCDataKind::RadialPhysical))
          {
            const auto radius = xyz[point * dim + 1];
            try
              {
                const auto &profile = profiles[marker];
                const auto sample = profile.samples.Evaluate(radius);
                const auto temperature = sample[1];
                const auto axial_velocity = sample[2];
                const auto radial_velocity = sample[3];

                mfem::Vector state;
                if (profile.legacy)
                  {
                    const auto values = CPGState(profile.pressure, temperature, axial_velocity,
                                                 radial_velocity, profile.gamma, profile.gas_constant);
                    state.SetSize(4);
                    for (int q = 0; q < 4; ++q)
                      {
                        state[q] = values[q];
                      }
                  }
                else
                  {
                    PhysicalStateInput input_state;
                    input_state.thermo = PressureTemperature{profile.pressure, temperature};
                    input_state.velocity = {axial_velocity, radial_velocity};
                    gas.ConservativeFromPhysical(input_state, state);
                  }

                bc.data_index = AppendBCVectorPayload(output, state);
                bc.data_kind = int(BCDataKind::VectorConstant);
              }
            catch (const std::exception &error)
              {
                const auto name =
                    marker < int(names.size()) ? names[marker] : std::to_string(marker);
                std::ostringstream message;
                message << "Boundary '" << name << "' at radius " << radius
                        << " (restriction point " << point << "): " << error.what();
                throw std::invalid_argument(message.str());
              }
          }
        point_descriptors[point] = bc;
      }
  }
} // namespace Theseus
