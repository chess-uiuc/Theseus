// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "SourceRegistry.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace Theseus::Sources
{
  struct ChannelVolumeAverages
  {
    mfem::real_t density = 0.0;
    mfem::real_t streamwise_momentum = 0.0;
    mfem::real_t internal_energy_density = 0.0;
  };

  class ChannelFeedbackModel
  {
    static constexpr int reduction_width = 256;
    MeshView mesh;
    std::vector<mfem::Vector> integrals;
    std::vector<int> element_counts;
    mfem::real_t total_volume = 0.0;
    mfem::real_t target_density;
    mfem::real_t target_streamwise_momentum;
    mfem::real_t relaxation_factor;
    mfem::real_t density_source = 0.0;
    mfem::real_t streamwise_momentum_source = 0.0;
    ChannelVolumeAverages volume_averages;

    const mfem::real_t *LocalIntegrals()
    {
      for (int level = 1; level < int(integrals.size()); ++level)
        {
          const int input_count = element_counts[level - 1];
          const int output_count = element_counts[level];
          const auto *input = integrals[level - 1].Read();
          auto *output = integrals[level].Write();
          mfem::forall_2D(3*output_count, reduction_width, 1,
            [=] MFEM_HOST_DEVICE (int index)
          {
            MFEM_SHARED mfem::real_t partial_sum[reduction_width];
            const int component = index/output_count;
            const int begin = (index % output_count)*reduction_width;
            MFEM_FOREACH_THREAD(lane, x, reduction_width)
              {
                const int entry = begin + lane;
                partial_sum[lane] = entry < input_count
                  ? input[component*input_count + entry] : 0.0;
              }
            MFEM_SYNC_THREAD;
            for (int stride = reduction_width/2; stride > 0; stride /= 2)
              {
                MFEM_FOREACH_THREAD(lane, x, stride)
                  {
                    partial_sum[lane] += partial_sum[lane + stride];
                  }
                MFEM_SYNC_THREAD;
              }
            MFEM_FOREACH_THREAD(lane, x, 1)
              {
                output[index] = partial_sum[0];
              }
          });
        }
      // Only three local sums are read back; all element sums stay on device.
      return integrals.back().HostRead();
    }

  public:
    ChannelFeedbackModel(const nlohmann::json &configuration, const BuildContext &context)
      : target_density(configuration.value("target_density", 1.0)),
        target_streamwise_momentum(configuration.value("target_streamwise_momentum", 515.68342221905391)),
        relaxation_factor(configuration.value("relaxation_factor", 0.7))
    {
      if (context.dimension != 3 || context.equations < 5 || context.axisymmetric)
        {
          throw std::invalid_argument("Channel feedback requires Cartesian 3D conservative state");
        }
      if (!std::isfinite(target_density) || target_density <= 0.0
          || !std::isfinite(target_streamwise_momentum)
          || !std::isfinite(relaxation_factor) || relaxation_factor <= 0.0)
        {
          throw std::invalid_argument("Invalid channel feedback targets or relaxation factor");
        }
    }

    Requirements Needs() const { return {}; }
    const ChannelVolumeAverages &Averages() const { return volume_averages; }
    mfem::real_t Volume() const { return total_volume; }

    void Initialize(const MeshView &view)
    {
      if (view.dimension != 3 || view.equations < 5 || view.elements < 0
          || view.dofs <= 0 || view.communicator == MPI_COMM_NULL
          || (view.elements > 0 && !view.weights))
        {
          throw std::invalid_argument("Invalid channel feedback mesh/weights/communicator");
        }
      mesh = view;
      element_counts.clear();
      integrals.clear();
      int count = mesh.elements > 0 ? mesh.elements : 1;
      element_counts.push_back(count);
      while (count > 1)
        {
          count = (count + reduction_width - 1)/reduction_width;
          element_counts.push_back(count);
        }
      integrals.reserve(element_counts.size());
      for (int size : element_counts)
        {
          integrals.emplace_back(3*size);
          integrals.back().UseDevice();
          integrals.back() = 0.0;
          integrals.back().Read();
        }
      integrals.back().HostRead();
      integrals.back().Read();
      const int elements = mesh.elements;
      const int dofs = mesh.dofs;
      const auto *weights = mesh.weights;
      auto *volume = integrals.front().ReadWrite();
      mfem::forall(elements, [=] MFEM_HOST_DEVICE (int element)
      {
        mfem::real_t sum = 0.0;
        for (int point = 0; point < dofs; ++point)
          {
            sum += weights[element*dofs + point];
          }
        volume[element] = sum;
      });
      const auto *local_volume = LocalIntegrals();
      MPI_Allreduce(local_volume, &total_volume, 1,
        mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, mesh.communicator);
      if (!std::isfinite(total_volume) || total_volume <= 0.0)
        {
          throw std::invalid_argument("Channel feedback requires positive finite global volume");
        }
      int rank = 0;
      MPI_Comm_rank(mesh.communicator, &rank);
      if (rank == 0)
        {
          std::cout << "Research channel forcing: enabled (tau=" << relaxation_factor
                    << "*step_dt)" << std::endl;
        }
    }

    void Prepare(const Stage &stage)
    {
      if (!std::isfinite(stage.step_size) || stage.step_size <= 0.0)
        {
          throw std::invalid_argument("Channel feedback requires a positive finite step size");
        }
      if (integrals.empty()) { throw std::logic_error("Initialize channel feedback before preparation"); }
      const int elements = mesh.elements;
      const int dofs = mesh.dofs;
      const int equations = mesh.equations;
      const auto layout = mesh.layout;
      const auto *weights = mesh.weights;
      const auto *state = stage.state;
      auto *element_integrals = integrals.front().Write();
      if (elements == 0)
        {
          mfem::forall(3, [=] MFEM_HOST_DEVICE (int i) { element_integrals[i] = 0.0; });
        }
      mfem::forall(elements, [=] MFEM_HOST_DEVICE (int element)
      {
        const auto *u = state + element*dofs*equations;
        mfem::real_t mass_integral = 0.0;
        mfem::real_t momentum_integral = 0.0;
        mfem::real_t internal_energy_integral = 0.0;
        for (int point = 0; point < dofs; ++point)
          {
            const auto rho = u[layout.eq_mass*dofs + point];
            mfem::real_t momentum_squared = 0.0;
            for (int d = 0; d < 3; ++d)
              {
                const auto momentum = u[layout.eq_mom[d]*dofs + point];
                momentum_squared += momentum*momentum;
              }
            const auto internal_energy = u[layout.eq_energy*dofs + point]
              - 0.5*momentum_squared/rho;
            const auto weight = weights[element*dofs + point];
            mass_integral += rho*weight;
            momentum_integral += u[layout.eq_mom0*dofs + point]*weight;
            internal_energy_integral += internal_energy*weight;
          }
        element_integrals[element] = mass_integral;
        element_integrals[elements + element] = momentum_integral;
        element_integrals[2*elements + element] = internal_energy_integral;
      });
      const auto *local = LocalIntegrals();
      mfem::real_t global[3];
      MPI_Allreduce(local, global, 3, mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_SUM, mesh.communicator);
      volume_averages.density = global[0]/total_volume;
      volume_averages.streamwise_momentum = global[1]/total_volume;
      volume_averages.internal_energy_density = global[2]/total_volume;
      const auto inverse_tau = 1.0/(relaxation_factor*stage.step_size);
      density_source = (target_density - volume_averages.density)*inverse_tau;
      streamwise_momentum_source =
        (target_streamwise_momentum - volume_averages.streamwise_momentum)*inverse_tau;
    }

    struct Evaluator
    {
      mfem::real_t density_source;
      mfem::real_t streamwise_momentum_source;
      MFEM_HOST_DEVICE void operator()(const Point &point, const SourceAccumulator &source) const
      {
        const auto &layout = point.mesh.layout;
        const auto streamwise_velocity = point.State(layout.eq_mom0)/point.State(layout.eq_mass);
        source.Add(layout.eq_mass, density_source);
        source.Add(layout.eq_mom0, streamwise_momentum_source);
        source.Add(layout.eq_energy, streamwise_velocity*streamwise_momentum_source);
      }
    };
    Evaluator MakeEvaluator(const Stage &) const
    {
      return {density_source, streamwise_momentum_source};
    }
  };

  inline void RegisterChannelFeedback(Registry &registry)
  {
    registry.Register("channel-feedback", [](const nlohmann::json &configuration, const BuildContext &context)
    {
      return std::make_unique<PointSource<ChannelFeedbackModel>>(ChannelFeedbackModel(configuration, context));
    });
  }

  // Research compatibility only; an explicit source list always takes precedence.
  inline nlohmann::json ChannelFeedbackConfiguration(const nlohmann::json &runtime)
  {
    auto configured = runtime;
    if (!configured.contains("sources"))
      {
        const auto initial = runtime.value("conditions", nlohmann::json::object())
          .value("initial_conditions", nlohmann::json::object());
        const bool enabled = runtime.value("channel_forcing",
          initial.value("function", std::string{}) == "StreamwiseParabolicIC");
        if (enabled)
          {
            configured["sources"] = nlohmann::json::array({{{"type", "channel-feedback"}}});
          }
      }
    return configured;
  }
}
