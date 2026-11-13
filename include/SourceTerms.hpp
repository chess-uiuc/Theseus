// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "mfem.hpp"
#include "GasState.hpp"
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Theseus::Sources
{
  struct Requirements
  {
    bool coordinates = false;
  };

  struct MeshView
  {
    int elements = 0;
    int dofs = 0;
    int equations = 0;
    int dimension = 0;
    Theseus::StateLayout layout;
    const mfem::real_t *coordinates = nullptr;
    const mfem::real_t *weights = nullptr;
    const int *element_attributes = nullptr;
    const int *attribute_marker = nullptr;
    MPI_Comm communicator = MPI_COMM_NULL;
  };

  // This is the only mesh metadata copied into device kernels.
  struct DeviceMesh
  {
    int elements;
    int dofs;
    int equations;
    int dimension;
    Theseus::StateLayout layout;
    const mfem::real_t *coordinates;
    const int *element_attributes;
    const int *attribute_marker;
  };

  struct Stage
  {
    mfem::real_t time = 0.0;
    mfem::real_t step_size = 0.0;
    const mfem::real_t *state = nullptr;
  };

  struct Point
  {
    DeviceMesh mesh;
    Stage stage;
    int element;
    int node;

    MFEM_HOST_DEVICE mfem::real_t State(int equation) const
    {
      return stage.state[(element*mesh.equations + equation)*mesh.dofs + node];
    }

    MFEM_HOST_DEVICE mfem::real_t Coordinate(int direction) const
    {
      return mesh.coordinates[(element*mesh.dofs + node)*mesh.dimension + direction];
    }
  };

  // Adds physical source terms to the conservative time derivative at one node.
  struct SourceAccumulator
  {
    mfem::real_t *element_rhs;
    int dofs;
    int node;

    MFEM_HOST_DEVICE void Add(int equation, mfem::real_t source_term) const
    {
      element_rhs[equation*dofs + node] += source_term;
    }
  };

  // Host-only lifecycle. No object or virtual function is captured by a kernel.
  class Source
  {
  public:
    virtual ~Source() = default;
    virtual Requirements Needs() const = 0;
    virtual void Initialize(const MeshView &mesh) = 0;
    virtual void Prepare(const Stage &stage) = 0;
    virtual void Add(const Stage &stage, mfem::real_t *rhs) const = 0;
  };

  // Stateless analytic models inherit these optional no-op lifecycle methods.
  struct AnalyticModel
  {
    Requirements Needs() const { return {}; }
    void Initialize(const MeshView &) {}
    void Prepare(const Stage &) {}
  };

  template<typename Model>
  class PointSource final : public Source
  {
    Model model;
    MeshView mesh;
  public:
    explicit PointSource(Model value) : model(std::move(value)) {}
    Requirements Needs() const override { return model.Needs(); }
    void Initialize(const MeshView &view) override
    {
      mesh = view;
      if (Needs().coordinates && mesh.elements > 0 && !mesh.coordinates)
        {
          throw std::invalid_argument("Source requires device coordinates");
        }
      model.Initialize(mesh);
    }
    void Prepare(const Stage &stage) override { model.Prepare(stage); }
    void Add(const Stage &stage, mfem::real_t *rhs) const override
    {
      // The concrete evaluator contains values/device views, not the host owner.
      const auto evaluator = model.MakeEvaluator(stage);
      static_assert(std::is_trivially_copyable_v<decltype(evaluator)>);
      DeviceMesh view{mesh.elements, mesh.dofs, mesh.equations, mesh.dimension,
                      mesh.layout, mesh.coordinates, mesh.element_attributes, mesh.attribute_marker};
      const int count = view.elements*view.dofs;
      if (count == 0) { return; }
      mfem::forall(count, [=] MFEM_HOST_DEVICE (int index)
      {
        const int element = index/view.dofs;
        if (view.attribute_marker && !view.attribute_marker[view.element_attributes[element] - 1])
          {
            return;
          }
        Point point{view, stage, element, index % view.dofs};
        SourceAccumulator output{rhs + element*view.equations*view.dofs,
                               view.dofs, index % view.dofs};
        evaluator(point, output);
      });
    }
  };

  class System
  {
    std::vector<std::unique_ptr<Source>> sources;
    bool initialized = false;
  public:
    bool Empty() const { return sources.empty(); }
    int Size() const { return int(sources.size()); }
    void Add(std::unique_ptr<Source> source)
    {
      if (initialized || !source)
        {
          throw std::invalid_argument("Add sources before initialization");
        }
      sources.push_back(std::move(source));
    }
    Requirements Needs() const
    {
      Requirements result;
      for (const auto &source : sources)
        {
          result.coordinates |= source->Needs().coordinates;
        }
      return result;
    }
    void Initialize(const MeshView &mesh)
    {
      if (mesh.elements < 0 || mesh.dofs <= 0 || mesh.dimension < 1
          || mesh.dimension > Theseus::MAXDIM || mesh.equations < mesh.dimension + 2
          || mesh.layout.dim != mesh.dimension
          || (mesh.attribute_marker && !mesh.element_attributes))
        {
          throw std::invalid_argument("Invalid source mesh dimensions");
        }
      for (auto &source : sources)
        {
          source->Initialize(mesh);
        }
      initialized = true;
    }
    void AddStage(const Stage &stage, mfem::real_t *rhs)
    {
      if (sources.empty()) { return; }
      if (!initialized)
        {
          throw std::logic_error("Initialize sources before evaluation");
        }
      for (auto &source : sources)
        {
          source->Prepare(stage);
          source->Add(stage, rhs);
        }
    }
  };
}
