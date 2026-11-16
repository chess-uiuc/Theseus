// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ConditionFactory.hpp"
#include "verification/mms/sources/EulerMMS.hpp"
#include "verification/mms/sources/TrigonometricEuler.hpp"
#include "verification/mms/sources/TrigonometricCNS.hpp"
#include "verification/mms/sources/TrigonometricSutherland.hpp"
#include <cmath>

namespace Theseus::Sources
{
  inline void RegisterBuiltinManufacturedSources(Registry &registry)
  {
    RegisterMMS(registry);
    RegisterTrigonometricEuler(registry);
    RegisterTrigonometricCNS(registry);
    RegisterTrigonometricSutherland(registry);
  }

  inline void RegisterManufacturedStates(Prandtl::ConditionFactory &factory)
  {
    factory.RegisterInitialCondition1("TemporalTrigonometricEulerMMS", [](mfem::real_t time_scale)
    {
      MFEM_VERIFY(std::isfinite(time_scale) && time_scale > 0., "MMS time_scale must be positive and finite");
      return [time_scale](const mfem::Vector &position, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && state.Size() == position.Size() + 2, "Temporal MMS requires 2D/3D state");
        mfem::real_t forcing[5];
        Theseus::Sources::TrigonometricEulerMMS(position.Size(), position.GetData(),
                                              0.0, state.HostWrite(), forcing, time_scale);
      };
    });
    factory.RegisterVectorTDFunctionBoundaryCondition1("TemporalTrigonometricEulerMMS", [](mfem::real_t time_scale)
    {
      MFEM_VERIFY(std::isfinite(time_scale) && time_scale > 0., "MMS time_scale must be positive and finite");
      return [time_scale](const mfem::Vector &position, mfem::real_t time, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && state.Size() == position.Size() + 2, "Temporal MMS requires 2D/3D state");
        mfem::real_t forcing[5];
        Theseus::Sources::TrigonometricEulerMMS(position.Size(), position.GetData(),
                                              time, state.HostWrite(), forcing, time_scale);
      };
    });
    factory.RegisterInitialCondition1("OrientedTrigonometricEulerMMS", [](mfem::real_t value)
    {
      MFEM_VERIFY(std::isfinite(value) && value == std::floor(value) && value >= 0 && value < 3,
                  "MMS orientation must be an integer axis permutation");
      const int orientation = int(value);
      return [orientation](const mfem::Vector &position, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && orientation < position.Size() && state.Size() == position.Size() + 2,
                    "Oriented MMS requires compatible dimension and conservative state");
        mfem::real_t forcing[5];
        Theseus::Sources::OrientedTrigonometricEulerMMS(position.Size(), orientation,
          position.GetData(), 0.0, state.HostWrite(), forcing);
      };
    });
    factory.RegisterVectorTDFunctionBoundaryCondition1("OrientedTrigonometricEulerMMS", [](mfem::real_t value)
    {
      MFEM_VERIFY(std::isfinite(value) && value == std::floor(value) && value >= 0 && value < 3,
                  "MMS orientation must be an integer axis permutation");
      const int orientation = int(value);
      return [orientation](const mfem::Vector &position, mfem::real_t time, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && orientation < position.Size() && state.Size() == position.Size() + 2,
                    "Oriented MMS requires compatible dimension and conservative state");
        mfem::real_t forcing[5];
        Theseus::Sources::OrientedTrigonometricEulerMMS(position.Size(), orientation,
          position.GetData(), time, state.HostWrite(), forcing);
      };
    });
    factory.RegisterInitialCondition0("TrigonometricEulerMMS", []()
    {
      return [](const mfem::Vector &position, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && state.Size() == position.Size() + 2,
                    "TrigonometricEulerMMS requires 2D/3D conservative state");
        mfem::real_t forcing[5];
        Theseus::Sources::TrigonometricEulerMMS(
          position.Size(), position.GetData(), 0.0, state.HostWrite(), forcing);
      };
    });
    factory.RegisterVectorTDFunctionBoundaryCondition0("TrigonometricEulerMMS", []()
    {
      return [](const mfem::Vector &position, mfem::real_t time, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && state.Size() == position.Size() + 2,
                    "TrigonometricEulerMMS requires 2D/3D conservative state");
        mfem::real_t forcing[5];
        Theseus::Sources::TrigonometricEulerMMS(
          position.Size(), position.GetData(), time, state.HostWrite(), forcing);
      };
    });
    factory.RegisterInitialCondition0("TrigonometricCNSMMS", []()
    {
      return [](const mfem::Vector &position, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && state.Size() == position.Size() + 2,
                    "TrigonometricEulerMMS requires 2D/3D conservative state");
        mfem::real_t forcing[5];
        Theseus::Sources::TrigonometricEulerMMS(
          position.Size(), position.GetData(), 0.0, state.HostWrite(), forcing);
      };
    });
    factory.RegisterVectorTDFunctionBoundaryCondition0("TrigonometricCNSMMS", []()
    {
      return [](const mfem::Vector &position, mfem::real_t time, mfem::Vector &state)
      {
        MFEM_VERIFY((position.Size() == 2 || position.Size() == 3)
                    && state.Size() == position.Size() + 2,
                    "TrigonometricEulerMMS requires 2D/3D conservative state");
        mfem::real_t forcing[5];
        Theseus::Sources::TrigonometricEulerMMS(
          position.Size(), position.GetData(), time, state.HostWrite(), forcing);
      };
    });
    factory.RegisterInitialCondition0("PeriodicEulerMMS", []()
    {
      return [](const mfem::Vector &position, mfem::Vector &state)
      {
        MFEM_VERIFY(position.Size() == 2 && state.Size() == 4,
                    "PeriodicEulerMMS requires 2D and four equations");
        mfem::real_t forcing[4];
        Theseus::Sources::PeriodicEulerMMS(position[0], position[1], 0.0, state.HostWrite(), forcing);
      };
    });
    factory.RegisterVectorTDFunctionBoundaryCondition0("PeriodicEulerMMS", []()
    {
      return [](const mfem::Vector &position, mfem::real_t time, mfem::Vector &state)
      {
        MFEM_VERIFY(position.Size() == 2 && state.Size() == 4,
                    "PeriodicEulerMMS requires 2D and four equations");
        mfem::real_t forcing[4];
        Theseus::Sources::PeriodicEulerMMS(position[0], position[1], time, state.HostWrite(), forcing);
      };
    });
  }
}
