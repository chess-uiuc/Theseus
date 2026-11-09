// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>

namespace Theseus::Debug
{
  // Host-only observer. Never pass this object into a device lambda/cache.
  // Set THESEUS_ARRAY_DEBUG to a JSON configuration file to enable observation.
  class ArrayInspector
  {
  public:
    ArrayInspector();
    ~ArrayInspector();
    ArrayInspector(const ArrayInspector &) = delete;
    ArrayInspector &operator=(const ArrayInspector &) = delete;

    void BeginCall(MPI_Comm comm, double time, int dimension, int equations,
                   int order, const std::string &gas, const std::string &flux);
    void Record(const std::string &name, const mfem::Vector &values,
                std::initializer_list<const char *> axes,
                std::initializer_list<std::int64_t> shape);
    void EndCall();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
  };
}
