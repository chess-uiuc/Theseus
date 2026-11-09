// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "ArrayDebug.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  try
  {
    const std::string backend = argc > 1 ? argv[1] : "cpu";
    const std::string mode = argc > 2 ? argv[2] : "normal";
    mfem::Device device(backend);
    Theseus::Debug::ArrayInspector inspector;
    mfem::Vector values(12);
    values.UseDevice();
    auto *write = values.Write();
    mfem::forall(12, [=] MFEM_HOST_DEVICE (int i) { write[i] = i - 2.0; });
    const bool host_valid = values.GetMemory().HostIsValid();
    const bool device_valid = values.GetMemory().DeviceIsValid();
    inspector.BeginCall(MPI_COMM_WORLD, 0.25, 3, 3, 2, "synthetic", "fixture");
    inspector.Record("initial", values, {"element", "equation", "point"}, {2, 3, 2});
    if (host_valid != values.GetMemory().HostIsValid() ||
        device_valid != values.GetMemory().DeviceIsValid())
    {
      throw std::runtime_error("observation changed source memory validity");
    }
    // Reuse the already acquired write pointer, as the solver's volume path does.
    mfem::forall(12, [=] MFEM_HOST_DEVICE (int i) { write[i] += 10.0; });
    inspector.Record("updated", values, {"element", "equation", "point"}, {2, 3, 2});
    if (mode == "shape")
    {
      inspector.Record("bad", values, {"point"}, {13});
    }
    if (mode == "nonfinite")
    {
      auto *host = values.HostReadWrite();
      host[1] = std::numeric_limits<mfem::real_t>::quiet_NaN();
      host[7] = std::numeric_limits<mfem::real_t>::infinity();
      host[11] = -std::numeric_limits<mfem::real_t>::infinity();
      inspector.Record("invalid", values, {"element", "equation", "point"}, {2, 3, 2});
    }
    mfem::Vector empty;
    inspector.Record("empty", empty, {"point"}, {0});
    inspector.EndCall();
    // Call selection should suppress this entire call with default test config.
    inspector.BeginCall(MPI_COMM_WORLD, 0.5, 3, 3, 2, "synthetic", "fixture");
    inspector.Record("unselected", values, {"point"}, {12});
    inspector.EndCall();
    if (mode != "nonfinite")
    {
      const auto *host = values.HostRead();
      for (int i = 0; i < 12; ++i)
      {
        if (host[i] != i + 8.0)
        {
          throw std::runtime_error("observer changed values");
        }
      }
    }
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
