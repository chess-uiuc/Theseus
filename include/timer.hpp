// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <chrono>
#include <cstdio>
#include <mpi.h>

namespace Theseus
{
  class ScopedTimer
  {
  public:
#ifdef ENABLE_TIMERS
    explicit ScopedTimer(const char *name)
      : name_(name),
        start_(clock::now()) {}

    ~ScopedTimer()
    {
      auto end = clock::now();
      double local_ms = std::chrono::duration<double, std::milli>(end - start_).count();
      double global_ms = local_ms;
      int rank = 0;
      int nranks = 1;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#ifdef TIMER_BARRIER
      MPI_Barrier(MPI_COMM_WORLD);
      end = clock::now();
      global_ms = std::chrono::duration<double, std::milli>(end - start_).count();
#endif
#ifdef TIMER_OUTPUT_ALLRANKS
      if(nranks > 1 && rank > 0)
        std::printf("[TIMER(%d)] %s : %.6f ms\n", rank, name_, local_ms);
#endif
      if(rank == 0 ){
        std::printf("[TIMER(%d)] %s : %.6f ms\n", rank, name_, local_ms);
#ifdef TIMER_BARRIER
        if(nranks > 1)
          std::printf("[TIMER(all)] %s : %.6f ms\n", name_, global_ms);
#endif
      }
    }
#else
    explicit ScopedTimer(const char *name)
      : name_(name),
        start_() {}
    ~ScopedTimer(){};
#endif
  private:
    using clock = std::chrono::steady_clock;
    const char *name_;
    clock::time_point start_;
  };

}
