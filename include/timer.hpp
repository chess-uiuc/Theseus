// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <chrono>
#include <cstdio>
#include <mpi.h>
#ifdef TIMER_SYNC_DEVICE
#include "mfem.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace Theseus
{
  class ScopedTimer
  {
  public:
#ifdef ENABLE_TIMERS
#ifndef TIMER_SYNC_DEVICE
    explicit ScopedTimer(const char *name)
      : name_(name),
	start_(clock::now())
    {}
#else
    explicit ScopedTimer(const char *name)
      : name_(name)
    {
      MFEM_DEVICE_SYNC;
      //      mfem::Device::Sync();
      start_ = clock::now();
    }
#endif

    ~ScopedTimer()
    {
#ifdef TIMER_SYNC_DEVICE
      MFEM_DEVICE_SYNC;
      // mfem::Device::Sync();
#endif
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
      if(rank == 0)
	MPI_Reduce(MPI_IN_PLACE,&global_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      else
	MPI_Reduce(&global_ms, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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


  class TimestepTimer
  {
  public:
    using clock = std::chrono::steady_clock;

    explicit TimestepTimer(MPI_Comm comm = MPI_COMM_WORLD)
      : comm_(comm)
    {
      MPI_Comm_rank(comm_, &rank_);
      MPI_Comm_size(comm_, &nranks_);
    }

    void Start()
    {
      Sync();
      start_ = clock::now();
      running_ = true;
    }

    void Stop()
    {
      if (!running_)
	{
	  return;
	}

      Sync();
      const auto stop = clock::now();

      const double elapsed_ms =
	std::chrono::duration<double, std::milli>(stop - start_).count();

      total_ms_ += elapsed_ms;
      min_ms_ = std::min(min_ms_, elapsed_ms);
      max_ms_ = std::max(max_ms_, elapsed_ms);
      ++count_;

      running_ = false;
    }

    void Reset()
    {
      count_ = 0;
      total_ms_ = 0.0;
      min_ms_ = std::numeric_limits<double>::max();
      max_ms_ = 0.0;
      running_ = false;
    }

    std::uint64_t Count() const
    {
      return count_;
    }

    double Total() const
    {
      return total_ms_;
    }

    double Min() const
    {
      return count_ > 0 ? min_ms_ : 0.0;
    }

    double Max() const
    {
      return count_ > 0 ? max_ms_ : 0.0;
    }

    double Mean() const
    {
      return count_ > 0 ? total_ms_ / static_cast<double>(count_) : 0.0;
    }

    void Finalize(std::ostream &os = std::cout) const
    {
      //
      // Local statistics.
      //
      const double local_total = Total();
      const double local_min   = Min();
      const double local_max   = Max();
      const double local_mean  = Mean();

      const unsigned long long local_count =
	static_cast<unsigned long long>(count_);

      //
      // Count statistics across ranks.
      //
      unsigned long long count_min = 0;
      unsigned long long count_max = 0;
      unsigned long long count_sum = 0;

      MPI_Reduce(&local_count, &count_min, 1,
                 MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, comm_);

      MPI_Reduce(&local_count, &count_max, 1,
                 MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm_);

      MPI_Reduce(&local_count, &count_sum, 1,
                 MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm_);

      //
      // For each local statistic, collect min/max/sum across ranks.
      //
      double total_min = 0.0;
      double total_max = 0.0;
      double total_sum = 0.0;

      double step_min_min = 0.0;
      double step_min_max = 0.0;
      double step_min_sum = 0.0;

      double step_max_min = 0.0;
      double step_max_max = 0.0;
      double step_max_sum = 0.0;

      double mean_min = 0.0;
      double mean_max = 0.0;
      double mean_sum = 0.0;

      ReduceMinMaxSum(local_total,
                      total_min, total_max, total_sum);

      ReduceMinMaxSum(local_min,
                      step_min_min, step_min_max, step_min_sum);

      ReduceMinMaxSum(local_max,
                      step_max_min, step_max_max, step_max_sum);

      ReduceMinMaxSum(local_mean,
                      mean_min, mean_max, mean_sum);

      if (rank_ != 0)
	{
	  return;
	}

      const double inv_nranks = 1.0 / static_cast<double>(nranks_);

      const double count_mean =
	static_cast<double>(count_sum) * inv_nranks;

      const double total_mean =
	total_sum * inv_nranks;

      const double step_min_mean =
	step_min_sum * inv_nranks;

      const double step_max_mean =
	step_max_sum * inv_nranks;

      const double mean_mean =
	mean_sum * inv_nranks;

      auto old_flags = os.flags();
      auto old_precision = os.precision();
      os << std::fixed << std::setprecision(3);
      os << "\n"
         << "========================================================================\n"
         << "Theseus timestep performance\n"
         << "========================================================================\n"
         << "MPI ranks         : " << nranks_ << "\n"
         << "\n"
         << "Per-rank timestep statistics\n"
         << "                                      min"
         << "          mean"
         << "           max\n"
         << "  Timed steps       : "
         << std::setw(12) << count_min
         << std::setw(14) << count_mean
         << std::setw(14) << count_max << "\n"
         << "  Total time (ms)   : "
         << std::setw(12) << total_min
         << std::setw(14) << total_mean
         << std::setw(14) << total_max << "\n"
         << "  Min step (ms)     : "
         << std::setw(12) << step_min_min
         << std::setw(14) << step_min_mean
         << std::setw(14) << step_min_max << "\n"
         << "  Mean step (ms)    : "
         << std::setw(12) << mean_min
         << std::setw(14) << mean_mean
         << std::setw(14) << mean_max << "\n"
         << "  Max step (ms)     : "
         << std::setw(12) << step_max_min
         << std::setw(14) << step_max_mean
         << std::setw(14) << step_max_max << "\n"
         << "\n"
         << "Critical-rank performance\n"
         << "  Mean timestep    : " << mean_max << " ms\n";

      if (mean_max > 0.0)
	{
	  os << "  Timesteps/sec    : "
	     << 1000.0 / mean_max << "\n";
	}

#ifdef TIMER_SYNC_DEVICE
      os << "  Device sync      : enabled\n";
#else
      os << "  Device sync      : disabled\n";
#endif

#ifdef TIMER_BARRIER
      os << "  MPI barrier      : enabled\n";
#else
      os << "  MPI barrier      : disabled\n";
#endif

      os << "========================================================================\n";
      os.flags(old_flags);
      os.precision(old_precision);
    }

  private:
    void Sync() const
    {
#ifdef TIMER_BARRIER
      MPI_Barrier(comm_);
#endif

#ifdef TIMER_SYNC_DEVICE
      MFEM_DEVICE_SYNC;
#endif
    }

    void ReduceMinMaxSum(double local,
			 double &minimum,
			 double &maximum,
			 double &sum) const
    {
      MPI_Reduce(&local, &minimum, 1,
		 MPI_DOUBLE, MPI_MIN, 0, comm_);

      MPI_Reduce(&local, &maximum, 1,
		 MPI_DOUBLE, MPI_MAX, 0, comm_);

      MPI_Reduce(&local, &sum, 1,
		 MPI_DOUBLE, MPI_SUM, 0, comm_);
    }

    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int nranks_ = 1;

    clock::time_point start_;

    std::uint64_t count_ = 0;
    double total_ms_ = 0.0;
    double min_ms_ = std::numeric_limits<double>::max();
    double max_ms_ = 0.0;

    bool running_ = false;
  };
}
