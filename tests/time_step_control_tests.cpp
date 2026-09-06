// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// SPDX-License-Identifier: BSD-3-Clause
#include "StabilityEstimate.hpp"
#include "TimeStepControl.hpp"
#include "unit_test.hpp"

using namespace Theseus;

TEST(fixed_step_is_unchanged_without_an_earlier_event)
{
  EXPECT_EQ(LimitTimeStepToEvents(0.1, 1.0, {2.0, mfem::infinity()}), 0.1);
  return 0;
}

TEST(fixed_step_lands_on_the_next_time_based_event)
{
  EXPECT_CLOSE(LimitTimeStepToEvents(0.1, 1.0, {1.04, 1.08}), 0.04, 1e-14);
  return 0;
}

TEST(an_event_at_the_current_time_does_not_create_a_zero_step)
{
  EXPECT_EQ(LimitTimeStepToEvents(0.1, 1.0, {1.0, 1.2}), 0.1);
  return 0;
}

TEST(event_detection_tolerates_roundoff_at_the_target_time)
{
  EXPECT_TRUE(TimeEventReached(0.3, 0.1 + 0.2));
  return 0;
}

TEST(stability_estimate_reports_nominal_and_shortened_step_cfl)
{
  StabilityEstimate estimate{2.0, 3.0, 2.0};
  EXPECT_EQ(estimate.CFL(0.1), 0.5);
  EXPECT_EQ(estimate.CFL(0.04), 0.2);
  return 0;
}

TEST(reference_advection_scale_uses_the_calibrated_spectral_radius)
{
  EXPECT_CLOSE(ReferenceAdvectionSpectralScale(3), 1.05 * 9.64849524786, 1e-12);
  EXPECT_CLOSE(ReferenceAdvectionSpectralScale(13), 0.65 * 14.0 * 14.0, 1e-12);
  return 0;
}

TEST(reference_br1_diffusion_scale_uses_the_calibrated_spectral_radius)
{
  EXPECT_CLOSE(ReferenceBR1DiffusionSpectralScale(3),
               1.25 * 82.9000427145, 1e-11);
  EXPECT_CLOSE(ReferenceBR1DiffusionSpectralScale(13),
               0.5 * 14.0 * 14.0 * 14.0 * 14.0, 1e-10);
  return 0;
}
