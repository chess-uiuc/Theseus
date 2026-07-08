// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "unit_test.hpp"

// Minimal test stub to verify that the unit test infrastructure works.
// This should always pass if the test harness is wired correctly.
TEST(unit_testing_funcs)
{
  double tol = 1e-16;
  double eps = 1e-18;
  EXPECT_TRUE(true);
  EXPECT_EQ(1+1, 2);
  EXPECT_CLOSE(1+1, 2-eps, tol);
  EXPECT_SMALL(eps, tol);
  return 0;
}
TEST(expect_true)
{
  int test_result = 0;
  EXPECT_TRUE(true);
  
  {
    SuppressFailures sf;
    EXPECT_TRUE(false);
    if(sf.count() != 1){
      test_result++;
    }
  }

  return test_result;
}

TEST(expect_eq)
{
  int test_result = 0;
  EXPECT_EQ(1 + 1, 2);
  {
    SuppressFailures sf;
    EXPECT_EQ(1+1, 3);
    if (sf.count() != 1) {
      test_result++;
    }
  }
  return test_result;
}

TEST(expect_close)
{
  int test_result = 0;
  double tol = 1.2e-16;
  double eps = 1e-16;
  double val1 = 1.0;
  double val2 = 2.0;
  double val3 = val1 + eps;
  double val4 = val1 - eps;
  double val5 = val1 - 2*eps;
  EXPECT_CLOSE(val1, val3, tol);
  EXPECT_CLOSE(val1, val4, tol);
  {
    SuppressFailures sf;
    EXPECT_CLOSE(val1, val2, tol);
    if (sf.count() != 1) {
      test_result++;
    }
    sf.clear();
    EXPECT_CLOSE(val1, val5, tol);
    if (sf.count() != 1) {
      test_result++;
    }
  }
  return test_result;
}

TEST(expect_small)
{
  int test_result = 0;
  double tol = 1e-16;
  double eps = 1e-18;
  double val1 = tol-eps;
  double val2 = tol+eps;

  EXPECT_SMALL(eps,  tol);
  EXPECT_SMALL(-eps, tol);
  EXPECT_SMALL(val1, tol);
  EXPECT_SMALL(-val1, tol);
  {
    SuppressFailures sf;
    EXPECT_SMALL(val2, tol);
    if (sf.count() != 1) {
      test_result++;
    }
    sf.clear();
    EXPECT_SMALL(-val2, tol);
    if (sf.count() != 1) {
      test_result++;
    }
  }
  return test_result;
}
