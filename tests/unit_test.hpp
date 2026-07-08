// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus testing suites
//
// SPDX-License-Identifier: MIT
//
// This file (along with unit_test_main.cpp) is the unit testing
// infrastructure for Theseus. Tests are arranged into "suites",
// where every suite has its own executable with "main" provided
// by `unit_test_main.cpp`.
// 
// To add a new test suite executable (for example named 'mysuite'):
//   - Create mysuite_tests.cpp (include 'unit_test.hpp')
//   - Define tests using TEST(Name) { ...; return 0 (for success) }
//   - In tests/CMakeLists.txt, add an executable 'mysuite_tests'
//     with unit_test_main.cpp and mysuite_tests.cpp as sources:
//   - Register it with CTest with:
//     add_test(NAME ... COMMAND suite_exe [TestName])
//   (See tests/CMakeLists.txt for examples)
//
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

inline int unit_test_verbosity = 0;

inline void set_unit_test_verbosity(int v)
{
  unit_test_verbosity = v;
}

// A single test case: name + function returning number of failures.
struct TestCase {
    const char* name;
    int (*func)();
};

inline std::vector<TestCase>& get_test_registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

// Registers a test in the global registry at static-init time.
struct TestRegistrar {
    TestRegistrar(const char* name, int (*func)()) {
        get_test_registry().push_back(TestCase{name, func});
    }
};

// Per-test context: how many EXPECT_ macros failed.
struct TestContext {
    int failures = 0;
};

// Pointer to the context for the currently running test.
inline TestContext* current_test_context = nullptr;

// ---- Assertion-style helpers ----

// Non-fatal: records failure but doesn't abort entire binary.
#define EXPECT_TRUE(cond)                                              \
    do {                                                               \
        if (!(cond)) {                                                 \
            if (current_test_context) {                                \
                current_test_context->failures++;                      \
            }                                                          \
            if(unit_test_verbosity) {                                 \
               std::cerr << "  EXPECT_TRUE failed at "                 \
                         << __FILE__ << ":"                            \
                         << __LINE__ << ": " #cond " is false\n";      \
            }                                                          \
        }                                                              \
    } while (0)

#define EXPECT_EQ(a, b)                                                    \
    do {                                                                   \
        auto _va = (a);                                                    \
        auto _vb = (b);                                                    \
        if (!(_va == _vb)) {                                               \
            if (current_test_context) {                                    \
                current_test_context->failures++;                          \
            }                                                              \
            if(unit_test_verbosity) {                                     \
               std::cerr << "  EXPECT_EQ failed at " << __FILE__ << ":"    \
                         << __LINE__ << ": " #a " == " #b                  \
                         << " (got " << _va << ", expected "               \
                         << _vb << ")\n";                                  \
            }                                                              \
        }                                                                  \
    } while (0)

#define EXPECT_CLOSE(a, b, tol)                                            \
    do {                                                                   \
        auto _va = (a);                                                    \
        auto _vb = (b);                                                    \
        auto _vtol = (tol);                                                \
        using std::abs;                                                    \
        auto _diff = abs(_va - _vb);                                       \
        if (!(_diff <= _vtol)) {                                           \
            if (current_test_context) {                                    \
                current_test_context->failures++;                          \
            }                                                              \
            if(unit_test_verbosity) {                                     \
               std::cerr << "  EXPECT_CLOSE failed at " << __FILE__ << ":" \
                         << __LINE__ << ": |" #a " - " #b "| <= " #tol     \
                         << " (|diff| = " << _diff                         \
                         << ", tol = " << _vtol << ")\n";                  \
            }                                                              \
        }                                                                  \
    } while (0)

#define EXPECT_SMALL(x, tol)                                               \
    do {                                                                   \
        auto _vx = (x);                                                    \
        auto _vtol = (tol);                                                \
        using std::abs;                                                    \
        auto _mag = abs(_vx);                                              \
        if (!(_mag <= _vtol)) {                                            \
            if (current_test_context) {                                    \
                current_test_context->failures++;                          \
            }                                                              \
            if(unit_test_verbosity) {                                     \
               std::cerr << "  EXPECT_SMALL failed at " << __FILE__ << ":" \
                         << __LINE__ << ": |" #x "| <= " #tol              \
                         << " (|x| = " << _mag                             \
                         << ", tol = " << _vtol << ")\n";                  \
            }                                                              \
        }                                                                  \
    } while (0)

// RAII helper that observes and suppresses EXPECT_* failures.
//
// Semantics:
//  - On construction, it records the current failure count in the active TestContext.
//  - While it is alive, EXPECT_* macros bump ctx->failures as usual.
//  - count() reports how many failures have occurred since the last clear()/construction.
//  - clear() returns that count and resets the baseline for future counting.
//  - On destruction, it restores ctx->failures to the value it had at construction,
//    so none of the failures observed while it was alive affect the outer test result.
class SuppressFailures {
public:
    SuppressFailures()
        : ctx_(current_test_context),
          start_failures_(ctx_ ? ctx_->failures : 0),
          mark_failures_(start_failures_)
    {}

    ~SuppressFailures()
    {
        if (ctx_) {
            // Discard all failures accumulated while we were active
            ctx_->failures = start_failures_;
        }
    }

    // Number of failures since construction or last clear().
    int count() const
    {
        if (!ctx_) {
            return 0;
        }
        int now = ctx_->failures;
        return now - mark_failures_;
    }

    // Return the current count and reset the "baseline" so that
    // further EXPECT_* calls are counted separately.
    int clear()
    {
        int c = count();
        if (ctx_) {
            mark_failures_ = ctx_->failures;
        }
        return c;
    }

private:
    TestContext* ctx_;
    int start_failures_;  // failures at construction
    int mark_failures_;   // baseline for count()/clear()
};

// ---- TEST(name) macro ----
//
// Usage:
//
//   TEST(MyCoolTest) {
//       EXPECT_TRUE(...);
//       EXPECT_EQ(...);
//       return 0; // or > 0 for failure
//   }
//
// This defines a function MyCoolTest_impl(), wraps it in a runner
// that manages TestContext, and registers that runner in the global
// registry so test_main can find and execute it.
#define TEST(Name)                                                 \
    int Name##_impl();                                             \
    int Name() {                                                   \
        TestContext ctx;                                           \
        current_test_context = &ctx;                               \
        int user_rc = Name##_impl();                               \
        current_test_context = nullptr;                            \
        int test_rc = ctx.failures + (user_rc != 0 ? 1 : 0);       \
        return test_rc;                                            \
    }                                                              \
    static TestRegistrar registrar_##Name(#Name, &Name);           \
    int Name##_impl()
