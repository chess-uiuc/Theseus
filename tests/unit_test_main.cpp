// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#include "unit_test.hpp"
#include <iostream>

static int list_tests()
{
  auto& tests = get_test_registry();
  std::cout << "Available tests:" << std::endl;
  for(const auto& tc : tests){
    std::cout << "  " << tc.name << std::endl;
  }

  return 0;
}

// Test return code conventions:
//   rc == 0 : test passed
//   rc > 0  : test failed
//   rc == -1: test not found
//   rc == -2: test threw exception
int run_test_by_name(const std::string &name)
{
  auto& tests = get_test_registry();
  for(const auto& tc : tests){
    if(name == tc.name){
      std::cout << "Running test (" << name << ")..." << std::endl
                << "[ RUN      ] " << tc.name << std::endl;
      int rc = 0;
      try {
        rc = tc.func();
      }
      catch(const std::exception& e){
        std::cerr << "  Exception in test '" << tc.name
                  << "': " << e.what() << std::endl;
        rc = -2;
      }
      catch(...){
        std::cerr << "  Unknown exception in test '" << tc.name
                  << "'" << std::endl;
        rc = -2;
      }
      if(rc == 0){
        std::cout << "[       OK ] " << tc.name << std::endl;
      } else {
        std::cout << "[  FAILED  ] " << tc.name << " (" << rc
                  << " failure" << (rc == 1 ? "" : "s")
                  << ")" << std::endl;
      }
      return rc;
    }
  }
  if(unit_test_verbosity){
    std::cerr << "Test not found: (" << name << ")" << std::endl
              << "Use '--list' to see available tests." << std::endl;
  }
  return -1;
}

// Returns 0 iff all tests found and passed
int run_tests(std::vector<std::string> &testNames)
{
    auto& tests = get_test_registry();
    int total = 0;
    int failed = 0;
    int except = 0;
    int notfound = 0;

    std::vector<std::string> passedTests;
    std::vector<std::string> failedTests;
    std::vector<std::string> missingTests;
    std::vector<std::string> exceptTests;
    if(testNames.empty()){
      for(const auto& tc : tests){
        testNames.push_back(tc.name);
      }
    }
    std::cout << "Checking " << testNames.size() << " tests...\n";
    for (const auto& name : testNames){
      int test_rc = run_test_by_name(name);
      if(test_rc == -1){
        notfound++;
        missingTests.push_back(name);
      } else if (test_rc != 0) {
        total++;
        failed++;
        failedTests.push_back(name);
        if (test_rc == -2){
          exceptTests.push_back(name);
          except++;
        }
      } else {
        passedTests.push_back(name);
        total++;
      }
    }

    std::cout << "\nSummary: " << std::endl
              << "   Total Tests: " << total << std::endl
              << "        Passed: " << total-failed << std::endl
              << "        Failed: " << failed << std::endl;
    if(except > 0){
      std::cout << "    Exceptions: " << except << std::endl;
    }
    if(notfound > 0){
      std::cout << "     Not Found: " << notfound << std::endl;
    }
    if(unit_test_verbosity){
      if(!passedTests.empty()){
        std::cout << "Passed( ";
        for(auto& name : passedTests){
          std::cout << name << " ";
        }
        std::cout << ")" << std::endl;
      }
      if(!failedTests.empty()){
        std::cout << "Failed( ";
        for(auto& name : failedTests){
          std::cout << name << " ";
        }
        std::cout << ")" << std::endl;
      }
      if(!missingTests.empty()){
        std::cout << "NotFound( ";
        for(auto& name : missingTests){
          std::cout << name << " ";
        }
        std::cout << ")" << std::endl;
      }
      if(!exceptTests.empty()){
        std::cout << "Exceptions( ";
        for(auto& name : exceptTests){
          std::cout << name << " ";
        }
        std::cout << ")" << std::endl;
      }
    }
    // Nonzero exit code if any tests failed: good for CTest/CI.
    return (failed+notfound == 0) ? 0 : 1;
}


int main(int argc, char** argv){
  int argn = 0;
  std::string programName(argv[argn++]);
  std::vector<std::string> testNames;
  while(argv[argn]){
    std::string arg(argv[argn++]);
    if(!arg.empty() && arg[0] != '-'){
      testNames.push_back(arg);
    } else {
      if(arg == "--list" || arg == "-l"){
        return list_tests();
      } else if(arg == "--help" || arg == "-h" ){
        std::cout << programName << " [--list,-l] [--verbose,-v] [testname]" << std::endl
                  << std::endl
                  << "    --help:  Print usage and quit." << std::endl
                  << "    --list:  Lists tests and quit." << std::endl
                  << " --verbose:  Advise stderr on each failure." << std::endl
                  << "  testname:  Optional; run single test by name."
                  << std::endl << std::endl;
        return 0;
      } else if(arg == "--verbose" || arg == "-v"){
        set_unit_test_verbosity(1);
      } else {
        std::cerr << programName << ": Unknown option '" << arg << "', --help for usage." << std::endl;
        return 1;
      }
    }
  }
  return run_tests(testNames);
}
