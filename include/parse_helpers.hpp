// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cctype>
#include <string>

auto to_lower = [](std::string s)
 {
   std::transform(s.begin(), s.end(), s.begin(),
                  [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                  });
   return s;
 };

auto starts_with = [](const std::string &s, const std::string &prefix)
 {
   return s.rfind(prefix, 0) == 0;
 };
