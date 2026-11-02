# Copyright (c) 2026 Board of Trustees of the University of Illinois
# SPDX-License-Identifier: BSD-3-Clause
"""Require successful valid execution or an explicit LTE rejection on CUDA."""
import os
import subprocess
import sys

executable, case = sys.argv[1:]
environment = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
result = subprocess.run([executable, case, "cuda"], env=environment,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, timeout=45)
print(result.stdout, end="")
if case == "valid":
    passed = result.returncode == 0 and "Valid lookup control PASS" in result.stdout
else:
    passed = result.returncode != 0 and "LTE runtime error:" in result.stdout
    # A raw CUDA fault without the rejection marker is never accepted.
sys.exit(0 if passed else 1)
