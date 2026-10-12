#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

RUN_OUTPUT="$THESEUS_TEST_OUTPUT/run"
RUN_OUTPUT_REL="$(recipe_output_relative_to_source "$RUN_OUTPUT")"

echo "==> Run the Isentropic Vortex smoke test"
(
  cd "$THESEUS_SOURCE"
  scripts/run_theseus.sh \
    -b "$THESEUS_BUILD_STANDARD" \
    -p "$THESEUS_RANKS" \
    -r "$THESEUS_DEVICE" \
    -c TestCases/Euler/2D/IsentropicVortex/config.json \
    -m "$THESEUS_SOURCE/TestCases/Euler/2D/IsentropicVortex/IsentropicVortex.mesh" \
    -t 0.002 \
    -n 100 \
    -o "$RUN_OUTPUT_REL"
)
