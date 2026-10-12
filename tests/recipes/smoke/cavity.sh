#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

RUN_OUTPUT="$THESEUS_TEST_OUTPUT/run"
RUN_OUTPUT_REL="$(recipe_output_relative_to_source "$RUN_OUTPUT")"

echo "==> Run the Lid-Driven Cavity smoke test"
(
  cd "$THESEUS_SOURCE"
  scripts/run_theseus.sh \
    -b "$THESEUS_BUILD_STANDARD" \
    -p "$THESEUS_RANKS" \
    -r "$THESEUS_DEVICE" \
    -c TestCases/NavierStokes/2D/LidDrivenCavity/config.json \
    -m "$THESEUS_SOURCE/TestCases/NavierStokes/2D/LidDrivenCavity/LidDrivenCavity.msh" \
    -t 0.0001 \
    -n 100 \
    -o "$RUN_OUTPUT_REL"
)
