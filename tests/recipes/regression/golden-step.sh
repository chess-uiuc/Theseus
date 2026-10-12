#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

RUN_OUTPUT="$THESEUS_TEST_OUTPUT/run"
REFERENCE_OUTPUT="$THESEUS_TEST_OUTPUT/reference"
RUN_OUTPUT_REL="$(recipe_output_relative_to_source "$RUN_OUTPUT")"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run the Forward-Facing Step regression case"
  (
    cd "$THESEUS_SOURCE"
    scripts/run_theseus.sh \
      -b "$THESEUS_BUILD_STANDARD" \
      -p "$THESEUS_RANKS" \
      -r "$THESEUS_DEVICE" \
      -c TestCases/Euler/2D/ForwardFacingStep/config.json \
      -m "$THESEUS_SOURCE/TestCases/Euler/2D/ForwardFacingStep/ForwardFacingStep.msh" \
      -t 0.0001 \
      -n 100 \
      -o "$RUN_OUTPUT_REL"
  )
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  echo "==> Prepare the approved reference data"
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/prepare_golden_reference.py" \
    "$THESEUS_SOURCE/TestCases/GoldenData" ForwardFacingStep "$REFERENCE_OUTPUT" \
    Cycle000100/data.pvtu
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Compare the result with the approved reference"
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/compare_viz.py" \
    "$RUN_OUTPUT/ForwardFacingStep/ParaView/Cycle000100/data.pvtu" \
    "$REFERENCE_OUTPUT/Cycle000100/data.pvtu" \
    --atol 3e-13 --rtol 1e-13
fi
