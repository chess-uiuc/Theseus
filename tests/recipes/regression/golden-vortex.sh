#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

RUN_OUTPUT="$THESEUS_TEST_OUTPUT/run"
REFERENCE_OUTPUT="$THESEUS_TEST_OUTPUT/reference"
RUN_OUTPUT_REL="$(recipe_output_relative_to_source "$RUN_OUTPUT")"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run the Isentropic Vortex regression case"
  (
    cd "$THESEUS_SOURCE"
    # THESEUS_SHOW run
    scripts/run_theseus.sh \
      -b "$THESEUS_BUILD_STANDARD" \
      -p "$THESEUS_RANKS" \
      -r "$THESEUS_DEVICE" \
      -c TestCases/Euler/2D/IsentropicVortex/config.json \
      -m "$THESEUS_SOURCE/TestCases/Euler/2D/IsentropicVortex/IsentropicVortex.mesh" \
      -t 0.001 \
      -n 500 \
      -o "$RUN_OUTPUT_REL"
    # THESEUS_SHOW_END
  )
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  echo "==> Prepare the approved reference data"
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/prepare_golden_reference.py" \
    "$THESEUS_SOURCE/TestCases/GoldenData" \
    IsentropicVortex \
    "$REFERENCE_OUTPUT" \
    Cycle000500/data.pvtu
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Compare the result with the approved reference"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/compare_viz.py" \
    "$RUN_OUTPUT/IsentropicVortex/ParaView/Cycle000500/data.pvtu" \
    "$REFERENCE_OUTPUT/Cycle000500/data.pvtu" \
    --atol 1e-13 \
    --rtol 1e-13
  # THESEUS_SHOW_END
fi
