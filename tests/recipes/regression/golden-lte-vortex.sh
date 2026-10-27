#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

RUN_OUTPUT="$THESEUS_TEST_OUTPUT/run"
REFERENCE_OUTPUT="$THESEUS_TEST_OUTPUT/reference"
RUN_OUTPUT_REL="$(recipe_output_relative_to_source "$RUN_OUTPUT")"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run the LTE Vortex regression case"
  (
    cd "$THESEUS_SOURCE"
    # THESEUS_SHOW run
    scripts/run_theseus.sh \
      -b "$THESEUS_BUILD_STANDARD" \
      -p "$THESEUS_RANKS" \
      -r "$THESEUS_DEVICE" \
      -c TestCases/LTE/Euler/LTEVortex/config.json \
      -m "$THESEUS_SOURCE/TestCases/LTE/Euler/LTEVortex/LTEVortex.mesh" \
      -t 1e-5 \
      -n 200 \
      -o "$RUN_OUTPUT_REL"
    # THESEUS_SHOW_END
  )
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  echo "==> Prepare the approved reference data"
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/prepare_golden_reference.py" \
    "$THESEUS_SOURCE/TestCases/GoldenData" LTEVortex "$REFERENCE_OUTPUT" \
    Cycle000200/data.pvtu
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Compare the result with the approved reference"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/compare_viz.py" \
    "$RUN_OUTPUT/LTEVortex/ParaView/Cycle000200/data.pvtu" \
    "$REFERENCE_OUTPUT/Cycle000200/data.pvtu" \
    --atol 1e-10 --rtol 1e-10
  # THESEUS_SHOW_END
fi
