#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

RUN_OUTPUT="$THESEUS_TEST_OUTPUT/run"
REFERENCE_OUTPUT="$THESEUS_TEST_OUTPUT/reference"
RUN_OUTPUT_REL="$(recipe_output_relative_to_source "$RUN_OUTPUT")"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run the Lid-Driven Cavity regression case"
  (
    cd "$THESEUS_SOURCE"
    # THESEUS_SHOW run
    scripts/run_theseus.sh \
      -b "$THESEUS_BUILD_STANDARD" \
      -p "$THESEUS_RANKS" \
      -r "$THESEUS_DEVICE" \
      -c TestCases/NavierStokes/2D/LidDrivenCavity/config.json \
      -m "$THESEUS_SOURCE/TestCases/NavierStokes/2D/LidDrivenCavity/LidDrivenCavity.msh" \
      -t 0.0001 \
      -n 4000 \
      -o "$RUN_OUTPUT_REL"
    # THESEUS_SHOW_END
  )
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  echo "==> Prepare the approved reference data"
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/prepare_golden_reference.py" \
    "$THESEUS_SOURCE/TestCases/GoldenData" LidDrivenCavity "$REFERENCE_OUTPUT" \
    Cycle004000/data.pvtu
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Compare the result with the approved reference"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/compare_viz.py" \
    "$RUN_OUTPUT/LidDrivenCavity/ParaView/Cycle004000/data.pvtu" \
    "$REFERENCE_OUTPUT/Cycle004000/data.pvtu" \
    --atol 1e-12 --rtol 1e-12
  # THESEUS_SHOW_END
fi
