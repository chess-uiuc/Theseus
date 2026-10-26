#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

RUN_OUTPUT="$THESEUS_TEST_OUTPUT/run"
RUN_OUTPUT_REL="$(recipe_output_relative_to_source "$RUN_OUTPUT")"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run the Isentropic Vortex cyclic regression case"
  (
    cd "$THESEUS_SOURCE"
    # THESEUS_SHOW run
    scripts/run_theseus.sh \
      -b "$THESEUS_BUILD_STANDARD" \
      -p "$THESEUS_RANKS" \
      -r "$THESEUS_DEVICE" \
      -c TestCases/Euler/2D/IsentropicVortex/config.json \
      -m "$THESEUS_SOURCE/TestCases/Euler/2D/IsentropicVortex/IsentropicVortex.mesh" \
      -o "$RUN_OUTPUT_REL"
    # THESEUS_SHOW_END
  )
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Compare the first and last density fields"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/scripts/compare_viz.py" \
    "$RUN_OUTPUT/IsentropicVortex/ParaView/ParaView.pvd" \
    --fields Density \
    --atol 5e-5 \
    --rtol 2e-6
  # THESEUS_SHOW_END
fi
