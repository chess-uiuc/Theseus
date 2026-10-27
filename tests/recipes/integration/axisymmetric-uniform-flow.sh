#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

CASES="$THESEUS_TEST_OUTPUT/cases"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  echo "==> Prepare the Axisymmetric Uniform Flow inputs"
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/axisymmetric_uniform_flow_integration_test.py" \
    prepare --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run the Axisymmetric Uniform Flow cases"
  cd "$THESEUS_SOURCE"
  # THESEUS_SHOW run
  scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
    -c "$CASES/euler-1/config.json" -o "$CASES/euler-1/launch" \
    -p 1 -r "$THESEUS_DEVICE" -P -k > "$CASES/euler-1/run.log" 2>&1
  scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
    -c "$CASES/euler-2/config.json" -o "$CASES/euler-2/launch" \
    -p 2 -r "$THESEUS_DEVICE" -P -k > "$CASES/euler-2/run.log" 2>&1
  scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
    -c "$CASES/navier-stokes-1/config.json" -o "$CASES/navier-stokes-1/launch" \
    -p 1 -r "$THESEUS_DEVICE" -P -k > "$CASES/navier-stokes-1/run.log" 2>&1
  scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
    -c "$CASES/navier-stokes-2/config.json" -o "$CASES/navier-stokes-2/launch" \
    -p 2 -r "$THESEUS_DEVICE" -P -k > "$CASES/navier-stokes-2/run.log" 2>&1
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Check the Axisymmetric Uniform Flow results"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/axisymmetric_uniform_flow_integration_test.py" \
    check --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi
