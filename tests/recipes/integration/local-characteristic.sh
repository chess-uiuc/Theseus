#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"
CASES="$THESEUS_TEST_OUTPUT/cases"
if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/local_characteristic_integration_test.py" prepare \
    --source "$THESEUS_SOURCE" --results "$CASES" --database "$THESEUS_PREFIX/database"
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  cd "$THESEUS_SOURCE"
  # THESEUS_SHOW run
  for provider in reference radial total reverse; do
    for model in euler cns; do
      scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
        -c "$CASES/$provider/$model-1/config.json" -o "$CASES/$provider/$model-1/launch" \
        -p 1 -r "$THESEUS_DEVICE" -P -k > "$CASES/$provider/$model-1/run.log" 2>&1
      scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
        -c "$CASES/$provider/$model-2/config.json" -o "$CASES/$provider/$model-2/launch" \
        -p 2 -r "$THESEUS_DEVICE" -P -k > "$CASES/$provider/$model-2/run.log" 2>&1
    done
  done
  for scenario in heated reverse; do
    scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
      -c "$CASES/chamber-$scenario-1/config.json" -o "$CASES/chamber-$scenario-1/launch" \
      -p 1 -r "$THESEUS_DEVICE" -P -k > "$CASES/chamber-$scenario-1/run.log" 2>&1
    scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
      -c "$CASES/chamber-$scenario-2/config.json" -o "$CASES/chamber-$scenario-2/launch" \
      -p 2 -r "$THESEUS_DEVICE" -P -k > "$CASES/chamber-$scenario-2/run.log" 2>&1
  done
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/local_characteristic_integration_test.py" check \
    --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi
