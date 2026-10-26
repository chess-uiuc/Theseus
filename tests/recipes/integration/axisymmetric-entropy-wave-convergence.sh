#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"

CASES="$THESEUS_TEST_OUTPUT/cases"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  echo "==> Prepare the Axisymmetric Entropy-Wave Convergence inputs"
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/axisymmetric_entropy_wave_convergence_test.py" \
    prepare --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run the Axisymmetric Entropy-Wave Convergence cases"
  cd "$THESEUS_SOURCE"
  # THESEUS_SHOW run
  scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
    -c "$CASES/level-0/config.json" -o "$CASES/level-0/launch" \
    -p 1 -r "$THESEUS_DEVICE" -P -k > "$CASES/level-0/run.log" 2>&1
  scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
    -c "$CASES/level-1/config.json" -o "$CASES/level-1/launch" \
    -p 1 -r "$THESEUS_DEVICE" -P -k > "$CASES/level-1/run.log" 2>&1
  scripts/run_theseus.sh -e "$THESEUS_BUILD_AXISYMMETRIC/theseus" \
    -c "$CASES/level-2/config.json" -o "$CASES/level-2/launch" \
    -p 1 -r "$THESEUS_DEVICE" -P -k > "$CASES/level-2/run.log" 2>&1
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Check the Axisymmetric Entropy-Wave Convergence results"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/axisymmetric_entropy_wave_convergence_test.py" \
    check --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi
