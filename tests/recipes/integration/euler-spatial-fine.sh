#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"
CASES="$THESEUS_TEST_OUTPUT/cases"
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == prepare ]]; then
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/euler_spatial_survey.py" prepare-fine \
    --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == run ]]; then
  cd "$THESEUS_SOURCE"
  # THESEUS_SHOW run
  for degree in 1 2 3 4 5; do
    for level in 0 1 2 3; do
      case_dir="$CASES/p${degree}-level-${level}"
      scripts/run_theseus.sh -e "$THESEUS_BUILD_STANDARD/theseus" \
        -c "$case_dir/config.json" -o "$case_dir/launch" \
        -p 1 -r "$THESEUS_DEVICE" -P -k > "$case_dir/run.log" 2>&1
    done
    case_dir="$CASES/p${degree}-half-dt"
    scripts/run_theseus.sh -e "$THESEUS_BUILD_STANDARD/theseus" \
      -c "$case_dir/config.json" -o "$case_dir/launch" \
      -p 1 -r "$THESEUS_DEVICE" -P -k > "$case_dir/run.log" 2>&1
  done
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == compare ]]; then
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/euler_spatial_survey.py" check \
    --source "$THESEUS_SOURCE" --results "$CASES" --device "$THESEUS_DEVICE"
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/euler_spatial_survey.py" plot \
    --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi
