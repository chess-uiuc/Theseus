#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"
CASES="$THESEUS_TEST_OUTPUT/cases"
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == prepare ]]; then
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/euler_orientation_study.py" prepare \
    --source "$THESEUS_SOURCE" --results "$CASES" --device "$THESEUS_DEVICE" --ranks "$THESEUS_RANKS"
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == run ]]; then
  cd "$THESEUS_SOURCE"
  # THESEUS_SHOW run
  for orientation in 0 1 2; do
    case_dir="$CASES/orientation-$orientation"
    scripts/run_theseus.sh -e "$THESEUS_BUILD_STANDARD/theseus" \
      -c "$case_dir/config.json" -o "$case_dir/launch" \
      -p "$THESEUS_RANKS" -r "$THESEUS_DEVICE" -P -k > "$case_dir/run.log" 2>&1
  done
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == compare ]]; then
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/euler_orientation_study.py" check \
    --source "$THESEUS_SOURCE" --results "$CASES" --device "$THESEUS_DEVICE"
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/euler_orientation_study.py" plot \
    --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi
