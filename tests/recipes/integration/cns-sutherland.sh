#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"
CASES="$THESEUS_TEST_OUTPUT/cases"
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == prepare ]]; then
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/cns_sutherland_study.py" prepare \
    --source "$THESEUS_SOURCE" --results "$CASES" --ranks "$THESEUS_RANKS"
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == run ]]; then
  cd "$THESEUS_SOURCE"
  # THESEUS_SHOW run
  while IFS=$'\t' read -r name ranks; do
    case_dir="$CASES/$name"
    scripts/run_theseus.sh -e "$THESEUS_BUILD_STANDARD/theseus" \
      -c "$case_dir/config.json" -o "$case_dir/launch" \
      -p "$ranks" -r "$THESEUS_DEVICE" -P -k < /dev/null > "$case_dir/run.log" 2>&1
  done < "$CASES/cases.tsv"
  # THESEUS_SHOW_END
fi
if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == compare ]]; then
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/cns_sutherland_study.py" check \
    --source "$THESEUS_SOURCE" --results "$CASES" --device "$THESEUS_DEVICE"
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/mms/cns_sutherland_study.py" plot \
    --source "$THESEUS_SOURCE" --results "$CASES"
  # THESEUS_SHOW_END
fi
