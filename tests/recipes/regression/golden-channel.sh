#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"
CASES="$THESEUS_TEST_OUTPUT/case"

if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == prepare ]]; then
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/channel_flow_regression.py" prepare \
    --source "$THESEUS_SOURCE" --build "$THESEUS_BUILD_STANDARD" \
    --database "$THESEUS_PREFIX/database" --results "$CASES"
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == run ]]; then
  cd "$THESEUS_SOURCE"
  echo "Running 50 fixed channel steps; log: $CASES/run.log"
  # THESEUS_SHOW run
  # Bound the solver/harness process group to nine minutes, leaving analysis margin.
  "$THESEUS_PYTHON" - "$CASES/wall-seconds.txt" \
    scripts/run_theseus.sh -e "$THESEUS_BUILD_STANDARD/theseus" \
    -c "$CASES/config.json" -o "$CASES/launch" -p "$THESEUS_RANKS" \
    -r "$THESEUS_DEVICE" -P -R -k > "$CASES/run.log" 2>&1 <<'PY'
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
start = time.monotonic()
process = subprocess.Popen(['bash', *sys.argv[2:]], start_new_session=True)
try:
    code = process.wait(timeout=540)
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    code = 124
    print('FAIL: channel run exceeded the nine-minute solver limit', flush=True)
Path(sys.argv[1]).write_text(str(time.monotonic()-start)+'\n')
raise SystemExit(code)
PY
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == all || "$THESEUS_RECIPE_STAGE" == compare ]]; then
  echo "Comparing channel fields and integral histories with the selected CPU golden"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/channel_flow_regression.py" check \
    --source "$THESEUS_SOURCE" --build "$THESEUS_BUILD_STANDARD" \
    --results "$CASES" --device "$THESEUS_DEVICE"
  # THESEUS_SHOW_END
fi
