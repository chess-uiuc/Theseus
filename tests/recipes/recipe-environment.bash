#!/usr/bin/env bash

recipe_usage() {
  echo "Usage: $1 --source DIR --build-standard DIR --prefix DIR --output DIR" >&2
  echo "          [--device cpu|cuda] [--python PYTHON] [--ranks N]" >&2
  echo "          [--stage all|run|prepare|compare]" >&2
}

recipe_parse_arguments() {
  THESEUS_SOURCE=""
  THESEUS_BUILD_STANDARD=""
  THESEUS_PREFIX=""
  THESEUS_TEST_OUTPUT=""
  THESEUS_DEVICE="cpu"
  THESEUS_PYTHON="python3"
  THESEUS_RANKS="2"
  THESEUS_RECIPE_STAGE="all"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --source) THESEUS_SOURCE="$2"; shift 2 ;;
      --build-standard) THESEUS_BUILD_STANDARD="$2"; shift 2 ;;
      --prefix) THESEUS_PREFIX="$2"; shift 2 ;;
      --output) THESEUS_TEST_OUTPUT="$2"; shift 2 ;;
      --device) THESEUS_DEVICE="$2"; shift 2 ;;
      --python) THESEUS_PYTHON="$2"; shift 2 ;;
      --ranks) THESEUS_RANKS="$2"; shift 2 ;;
      --stage) THESEUS_RECIPE_STAGE="$2"; shift 2 ;;
      *) recipe_usage "$0"; echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "$THESEUS_SOURCE" || -z "$THESEUS_BUILD_STANDARD" ||
        -z "$THESEUS_PREFIX" || -z "$THESEUS_TEST_OUTPUT" ]]; then
    recipe_usage "$0"
    exit 2
  fi
  THESEUS_SOURCE="$(cd "$THESEUS_SOURCE" && pwd)"
  THESEUS_BUILD_STANDARD="$(cd "$THESEUS_BUILD_STANDARD" && pwd)"
  THESEUS_PREFIX="$(cd "$THESEUS_PREFIX" && pwd)"
  if [[ ! -x "$THESEUS_BUILD_STANDARD/theseus" ]]; then
    echo "Missing executable: $THESEUS_BUILD_STANDARD/theseus" >&2
    exit 2
  fi
  if [[ ! "$THESEUS_RANKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Ranks must be a positive integer: $THESEUS_RANKS" >&2
    exit 2
  fi
  case "$THESEUS_RECIPE_STAGE" in
    all|run|prepare|compare) ;;
    *) echo "Unknown recipe stage: $THESEUS_RECIPE_STAGE" >&2; exit 2 ;;
  esac
  THESEUS_TEST_OUTPUT="$("$THESEUS_PYTHON" - "$THESEUS_TEST_OUTPUT" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).resolve())
PY
)"
  case "$THESEUS_RECIPE_STAGE" in
    all|run)
      if [[ -e "$THESEUS_TEST_OUTPUT" ]]; then
        echo "Output path already exists: $THESEUS_TEST_OUTPUT" >&2
        exit 2
      fi
      mkdir -p "$THESEUS_TEST_OUTPUT"
      mkdir -p "$THESEUS_TEST_OUTPUT/tpl"
      ln -s "$THESEUS_PREFIX" "$THESEUS_TEST_OUTPUT/tpl/install"
      ;;
    prepare|compare)
      if [[ ! -d "$THESEUS_TEST_OUTPUT" ]]; then
        echo "Output path does not exist: $THESEUS_TEST_OUTPUT" >&2
        exit 2
      fi
      ;;
  esac
}

recipe_output_relative_to_source() {
  "$THESEUS_PYTHON" - "$THESEUS_SOURCE" "$1" <<'PY'
import os
import sys
print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
}
