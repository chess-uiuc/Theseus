#!/usr/bin/env bash

recipe_usage() {
  echo "Usage: $1 --source DIR --build-standard DIR --prefix DIR --output DIR" >&2
  echo "          [--build-axisymmetric DIR]" >&2
  echo "          [--device cpu|cuda] [--python PYTHON] [--ranks N]" >&2
  echo "          [--stage all|run|prepare|compare]" >&2
}

recipe_parse_arguments() {
  THESEUS_SOURCE=""
  THESEUS_BUILD_STANDARD=""
  THESEUS_BUILD_AXISYMMETRIC=""
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
      --build-axisymmetric) THESEUS_BUILD_AXISYMMETRIC="$2"; shift 2 ;;
      --prefix) THESEUS_PREFIX="$2"; shift 2 ;;
      --output) THESEUS_TEST_OUTPUT="$2"; shift 2 ;;
      --device) THESEUS_DEVICE="$2"; shift 2 ;;
      --python) THESEUS_PYTHON="$2"; shift 2 ;;
      --ranks) THESEUS_RANKS="$2"; shift 2 ;;
      --stage) THESEUS_RECIPE_STAGE="$2"; shift 2 ;;
      *) recipe_usage "$0"; echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "$THESEUS_SOURCE" ||
        ( -z "$THESEUS_BUILD_STANDARD" && -z "$THESEUS_BUILD_AXISYMMETRIC" ) ||
        -z "$THESEUS_PREFIX" || -z "$THESEUS_TEST_OUTPUT" ]]; then
    recipe_usage "$0"
    exit 2
  fi
  THESEUS_SOURCE="$(cd "$THESEUS_SOURCE" && pwd)"
  if [[ -n "$THESEUS_BUILD_STANDARD" ]]; then
    THESEUS_BUILD_STANDARD="$(cd "$THESEUS_BUILD_STANDARD" && pwd)"
  fi
  if [[ -n "$THESEUS_BUILD_AXISYMMETRIC" ]]; then
    THESEUS_BUILD_AXISYMMETRIC="$(cd "$THESEUS_BUILD_AXISYMMETRIC" && pwd)"
  fi
  THESEUS_PREFIX="$(cd "$THESEUS_PREFIX" && pwd)"
  for build in "$THESEUS_BUILD_STANDARD" "$THESEUS_BUILD_AXISYMMETRIC"; do
    if [[ -n "$build" && ! -x "$build/theseus" ]]; then
      echo "Missing executable: $build/theseus" >&2
      exit 2
    fi
  done
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
  recipe_create_output() {
    mkdir -p "$THESEUS_TEST_OUTPUT"
    mkdir -p "$THESEUS_TEST_OUTPUT/tpl"
    ln -s "$THESEUS_PREFIX" "$THESEUS_TEST_OUTPUT/tpl/install"
  }
  case "$THESEUS_RECIPE_STAGE" in
    all)
      if [[ -e "$THESEUS_TEST_OUTPUT" ]]; then
        echo "Output path already exists: $THESEUS_TEST_OUTPUT" >&2
        exit 2
      fi
      recipe_create_output
      ;;
    run)
      if [[ ! -e "$THESEUS_TEST_OUTPUT" ]]; then
        recipe_create_output
      elif [[ -f "$THESEUS_TEST_OUTPUT/.theseus-recipe-prepared" ]]; then
        rm "$THESEUS_TEST_OUTPUT/.theseus-recipe-prepared"
      else
        echo "Output path already exists: $THESEUS_TEST_OUTPUT" >&2
        exit 2
      fi
      ;;
    prepare)
      if [[ ! -e "$THESEUS_TEST_OUTPUT" ]]; then
        recipe_create_output
        touch "$THESEUS_TEST_OUTPUT/.theseus-recipe-prepared"
      fi
      ;;
    compare)
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
