#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
TPL_DIR="${TPL_DIR:-$ROOT_DIR/tpl}"
SRC_DIR="${SRC_DIR:-$TPL_DIR/src}"
BUILD_DIR="${BUILD_DIR:-$TPL_DIR/build}"
PREFIX="${PREFIX:-$TPL_DIR/install}"
LOG_DIR="${LOG_DIR:-$TPL_DIR/logs}"

JOBS="${JOBS:-4}"
DEVICE="${DEVICE:-cpu}"       # cpu, cuda, hip
CUDA_ARCH="${CUDA_ARCH:-75}"
HIP_ARCH="${HIP_ARCH:-}"

export CC="${CC:-mpicc}"
export CXX="${CXX:-mpicxx}"
export FC="${FC:-mpif90}"

GKLIB_REPO="${GKLIB_REPO:-https://github.com/KarypisLab/GKlib.git}"
GKLIB_REF="${GKLIB_REF:-master}"

METIS_REPO="${METIS_REPO:-https://github.com/KarypisLab/METIS.git}"
METIS_REF="${METIS_REF:-master}"

HYPRE_REPO="${HYPRE_REPO:-https://github.com/hypre-space/hypre.git}"
HYPRE_REF="${HYPRE_REF:-master}"

MFEM_REPO="${MFEM_REPO:-https://github.com/mfem/mfem.git}"
MFEM_REF="${MFEM_REF:-master}"

PLATO_REPO="${PLATO_REPO:-https://github.com/chess-uiuc/plato.git}"
PLATO_REF="${PLATO_REF:-main}"

THERMO_DATABASE_REPO="${THERMO_DATABASE_REPO:-https://github.com/chess-uiuc/database.git}"
THERMO_DATABASE_REF="${THERMO_DATABASE_REF:-main}"

mkdir -p "$SRC_DIR" "$BUILD_DIR" "$PREFIX" "$LOG_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
MANIFEST="$LOG_DIR/deps-build-$STAMP.txt"

log() {
  echo "$*" | tee -a "$MANIFEST"
}

run_logged() {
  local name="$1"
  shift
  log ""
  log "### $name"
  log "PWD: $(pwd)"
  log "COMMAND: $*"
  "$@" 2>&1 | tee "$LOG_DIR/${STAMP}-${name}.log"
}

record_system_info() {
  {
    echo "Dependency build manifest"
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "ROOT_DIR: $ROOT_DIR"
    echo "TPL_DIR: $TPL_DIR"
    echo "SRC_DIR: $SRC_DIR"
    echo "BUILD_DIR: $BUILD_DIR"
    echo "PREFIX: $PREFIX"
    echo "LOG_DIR: $LOG_DIR"
    echo "JOBS: $JOBS"
    echo "DEVICE: $DEVICE"
    echo "CUDA_ARCH: $CUDA_ARCH"
    echo "HIP_ARCH: $HIP_ARCH"
    echo "CC: $CC"
    echo "CXX: $CXX"
    echo ""
    echo "Tool versions:"
    command -v "$CC" >/dev/null 2>&1 && "$CC" --version | head -n 1 || true
    command -v "$CXX" >/dev/null 2>&1 && "$CXX" --version | head -n 1 || true
    command -v cmake >/dev/null 2>&1 && cmake --version | head -n 1 || true
    command -v git >/dev/null 2>&1 && git --version || true
    command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n 1 || true
    command -v hipcc >/dev/null 2>&1 && hipcc --version | head -n 1 || true
    echo ""
  } | tee "$MANIFEST"
}

clone_or_update() {
  local repo="$1"
  local ref="$2"
  local dir="$3"

  if [[ ! -d "$dir/.git" ]]; then
    run_logged "clone-$(basename "$dir")" git clone "$repo" "$dir"
  fi

  cd "$dir"
  run_logged "fetch-$(basename "$dir")" git fetch --all --tags
  run_logged "checkout-$(basename "$dir")" git checkout "$ref"

  log ""
  log "Repo provenance: $dir"
  log "Remote:"
  git remote -v | tee -a "$MANIFEST"
  log "HEAD:"
  git rev-parse HEAD | tee -a "$MANIFEST"
  log "Describe:"
  git describe --tags --always --dirty 2>/dev/null | tee -a "$MANIFEST" || true
  log "Status:"
  git status --short | tee -a "$MANIFEST"
  cd - >/dev/null
}

build_gklib() {
  local src="$SRC_DIR/GKlib"

  cd "$src"

  run_logged "gklib-distclean" make distclean || true
  run_logged "gklib-config" make config shared=0 cc="$CC" prefix="$PREFIX"
  run_logged "gklib-build" make -j "$JOBS"
  run_logged "gklib-install" make install

  cd - >/dev/null
}

build_metis() {
  local src="$SRC_DIR/METIS"

  cd "$src"

  run_logged "metis-distclean" make distclean || true
  #  gklib_path="$SRC_DIR/GKlib"
  run_logged "metis-config" make config shared=0 cc="$CC" prefix="$PREFIX"
  run_logged "metis-build" make -j "$JOBS"
  run_logged "metis-install" make install

  cd - >/dev/null
}

build_plato() {
    local src="$SRC_DIR/plato"

    cd "$src"

    run_logged "plato-distclean" make distclean || true
    run_logged "plato-autogen" ./autogen.sh
    run_logged "plato-config" ./configure --prefix="$PREFIX"
    run_logged "plato-build" make
    run_logged "plato-install" make install

    cd - >/dev/null
}

build_thermo_database() {
    local src="$SRC_DIR/database"
    cd $SRC_DIR
    local asrc="$(pwd)"

    cd "$PREFIX"
    ln -s "${asrc}/database" .
    cd - >/dev/null
}

build_hypre() {
  local src="$SRC_DIR/hypre/src"

  cd "$src"

  local opts=(
    "--with-MPI"
    "--disable-fortran"
    "--prefix=$PREFIX"
    "--enable-mixedint"
  )

  case "$DEVICE" in
    cpu)
      ;;
    cuda)
      opts+=("--with-cuda" "--with-gpu-arch=$CUDA_ARCH")
      ;;
    hip)
      opts+=("--with-hip" "--enable-mixedint")
      if [[ -n "$HIP_ARCH" ]]; then
        opts+=("--with-gpu-arch=$HIP_ARCH")
      fi
      ;;
    *)
      echo "ERROR: unknown DEVICE=$DEVICE. Use cpu, cuda, or hip."
      exit 1
      ;;
  esac

  log ""
  log "HYPRE configure options:"
  printf '  %q\n' ./configure "${opts[@]}" | tee -a "$MANIFEST"

  run_logged "hypre-configure" ./configure "${opts[@]}"
  run_logged "hypre-build" make -j "$JOBS"
  run_logged "hypre-install" make install

  cd - >/dev/null
}

build_mfem() {
  local src="$SRC_DIR/mfem"
  local build="$BUILD_DIR/mfem"

  rm -rf "$build"
  mkdir -p "$build"

  cd "$build"

  local opts=(
    "$src"
    "-DCMAKE_INSTALL_PREFIX=$PREFIX"
    "-DCMAKE_PREFIX_PATH=$PREFIX"
    "-DMFEM_USE_MPI=YES"
    "-DMFEM_USE_METIS_5=YES"
  )

  case "$DEVICE" in
    cpu)
      ;;
    cuda)
      opts+=("-DMFEM_USE_CUDA=YES" "-DMFEM_CUDA_ARCH=$CUDA_ARCH")
      ;;
    hip)
      opts+=("-DMFEM_USE_HIP=YES")
      ;;
  esac

  log ""
  log "MFEM CMake options:"
  printf '  %q\n' cmake "${opts[@]}" | tee -a "$MANIFEST"

  run_logged "mfem-cmake" cmake "${opts[@]}"
  run_logged "mfem-build" make -j "$JOBS"
  run_logged "mfem-install" make install

  cd - >/dev/null
}

record_final_summary() {
  log ""
  log "Installed TPL prefix:"
  log "$PREFIX"
  log ""
  log "Use for your application:"
  log "  -DCMAKE_PREFIX_PATH=$PREFIX"
  log ""
  log "Manifest:"
  log "$MANIFEST"
}

record_system_info

clone_or_update "$PLATO_REPO" "$PLATO_REF" "$SRC_DIR/plato"
clone_or_update "$THERMO_DATABASE_REPO" "$THERMO_DATABASE_REF" "$SRC_DIR/database"
clone_or_update "$GKLIB_REPO" "$GKLIB_REF" "$SRC_DIR/GKlib"
clone_or_update "$METIS_REPO" "$METIS_REF" "$SRC_DIR/METIS"
clone_or_update "$HYPRE_REPO" "$HYPRE_REF" "$SRC_DIR/hypre"
clone_or_update "$MFEM_REPO" "$MFEM_REF" "$SRC_DIR/mfem"

build_plato
build_thermo_database
build_gklib
build_metis
build_hypre
build_mfem

record_final_summary
