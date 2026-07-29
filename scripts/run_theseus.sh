#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Set PYTHON=/path/to/python3." >&2
  exit 2
fi

# --- Harness state (do NOT set -e inside per-case runs) ---
declare -a SUCCEEDED=()
declare -a FAILED=()

# Zero-pad cycle index to 6 digits: 0 -> 000000, 100 -> 000100
fmt_cycle() {
  printf "%06d" "${1:-0}"
}

# Verify expected ParaView outputs exist for N steps under out/ParaView
# Requires ParaView.pvd and both Cycle000000 and Cycle00NNNN
check_outputs() {
  local outdir="$1" ; local nsteps="$2"
  local pv="${outdir}/ParaView/ParaView.pvd"
  local c0="${outdir}/ParaView/Cycle$(fmt_cycle 0)"
  local cN="${outdir}/ParaView/Cycle$(fmt_cycle "${nsteps}")"
  [[ -f "${pv}" && -d "${c0}" && -d "${cN}" ]]
}


# Default knobs
NSTEPS=0
TOP=$(pwd)
BUILDDIR="${TOP}/build"
EXE="${BUILDDIR}/theseus"
OUTDIR="RunTheseus"
LISTFILE=""
ONECFG=""
CFL=""
CFL_OVERRIDE=0
DT=0.0001
NSTEPS_OVERRIDE=0
DT_OVERRIDE=0
NMPIRANKS=2
DEVICE="cpu"
NHOSTS="1"
UTILDIR="${TOP}/scripts"
HOST_SHORT="$(hostname -s)"
MESHNAME="default"
MESH_OVERRIDE=0
ORDER="default"
ORDER_OVERRIDE=0
CHECKOUT=1
MSHREF_OVERRIDE=0
MSHREF_LVL="default"
DISABLE_VIZ=0

usage() {
  cat <<EOF
Usage: $0 [-n STEPS] [-b BUILDDIR] [-e EXECUTABLE] [-H NUMHOSTS] [-o RUNDIR] [-p NUMPROC] [-r DEVICE] [-m MESHNAME] [-x REFLEVEL] [-y ORDER] [-k] [-z] (-c CONFIG.json | -l LIST.txt)

  -n STEPS      Number of steps to run (default: None, use case default)
  -t TIMESTEP   Fixed timestep size (default: None, use case default)
  -s CFL        Fixed CFL (default: None, use case default)
  -b BUILDDIR   Build directory (default: ${BUILDDIR})
  -e EXECUTABLE Path to Theseus executable (default: ${EXE})
  -o OUTDIR     Directory to create run output (default: ${OUTDIR})
  -c CONFIG     Single example config.json to run
  -H NHOSTS     Number of compute nodes to use (default: 1)
  -h            Show this help message
  -p NUMPROC    Number of MPI processes to run
  -r DEVICE     Compute device to run on (e.g. cpu, hip, or cuda, default: cpu)
  -k            Disable output check
  -l LIST       List file with one config.json path per line (comments (#) allowed)
  -m MESHNAME   Replace the meshname with this one
  -x REFLEVEL   Mesh refinement level
  -y ORDER      Polynomial order for spatial discretization
  -z            Disable visualization output
Examples:
  $0 -c TestCases/NavierStokes/2D/LidDrivenCavity/config.json
  $0 -l examples.txt
EOF
}

# ---- Parse args
while getopts ":zkx:y:m:n:t:s:b:e:o:p:r:c:l:H:h" opt; do
  case $opt in
      n) NSTEPS="${OPTARG}"; NSTEPS_OVERRIDE=1;;
      t) DT="${OPTARG}"; DT_OVERRIDE=1;;
      s) CFL="${OPTARG}"; CFL_OVERRIDE=1;;
      b) BUILDDIR="${OPTARG}"; EXE="${BUILDDIR}/theseus";;
      e) EXE="${OPTARG}";;
      o) OUTDIR="${OPTARG}";;
      p) NMPIRANKS="${OPTARG}";;
      H) NHOSTS="${OPTARG}";;
      r) DEVICE="${OPTARG}";;
      c) ONECFG="${OPTARG}";;
      k) CHECKOUT=0;;
      z) DISABLE_VIZ=1; CHECKOUT=0;;
      l) LISTFILE="${OPTARG}";;
      m) MESHNAME="${OPTARG}"; MESH_OVERRIDE=1;;
      x) MSHREF_LVL="${OPTARG}"; MSHREF_OVERRIDE=1;;
      y) ORDER="${OPTARG}"; ORDER_OVERRIDE=1;;
      h) usage; exit 0;;
      \?) echo "Unknown option -$OPTARG" >&2; usage; exit 2;;
      :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2;;
  esac
done

if [[ ! -x "${EXE}" ]]; then
  echo "ERROR: Theseus executable not found at ${EXE}" >&2
  exit 2
fi

# ---- Resolve which configs to run
declare -a CFGS
if [[ -n "${ONECFG}" && -n "${LISTFILE}" ]]; then
  echo "ERROR: choose either -c or -l, not both." >&2; exit 2
elif [[ -n "${ONECFG}" ]]; then
  CFGS+=("${ONECFG}")
elif [[ -n "${LISTFILE}" ]]; then
  while IFS= read -r line; do
    # skip empties and comments
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    CFGS+=("${line}")
  done < "${LISTFILE}"
else
  echo "ERROR: must provide -c CONFIG.json or -l LIST.txt" >&2; exit 2
fi

# ---- Ensure run sandbox
RUNDIR="${TOP}/${OUTDIR}"
mkdir -p "${RUNDIR}"
# Copy the executable into the run dir (your preferred workflow)
cp -f "${EXE}" "${RUNDIR}/theseus"

# ---- Function to run one example
run_one() {
  local cfg_rel="$1"
  local cfg_abs
  cfg_abs="$(cd "$(dirname "${cfg_rel}")" && pwd)/$(basename "${cfg_rel}")"

  if [[ ! -f "${cfg_abs}" ]]; then
    echo "ERROR: config not found: ${cfg_rel}" >&2
    return 1
  fi

  echo "==> Running example: ${cfg_rel} with ${NMPIRANKS} MPI procs."
  echo "    Working dir: ${RUNDIR}"
  echo "    Compute device: ${DEVICE}"

  # Prepare per-example working area
  local runname
  runname="$(basename "$(dirname "${cfg_abs}")")"   # e.g., LidDrivenCavity
  local work="${RUNDIR}/${runname}"
  rm -rf "${work}"
  mkdir -p "${work}"
  local outdir="${work}"


  local patched="${work}/config.patched.json"
  local nsteps="${NSTEPS}"

  if [[ "${nsteps}" == "0" ]]; then
      nsteps=100
  fi

  python3 - "${cfg_abs}" "${patched}" "${nsteps}" "${DT}" "${MESHNAME}" "${ORDER}" "${CFL}" "${MSHREF_LVL}"\
      "${NSTEPS_OVERRIDE}" "${DT_OVERRIDE}" "${MESH_OVERRIDE}" "${ORDER_OVERRIDE}" "${CFL_OVERRIDE}" "${MSHREF_OVERRIDE}" "${DISABLE_VIZ}" << 'PY'
import json
import sys
import os

src, dst, nsteps_s, dt_s, meshname, order_s, cfl_s, reflvl_s, nsteps_override_s, dt_override_s, mesh_override_s, order_override_s, cfl_override_s, ref_override_s, disable_viz_s = sys.argv[1:]

nsteps = int(nsteps_s)
dt = float(dt_s)
nsteps_override = int(nsteps_override_s)
cfl_override = int(cfl_override_s)
dt_override = int(dt_override_s)
mesh_override = int(mesh_override_s)
order_override = int(order_override_s)
ref_override = int(ref_override_s)
disable_viz = int(disable_viz_s)
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

rt = cfg.setdefault("runTime", {})

if disable_viz > 0:
    rt["visualize"] = False
else:
    rt["visualize"] = True
rt["paraview"] = True
rt["visit"] = False
rt["nancheck"] = True
rt["output_file_path"] = "./"
rt["checkpoint_load"] = False

if order_override:
   order = int(order_s)
   rt["order"] = order

if mesh_override:
    current_file = rt["mesh_file"]
    current_dir = os.path.dirname(current_file)
    rt["mesh_file"] = os.path.join(current_dir, meshname) if current_dir else meshname

if ref_override:
    ref_lvl = int(reflvl_s)
    rt["par_ref_levels"] = ref_lvl

if dt_override:
    rt["variable_dt"] = False
    rt["dt"] = dt

if cfl_override:
    cfl = float(cfl_s)
    rt["cfl"] = cfl

if nsteps_override:
    rt["nsteps_max"] = nsteps

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

printf "Theseus configuration:\n"
cat "${patched}"

local -a MPI_LAUNCHER="mpiexec -n ${NMPIRANKS}"
local -a LAUNCH_UTIL=""

# if [[ "${DEVICE}" != "cpu" ]]; then
# fi

# Override MPI_LAUNCHER if required for this platform:
case "${HOST_SHORT}" in
    tuo*)
        # Tuolumne@LC
        MPI_LAUNCHER="flux run --exclusive -N ${NHOSTS} -n ${NMPIRANKS}"
        ;;
    front*)
        # Frontera@TACC
        MPI_LAUNCHER="ibrun -n ${NMPIRANKS}"
	# Determine if any launcher is required ...
	cp "${UTILDIR}/launch_frontera_device.sh" "${RUNDIR}/device_launcher.sh"
	LAUNCH_UTIL="../device_launcher.sh"
        ;;
    c[0-9]*-[0-9]*)
        # Frontera@TACC compute nodes
        MPI_LAUNCHER="ibrun -n ${NMPIRANKS}"
	# Determine if any launcher is required ...
	cp "${UTILDIR}/launch_frontera_device.sh" "${RUNDIR}/device_launcher.sh"
	LAUNCH_UTIL="../device_launcher.sh"
        ;;
esac
echo "mpi launcher: ${MPI_LAUNCHER} ${LAUNCH_UTIL}"
# Run from the per-example dir; keep your “two levels down” invariant
# Run example (isolate failures; do NOT exit on first error)
# mpiexec -n "${NMPIRANKS}" 
set +e
( cd "${work}" && eval ${MPI_LAUNCHER} ${LAUNCH_UTIL} ../theseus -d "${DEVICE}" -c "${patched}" )
local run_rc=$?
set -e

# Basic regression: require ParaView.pvd + Cycle000000 + Cycle00NNNN
CHECK=1
if [[ ${CHECKOUT} -eq 1 ]]; then
    if check_outputs "${outdir}" "${NSTEPS}"; then
        CHECK=1
    else
        CHECK=0
    fi
fi

if [[ ${run_rc} -eq 0 ]] && [[ ${CHECK} -eq 1 ]]; then
    if [[ ${CHECKOUT} -eq 1 ]]; then
	echo "✓ Theseus ran OK: ${runname} (outputs in ${outdir})"
    else
	echo "✓ Theseus ran OK: ${runname}"
    fi
    SUCCEEDED+=("${cfg_rel}")
    return 0
else
    echo "✗ Theseus run FAILED: ${runname}"
    [[ ${run_rc} -ne 0 ]] && echo "  - runtime exit code: ${run_rc}"

    if [[ ${CHECKOUT} -eq 1 ]]; then
        [[ ! -f "${outdir}/ParaView/ParaView.pvd" ]] && \
            echo "  - missing: ${outdir}/ParaView/ParaView.pvd"

        [[ ! -d "${outdir}/ParaView/Cycle$(fmt_cycle 0)" ]] && \
            echo "  - missing: ${outdir}/ParaView/Cycle$(fmt_cycle 0)"
	if [[ ${NSTEPS_OVERRIDE} -eq 1 ]]; then
            [[ ! -d "${outdir}/ParaView/Cycle$(fmt_cycle "${NSTEPS}")" ]] && \
		echo "  - missing: ${outdir}/ParaView/Cycle$(fmt_cycle "${NSTEPS}")"
	fi
    fi

    FAILED+=("${cfg_rel}")
    return 1
fi

}

# ---- Iterate
rc=0
for cfg in "${CFGS[@]}"; do
    run_one "${cfg}" || rc=1
done

# ---- Summary
echo
echo "===== Theseus Runtime Summary ====="
echo "Total: ${#CFGS[@]} | Succeeded: ${#SUCCEEDED[@]} | Failed: ${#FAILED[@]}"
if (( ${#SUCCEEDED[@]} > 0 )); then
    printf '  ✓ %s\n' "${SUCCEEDED[@]}"
fi
if (( ${#FAILED[@]} > 0 )); then
    printf '  ✗ %s\n' "${FAILED[@]}"
fi

exit ${rc}
