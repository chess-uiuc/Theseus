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

# Verify expected ParaView outputs under out/ParaView. Explicit step-count
# runs require Cycle00NNNN. Final-time runs require distinct first and last
# datasets referenced by ParaView.pvd.
check_outputs() {
  local outdir="$1" ; local nsteps_override="$2" ; local nsteps="$3"
  local pv="${outdir}/ParaView/ParaView.pvd"
  local c0="${outdir}/ParaView/Cycle$(fmt_cycle 0)"

  [[ -f "${pv}" && -d "${c0}" ]] || return 1

  if [[ "${nsteps_override}" -eq 1 ]]; then
    [[ -d "${outdir}/ParaView/Cycle$(fmt_cycle "${nsteps}")" ]]
    return
  fi

  "${PYTHON}" - "${pv}" <<'PY'
import os
import sys
import xml.etree.ElementTree as ET

pvd = sys.argv[1]
datasets = ET.parse(pvd).getroot().findall(".//DataSet")
if len(datasets) < 2:
    raise SystemExit(1)

first = datasets[0].attrib.get("file")
last = datasets[-1].attrib.get("file")
if not first or not last or first == last:
    raise SystemExit(1)

base = os.path.dirname(pvd)
if not os.path.exists(os.path.join(base, first)):
    raise SystemExit(1)
if not os.path.exists(os.path.join(base, last)):
    raise SystemExit(1)
PY
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
MPI_LAUNCHER_OVERRIDE="${THESEUS_MPI_LAUNCHER:-}"
MESHNAME="default"
MESH_OVERRIDE=0
ORDER="default"
ORDER_OVERRIDE=0
CHECKOUT=1
MSHREF_OVERRIDE=0
MSHREF_LVL="default"
DISABLE_VIZ=0
ALLOW_RESTART=0
RESTART_SELECTION=""
WIPE_RUNDIR=0
PRESERVE_CONFIG=0
DEBUG=0

usage() {
  cat <<EOF
Usage: $0 [-n STEPS] [-b BUILDDIR] [-e EXECUTABLE] [-H NUMHOSTS] [-o RUNDIR] [-p NUMPROC] [-r DEVICE] [-m MESHNAME] [-x REFLEVEL] [-y ORDER] [-d] [-k] [-P] [-R | -L | -C CYCLE] [-w] [-z] (-c CONFIG.json | -l LIST.txt)

  -n STEPS      Number of steps to run (default: None, use case default)
  -t TIMESTEP   Fixed timestep size (default: None, use case default)
  -s CFL        Fixed CFL (default: None, use case default)
  -b BUILDDIR   Build directory (default: ${BUILDDIR})
  -d            Set debug flag for simulation
  -e EXECUTABLE Path to Theseus executable (default: ${EXE})
  -o OUTDIR     Directory to create run output (default: ${OUTDIR})
  -c CONFIG     Single example config.json to run
  -H NHOSTS     Number of compute nodes to use (default: 1)
  -h            Show this help message
  -p NUMPROC    Number of MPI processes to run
  -r DEVICE     Compute device to run on (e.g. cpu, hip, or cuda, default: cpu)
  -k            Disable output check
  -l LIST       List file with one config.json path per line (comments (#) allowed)
  -m MESH       Replace the configured mesh. A filename keeps the configured
                directory; a relative path is resolved from the launch directory;
                an absolute path is used unchanged.
  -w            Wipe out the run directory before starting
  -x REFLEVEL   Mesh refinement level
  -y ORDER      Polynomial order for spatial discretization
  -z            Disable visualization output
  -P            Preserve input settings except explicit overrides; use -k for custom output checks
  -R            Preserve checkpoint_load from the input configuration
  -L            Restart from the latest complete checkpoint in the run output
  -C CYCLE      Restart from the specified complete checkpoint cycle
                -L and -C enable restart; neither may be combined with -w
Examples:
  $0 -c TestCases/NavierStokes/2D/LidDrivenCavity/config.json
  $0 -l examples.txt
EOF
}

# ---- Parse args
while getopts ":PdRLzkwC:x:y:m:n:t:s:b:e:o:p:r:c:l:H:h" opt; do
  case $opt in
      n) NSTEPS="${OPTARG}"; NSTEPS_OVERRIDE=1;;
      t) DT="${OPTARG}"; DT_OVERRIDE=1;;
      s) CFL="${OPTARG}"; CFL_OVERRIDE=1;;
      b) BUILDDIR="${OPTARG}"; EXE="${BUILDDIR}/theseus";;
      d) DEBUG=1;;
      e) EXE="${OPTARG}";;
      o) OUTDIR="${OPTARG}";;
      p) NMPIRANKS="${OPTARG}";;
      H) NHOSTS="${OPTARG}";;
      r) DEVICE="${OPTARG}";;
      c) ONECFG="${OPTARG}";;
      k) CHECKOUT=0;;
      z) DISABLE_VIZ=1; CHECKOUT=0;;
      R) ALLOW_RESTART=1;;
      L)
          if [[ -n "${RESTART_SELECTION}" ]]; then
              echo "ERROR: choose one restart selector: -L or -C CYCLE." >&2
              exit 2
          fi
          RESTART_SELECTION="latest"
          ALLOW_RESTART=1;;
      C)
          if [[ -n "${RESTART_SELECTION}" || ! "${OPTARG}" =~ ^[0-9]+$ ||
                ! "${OPTARG}" =~ [1-9] ]]; then
              echo "ERROR: -C requires a positive cycle and cannot be combined with -L." >&2
              exit 2
          fi
          RESTART_SELECTION="${OPTARG}"
          ALLOW_RESTART=1;;
      P) PRESERVE_CONFIG=1;;
      l) LISTFILE="${OPTARG}";;
      m) MESHNAME="${OPTARG}"; MESH_OVERRIDE=1;;
      w) WIPE_RUNDIR=1;;
      x) MSHREF_LVL="${OPTARG}"; MSHREF_OVERRIDE=1;;
      y) ORDER="${OPTARG}"; ORDER_OVERRIDE=1;;
      h) usage; exit 0;;
      \?) echo "Unknown option -$OPTARG" >&2; usage; exit 2;;
      :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2;;
  esac
done

if [[ -n "${RESTART_SELECTION}" && ${WIPE_RUNDIR} -eq 1 ]]; then
  echo "ERROR: cannot combine checkpoint selection (-L/-C) with -w." >&2
  exit 2
fi

if [[ "${DT_OVERRIDE}" -eq 1 && "${CFL_OVERRIDE}" -eq 1 ]]; then
  echo "ERROR: choose either -t TIMESTEP or -s CFL, not both." >&2
  exit 2
fi

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
case "${OUTDIR}" in
    /*) RUNDIR="${OUTDIR}";;
    *)  RUNDIR="${TOP}/${OUTDIR}";;
esac
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
  if [[ ${ALLOW_RESTART} -eq 0 ]]; then
     rm -rf "${work}"
  else
      if [[ ${WIPE_RUNDIR} -eq 1 ]]; then
	  printf "WARNING: Restart AND Wipe enabled, removing target work directory."
      fi
  fi
  if [[ ${WIPE_RUNDIR} -eq 1 ]]; then
      rm -rf "${work}"
  fi
  mkdir -p "${work}"
  local outdir="${work}"


  local patched="${work}/config.patched.json"
  local nsteps="${NSTEPS}"

  if [[ "${nsteps}" == "0" ]]; then
      nsteps=100
  fi

  if ! "${PYTHON}" - "${cfg_abs}" "${patched}" "${nsteps}" "${DT}" "${MESHNAME}" "${ORDER}" "${CFL}" "${MSHREF_LVL}" "${TOP}"\
      "${NSTEPS_OVERRIDE}" "${DT_OVERRIDE}" "${MESH_OVERRIDE}" "${ORDER_OVERRIDE}" "${CFL_OVERRIDE}" "${MSHREF_OVERRIDE}"\
      "${DISABLE_VIZ}" "${ALLOW_RESTART}" "${PRESERVE_CONFIG}" "${DEBUG}" "${RESTART_SELECTION}" "${NMPIRANKS}" << 'PY'
import json
import sys
import os
from pathlib import Path
import re

src, dst, nsteps_s, dt_s, meshname, order_s, cfl_s, reflvl_s, launch_dir, nsteps_override_s, dt_override_s, mesh_override_s, order_override_s, cfl_override_s, ref_override_s, disable_viz_s, allow_restart_s, preserve_config_s, debug_sim, restart_selection, ranks_s = sys.argv[1:]

nsteps = int(nsteps_s)
dt = float(dt_s)
nsteps_override = int(nsteps_override_s)
cfl_override = int(cfl_override_s)
debug = int(debug_sim)
dt_override = int(dt_override_s)
mesh_override = int(mesh_override_s)
order_override = int(order_override_s)
ref_override = int(ref_override_s)
disable_viz = int(disable_viz_s)
allow_restart = int(allow_restart_s)
preserve_config = int(preserve_config_s)
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

rt = cfg.setdefault("runTime", {})

if debug:
    rt["debug"] = True

if not preserve_config:
    rt["visualize"] = True
    rt["paraview"] = True
    rt["visit"] = False
    rt["nancheck"] = True
    rt["output_file_path"] = "./"
    if not allow_restart:
        rt["checkpoint_load"] = False

if disable_viz:
    rt["visualize"] = False

if order_override:
   order = int(order_s)
   rt["order"] = order

if mesh_override:
    if os.path.isabs(meshname):
        rt["mesh_file"] = meshname
    elif os.path.dirname(meshname):
        rt["mesh_file"] = os.path.abspath(os.path.join(launch_dir, meshname))
    else:
        current_dir = os.path.dirname(rt["mesh_file"])
        rt["mesh_file"] = os.path.join(current_dir, meshname) if current_dir else meshname

if ref_override:
    ref_lvl = int(reflvl_s)
    rt["par_ref_levels"] = ref_lvl

if dt_override:
    rt["variable_dt"] = False
    rt["dt"] = dt
elif cfl_override:
    cfl = float(cfl_s)
    rt["variable_dt"] = True
    rt["cfl"] = cfl

if nsteps_override:
    rt["nsteps_max"] = nsteps

if restart_selection:
    work = Path(dst).parent
    output = work / rt.get("output_file_path", ".")
    checkpoints = output / rt.get("checkpoints_folder", "Checkpoints")
    ranks = int(ranks_s)

    def reject_reason(cycle):
        directory = checkpoints / f"Cycle{cycle}"
        metadata_path = directory / f"checkpoint_cycle_{cycle}.json"
        try:
            metadata = json.loads(metadata_path.read_text())
            if not isinstance(metadata, dict) or metadata.get("cycle") != cycle:
                return "missing or inconsistent cycle metadata"
            if metadata.get("mpi_ranks", ranks) != ranks:
                return "MPI rank count differs"
            for key, setting in (("order", "order"), ("dimension", "dim"),
                                 ("num_equations", "num_equations")):
                if key in metadata and setting in rt and metadata[key] != rt[setting]:
                    return f"{key} differs"
            expected = {f"checkpoint_cycle_{cycle}.{rank:08d}.chk" for rank in range(ranks)}
            actual = {path.name for path in directory.glob(f"checkpoint_cycle_{cycle}.*.chk")}
            if actual != expected:
                return "missing rank files or different MPI layout"
            if any((directory / name).stat().st_size == 0 for name in expected):
                return "empty rank file"
        except (OSError, ValueError, TypeError) as error:
            return str(error)
        return None

    if restart_selection == "latest":
        candidates = []
        for directory in checkpoints.glob("Cycle*"):
            match = re.fullmatch(r"Cycle([1-9][0-9]*)", directory.name)
            if directory.is_dir() and match:
                candidates.append(int(match.group(1)))
        selected = None
        for cycle in sorted(candidates, reverse=True):
            reason = reject_reason(cycle)
            if reason is None:
                selected = cycle
                break
            print(f"Skipping checkpoint cycle {cycle}: {reason}", flush=True)
        if selected is None:
            raise SystemExit(f"ERROR: no complete matching checkpoint found in {checkpoints}")
    else:
        selected = int(restart_selection)
        reason = reject_reason(selected)
        if reason is not None:
            raise SystemExit(f"ERROR: checkpoint cycle {selected} cannot be selected: {reason}")
    rt["checkpoint_load"] = True
    rt["checkpoint_cycle"] = selected
    print(f"Restart selected: cycle {selected} from {checkpoints / f'Cycle{selected}'}", flush=True)

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
  then
      FAILED+=("${cfg_rel}")
      return 1
  fi

printf "Theseus configuration:\n"
cat "${patched}"

local -a run_command
if [[ -n "${MPI_LAUNCHER_OVERRIDE}" ]]; then
    if ! command -v "${MPI_LAUNCHER_OVERRIDE}" >/dev/null 2>&1; then
        echo "ERROR: THESEUS_MPI_LAUNCHER is not executable: ${MPI_LAUNCHER_OVERRIDE}" >&2
        return 2
    fi
    run_command=("${MPI_LAUNCHER_OVERRIDE}" -n "${NMPIRANKS}")
else
    run_command=(mpiexec -n "${NMPIRANKS}")
    case "${HOST_SHORT}" in
    tuo*)
        # Tuolumne@LC
        run_command=(flux run --exclusive -N "${NHOSTS}" -n "${NMPIRANKS}")
        ;;
    front*)
        # Frontera@TACC
	cp "${UTILDIR}/launch_frontera_device.sh" "${RUNDIR}/device_launcher.sh"
        run_command=(ibrun -n "${NMPIRANKS}" ../device_launcher.sh)
        ;;
    c[0-9]*-[0-9]*)
        # Frontera@TACC compute nodes
	cp "${UTILDIR}/launch_frontera_device.sh" "${RUNDIR}/device_launcher.sh"
        run_command=(ibrun -n "${NMPIRANKS}" ../device_launcher.sh)
        ;;
    gh*)
        # GraceHopper/Delta
	cp "${UTILDIR}/launch_delta_device.sh" "${RUNDIR}/device_launcher.sh"
        run_command=(srun -N "${NHOSTS}" -n "${NMPIRANKS}" ../device_launcher.sh)
        ;;
    esac
fi
run_command+=(../theseus -d "${DEVICE}" -c "${patched}")
printf 'run command:'
printf ' %q' "${run_command[@]}"
printf '\n'
# Run from the per-example dir; keep your “two levels down” invariant
# Run example (isolate failures; do NOT exit on first error)
# mpiexec -n "${NMPIRANKS}" 
set +e
( cd "${work}" && "${run_command[@]}" )
local run_rc=$?
set -e

# Basic regression: require ParaView.pvd + Cycle000000 + Cycle00NNNN
CHECK=1
if [[ ${CHECKOUT} -eq 1 ]]; then
    if check_outputs "${outdir}" "${NSTEPS_OVERRIDE}" "${NSTEPS}"; then
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
	else
	    echo "  - ParaView.pvd must reference distinct first and last datasets that exist"
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
