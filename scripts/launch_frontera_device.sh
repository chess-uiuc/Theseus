#!/usr/bin/env bash
set -euo pipefail

# echo "RANK-local ENV:"
# env

# Intel MPI local rank variable
LOCAL_RANK="${I_MPI_LOCAL_RANK:-}"

# Fallbacks for other MPIs
if [[ -z "${LOCAL_RANK}" ]]; then
    LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-0}}}"
fi

NGPU=${NGPU:-4}
GPU_ID=$(( LOCAL_RANK % NGPU ))

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "host=$(hostname -s) local_rank=${LOCAL_RANK} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2

exec "$@"
#exec compute-sanitizer --tool memcheck --show-backtrace yes "$@"
#exec gdb -batch \
#    -ex 'set pagination off' \
#    -ex 'break mfem::mfem_error' \
#    -ex run \
#    -ex 'bt 12' \
#    --args "$@"
