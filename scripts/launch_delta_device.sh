#!/usr/bin/env bash
set -euo pipefail

LOCAL_RANK="${SLURM_LOCALID:-}"

if [[ -z "${LOCAL_RANK}" ]]; then
    LOCAL_RANK="${I_MPI_LOCAL_RANK:-}"
fi

if [[ -z "${LOCAL_RANK}" ]]; then
    LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-0}}}"
fi

NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-4}}"
GPU_ID=$(( LOCAL_RANK % NGPU ))

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "host=$(hostname -s) \
rank=${SLURM_PROCID:-?} \
local_rank=${LOCAL_RANK} \
gpu=${GPU_ID} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2

exec "$@"
