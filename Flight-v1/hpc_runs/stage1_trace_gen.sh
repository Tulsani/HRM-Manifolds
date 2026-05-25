#!/usr/bin/env bash
#SBATCH --job-name=flight-stage1-traces
#SBATCH --output=hpc_runs/logs/%x_%j.out
#SBATCH --error=hpc_runs/logs/%x_%j.err
#SBATCH --partition=a100_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=at6646@nyu.edu

set -eo pipefail

# Environment
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

CONDA_ENV="${CONDA_ENV:-flight-v1}"
if [ "${SKIP_CONDA:-0}" != "1" ]; then
    conda activate "${CONDA_ENV}"
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/trace_gen.yaml}"
LOG_DIR="${PROJECT_DIR}/hpc_runs/logs"
TRACE_DIR="${TRACE_DIR:-${PROJECT_DIR}/trace_library}"

mkdir -p "${LOG_DIR}" "${TRACE_DIR}"

# Pre-flight checks
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Missing config: ${CONFIG}"
    exit 1
fi

echo "============================================================"
echo "Stage       : 1 - teacher trace generation"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "Project dir : ${PROJECT_DIR}"
echo "Config      : ${CONFIG}"
echo "Trace dir   : ${TRACE_DIR}"
echo "Conda env   : ${CONDA_ENV}"
echo "Start time  : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

set +e
python3 scripts/run_trace_gen.py --config "${CONFIG}"
EXIT_CODE=$?
set -e

echo "============================================================"
echo "Stage 1 finished : $(date)"
echo "Exit code        : ${EXIT_CODE}"
if [ -d "${TRACE_DIR}" ]; then
    N_TRACE_FILES=$(find "${TRACE_DIR}" -type f | wc -l | tr -d ' ')
    echo "Trace files      : ${N_TRACE_FILES}"
fi
echo "============================================================"

exit "${EXIT_CODE}"
