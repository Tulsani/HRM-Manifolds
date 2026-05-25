#!/usr/bin/env bash
#SBATCH --job-name=flight-stage4-router
#SBATCH --output=hpc_runs/logs/%x_%j.out
#SBATCH --error=hpc_runs/logs/%x_%j.err
#SBATCH --partition=a100_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
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
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/stage4.yaml}"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/hpc_runs/logs"
EMBEDDINGS="${PROJECT_DIR}/trace_library/embeddings/problem_embeddings.npz"
SKILL_LABELS="${PROJECT_DIR}/outputs/skill_labels.json"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${PROJECT_DIR}/outputs"

# Pre-flight checks
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Missing config: ${CONFIG}"
    exit 1
fi

if [ ! -f "${EMBEDDINGS}" ]; then
    echo "ERROR: Missing embeddings: ${EMBEDDINGS}"
    echo "Run stage2_geometry.sh first."
    exit 1
fi

if [ ! -f "${SKILL_LABELS}" ]; then
    echo "ERROR: Missing skill labels: ${SKILL_LABELS}"
    echo "Run stage2_geometry.sh first."
    exit 1
fi

echo "============================================================"
echo "Stage       : 4 - router training"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "Project dir : ${PROJECT_DIR}"
echo "Config      : ${CONFIG}"
echo "Embeddings  : ${EMBEDDINGS}"
echo "Checkpoints : ${CKPT_DIR}"
echo "Conda env   : ${CONDA_ENV}"
echo "Start time  : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

set +e
python3 scripts/run_stage4.py --config "${CONFIG}"
EXIT_CODE=$?
set -e

echo "============================================================"
echo "Stage 4 finished : $(date)"
echo "Exit code        : ${EXIT_CODE}"
if [ -f "${CKPT_DIR}/router_stage4.pt" ]; then
    echo "Router ckpt      : ${CKPT_DIR}/router_stage4.pt"
fi
if [ -f "${PROJECT_DIR}/outputs/router_calibration.json" ]; then
    echo "Calibration      : ${PROJECT_DIR}/outputs/router_calibration.json"
fi
echo "============================================================"

exit "${EXIT_CODE}"
