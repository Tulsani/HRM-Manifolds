#!/usr/bin/env bash
#SBATCH --job-name=flight-stage6-final
#SBATCH --output=hpc_runs/logs/%x_%j.out
#SBATCH --error=hpc_runs/logs/%x_%j.err
#SBATCH --partition=a100_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/stage6.yaml}"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/hpc_runs/logs"
SYSTEM_CKPT="${CKPT_DIR}/full_system_stage5.pt"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${PROJECT_DIR}/outputs"

# Pre-flight checks
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Missing config: ${CONFIG}"
    exit 1
fi

if [ ! -f "${SYSTEM_CKPT}" ]; then
    echo "ERROR: Missing full-system checkpoint: ${SYSTEM_CKPT}"
    echo "Run stage5_distill.sh first."
    exit 1
fi

echo "============================================================"
echo "Stage       : 6 - final fine-tuning, calibration, evaluation"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "Project dir : ${PROJECT_DIR}"
echo "Config      : ${CONFIG}"
echo "System ckpt : ${SYSTEM_CKPT}"
echo "Conda env   : ${CONDA_ENV}"
echo "Start time  : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

set +e
python3 scripts/run_stage6.py --config "${CONFIG}"
EXIT_CODE=$?
set -e

echo "============================================================"
echo "Stage 6 finished : $(date)"
echo "Exit code        : ${EXIT_CODE}"
if [ -f "${CKPT_DIR}/system_finetuned.pt" ]; then
    echo "Finetuned ckpt   : ${CKPT_DIR}/system_finetuned.pt"
fi
if [ -f "${CKPT_DIR}/final.pt" ]; then
    echo "Final checkpoint : ${CKPT_DIR}/final.pt"
fi
if [ -f "${PROJECT_DIR}/outputs/final_report.md" ]; then
    echo "Final report     : ${PROJECT_DIR}/outputs/final_report.md"
fi
if [ -f "${PROJECT_DIR}/outputs/stage6_eval.json" ]; then
    echo "Eval results     : ${PROJECT_DIR}/outputs/stage6_eval.json"
fi
echo "============================================================"

exit "${EXIT_CODE}"
