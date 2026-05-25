#!/usr/bin/env bash
#SBATCH --job-name=flight-stage5-distill
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
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/stage5.yaml}"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/hpc_runs/logs"
ROUTER_CKPT="${CKPT_DIR}/router_stage4.pt"
GEOMETRY_CONFIG="${PROJECT_DIR}/outputs/geometry_config.json"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${PROJECT_DIR}/outputs"

# Pre-flight checks
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Missing config: ${CONFIG}"
    exit 1
fi

if [ ! -f "${ROUTER_CKPT}" ]; then
    echo "ERROR: Missing router checkpoint: ${ROUTER_CKPT}"
    echo "Run stage4_router.sh first."
    exit 1
fi

if [ ! -f "${GEOMETRY_CONFIG}" ]; then
    echo "ERROR: Missing geometry config: ${GEOMETRY_CONFIG}"
    echo "Run stage2_geometry.sh first."
    exit 1
fi

N_EXPERTS=$(find "${CKPT_DIR}" -name "expert_*_stage3.pt" | wc -l | tr -d ' ')
if [ "${N_EXPERTS}" -lt 1 ]; then
    echo "ERROR: No expert checkpoints found in ${CKPT_DIR}."
    echo "Run stage3_experts.sh first."
    exit 1
fi

if [ -f "${CKPT_DIR}/system_best.pt" ]; then
    echo "Existing best system checkpoint found: ${CKPT_DIR}/system_best.pt"
fi

echo "============================================================"
echo "Stage       : 5 - full-system distillation"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "Project dir : ${PROJECT_DIR}"
echo "Config      : ${CONFIG}"
echo "Router ckpt : ${ROUTER_CKPT}"
echo "Expert ckpts: ${N_EXPERTS}"
echo "Checkpoints : ${CKPT_DIR}"
echo "Conda env   : ${CONDA_ENV}"
echo "Start time  : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

set +e
python3 scripts/run_stage5.py --config "${CONFIG}"
EXIT_CODE=$?
set -e

echo "============================================================"
echo "Stage 5 finished : $(date)"
echo "Exit code        : ${EXIT_CODE}"
if [ -f "${CKPT_DIR}/system_best.pt" ]; then
    echo "Best checkpoint  : ${CKPT_DIR}/system_best.pt"
fi
if [ -f "${CKPT_DIR}/full_system_stage5.pt" ]; then
    echo "Final checkpoint : ${CKPT_DIR}/full_system_stage5.pt"
fi
if [ -f "${PROJECT_DIR}/outputs/stage5_eval.json" ]; then
    echo "Eval results     : ${PROJECT_DIR}/outputs/stage5_eval.json"
fi
echo "============================================================"

exit "${EXIT_CODE}"
