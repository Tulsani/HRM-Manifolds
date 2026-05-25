#!/usr/bin/env bash
#SBATCH --job-name=flight-stage2-geometry
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
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/stage2.yaml}"
TRACE_DIR="${TRACE_DIR:-${PROJECT_DIR}/trace_library}"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
LOG_DIR="${PROJECT_DIR}/hpc_runs/logs"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Pre-flight checks
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Missing config: ${CONFIG}"
    exit 1
fi

if [ ! -d "${TRACE_DIR}" ]; then
    echo "ERROR: Missing trace library: ${TRACE_DIR}"
    echo "Run stage1_trace_gen.sh first."
    exit 1
fi

N_TRACE_FILES=$(find "${TRACE_DIR}" -type f | wc -l | tr -d ' ')
if [ "${N_TRACE_FILES}" -lt 1 ]; then
    echo "ERROR: Trace library is empty: ${TRACE_DIR}"
    echo "Run stage1_trace_gen.sh first."
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/checkpoints/backbone_stage0.pt" ]; then
    echo "WARNING: Missing ${PROJECT_DIR}/checkpoints/backbone_stage0.pt"
    echo "Stage 2 may fail unless the config points to an available backbone checkpoint."
fi

echo "============================================================"
echo "Stage       : 2 - skill refinement and geometry analysis"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "Project dir : ${PROJECT_DIR}"
echo "Config      : ${CONFIG}"
echo "Trace dir   : ${TRACE_DIR} (${N_TRACE_FILES} files)"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Conda env   : ${CONDA_ENV}"
echo "Start time  : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

set +e
python3 scripts/run_stage2.py --config "${CONFIG}"
EXIT_CODE=$?
set -e

echo "============================================================"
echo "Stage 2 finished : $(date)"
echo "Exit code        : ${EXIT_CODE}"
if [ -f "${PROJECT_DIR}/outputs/geometry_config.json" ]; then
    echo "Geometry config  : ${PROJECT_DIR}/outputs/geometry_config.json"
fi
if [ -f "${PROJECT_DIR}/outputs/skill_labels.json" ]; then
    echo "Skill labels     : ${PROJECT_DIR}/outputs/skill_labels.json"
fi
echo "============================================================"

exit "${EXIT_CODE}"
