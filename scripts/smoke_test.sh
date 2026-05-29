#!/bin/bash
#SBATCH --job-name=medic-smoke
#SBATCH -A isaac-utk0256
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --export=ALL,OMP_NUM_THREADS=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# MEDiC GPU Smoke Test
# Tests all 4 config variants with real CLIP teacher, bf16, synthetic data
# Usage: sbatch scripts/smoke_test.sh

set -e

module load cuda/12.1.1-binary
module load cudnn/8.9.4-binary

cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "================================================================"
echo "MEDiC GPU Smoke Test"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "SLURM Job: ${SLURM_JOB_ID:-none}"
echo "================================================================"

source .venv/bin/activate
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=16

python tests/test_gpu_smoke.py

echo ""
echo "================================================================"
echo "GPU Smoke test completed successfully"
echo "================================================================"
