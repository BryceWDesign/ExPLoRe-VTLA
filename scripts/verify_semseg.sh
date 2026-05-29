#!/bin/bash -l
#SBATCH -J explore-verify-seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-01:00:00
#SBATCH -A isaac-utk0256
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --output=logs/verify/semseg_%j.out
#SBATCH --error=logs/verify/semseg_%j.err

# ExPLoRe Semseg Verification - 1000 iters, 1 GPU
# Verifies: backbone loading, FPN features, UPerNet training loop
set -e

PRETRAINED=${1:?Usage: sbatch scripts/verify_semseg.sh <checkpoint_path>}
ADE20K_ROOT=${2:-/path/to/ADE20K}

module load cuda/12.1.1-binary
module load cudnn/8.9.4-binary

cd "${SLURM_SUBMIT_DIR}" || exit 1
source .venv/bin/activate

SEMSEG_DIR="${SLURM_SUBMIT_DIR}/src/downstream/segmentation"
export PYTHONPATH="${SEMSEG_DIR}:${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "================================================================"
echo "ExPLoRe Semseg Verification (1000 iters)"
echo "Checkpoint: ${PRETRAINED}"
echo "ADE20K: ${ADE20K_ROOT}"
echo "================================================================"

cd "${SEMSEG_DIR}" || exit 1

# Use training config but override to 1000 iters
python tools/train.py \
  "src/downstream/segmentation/configs/medic/upernet_medic_base_512_160k_ade20k.py" \
  --work-dir "/tmp/explore_verify_semseg_${SLURM_JOB_ID}" \
  --options \
    model.backbone.pretrained="${PRETRAINED}" \
    data.samples_per_gpu=2 \
    runner.max_iters=1000 \
    checkpoint_config.interval=5000 \
    data_root="${ADE20K_ROOT}" \
    data.train.data_root="${ADE20K_ROOT}" \
    data.val.data_root="${ADE20K_ROOT}" \
    data.test.data_root="${ADE20K_ROOT}" \
  --no-validate \
  2>&1

echo "================================================================"
echo "Semseg verification complete at: $(date)"
echo "================================================================"
