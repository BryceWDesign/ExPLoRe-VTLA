#!/bin/bash
# ExPLoRe end-to-end verification driver
#
# Submits: pretrain smoke (2 epochs on subset), kNN eval, finetune (1 ep),
# linprobe (1 ep), semseg (1000 iter). No detection (out of paper scope).
#
# Usage:
#   bash scripts/verify_all.sh <pretrained_ckpt> [imagenet_path] [ade20k_path]
# Or set as env vars:
#   EXPLORE_CKPT=path/to/ckpt.pth bash scripts/verify_all.sh

set -e
cd "$(dirname "$0")/.."

CKPT="${1:-${EXPLORE_CKPT:?Usage: bash scripts/verify_all.sh <ckpt> [imagenet_path] [ade20k_path], or set EXPLORE_CKPT}}"
DATA="${2:-${IMAGENET_PATH:?set IMAGENET_PATH or pass as 2nd positional arg}}"
ADE20K="${3:-${ADE20K_PATH:?set ADE20K_PATH or pass as 3rd positional arg}}"

mkdir -p logs/verify

echo "================================================================"
echo "ExPLoRe end-to-end verification"
echo "  Checkpoint: ${CKPT}"
echo "  ImageNet:   ${DATA}"
echo "  ADE20K:     ${ADE20K}"
echo "================================================================"

# 1. Pretrain smoke (lightest sanity check first)
echo ""
echo "--- Pretrain smoke (2 epochs, subset=10) ---"
SMOKE=$(sbatch --parsable scripts/smoke_train_real.sh configs/pretrain_explore_smoke.yaml)
echo "  Job: $SMOKE"

# 2. kNN evaluation
echo ""
echo "--- kNN eval ---"
KNN=$(sbatch --parsable scripts/eval_knn.sh "${CKPT}")
echo "  Job: $KNN"

# 3. Finetune verification (1 epoch)
echo ""
echo "--- Finetune (1 epoch) ---"
FT=$(sbatch --parsable scripts/verify_finetune.sh "${CKPT}" "${DATA}")
echo "  Job: $FT"

# 4. Linear probe verification (1 epoch)
echo ""
echo "--- Linear probe (1 epoch) ---"
LP=$(sbatch --parsable scripts/verify_linprobe.sh "${CKPT}" "${DATA}")
echo "  Job: $LP"

# 5. Semseg verification (1000 iter)
echo ""
echo "--- Semseg (1000 iter) ---"
SEG=$(sbatch --parsable scripts/verify_semseg.sh "${CKPT}" "${ADE20K}")
echo "  Job: $SEG"

echo ""
echo "================================================================"
echo "All verification jobs submitted:"
echo "  smoke=${SMOKE}  knn=${KNN}  ft=${FT}  lp=${LP}  seg=${SEG}"
echo ""
echo "Tail logs with:"
echo "  tail -f logs/verify/finetune_${FT}.out"
echo "  tail -f logs/verify/linprobe_${LP}.out"
echo "  tail -f logs/verify/semseg_${SEG}.out"
echo "================================================================"
