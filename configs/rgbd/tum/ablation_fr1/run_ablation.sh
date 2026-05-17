#!/bin/bash
# Ablation study: Baseline vs VO-prior on fr1_desk
# Usage: cd VPAR-GS-SLAM && bash configs/rgbd/tum/ablation_fr1/run_ablation.sh
set -eu

PROJECT_ROOT="$(pwd)"
RESULT_DIR="$PROJECT_ROOT/results/Ablation/fr1_desk"
CONFIG_DIR="$PROJECT_ROOT/configs/rgbd/tum/ablation_fr1"

mkdir -p "$RESULT_DIR"

GPU_ID="${GPU_ID:-1}"

echo "=============================================="
echo "Ablation: fr1_desk"
echo "  GPU: $GPU_ID"
echo "  Results: $RESULT_DIR"
echo "=============================================="

echo ""
echo "=== [A] Baseline: VO=off, tracking_itr=100 ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/A_baseline.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/A_baseline.log"

echo ""
echo "=== [B] A+VO: VO=on ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/B_AplusVO.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/B_AplusVO.log"

echo ""
echo "=============================================="
echo "Ablation complete. Logs:"
echo "  $RESULT_DIR/A_baseline.log"
echo "  $RESULT_DIR/B_AplusVO.log"
echo "=============================================="
