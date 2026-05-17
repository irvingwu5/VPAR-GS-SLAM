#!/bin/bash
# Ablation study: Baseline vs VO-prior on fr3_office
# Usage: cd VPAR-GS-SLAM && bash configs/rgbd/tum/ablation_fr3/run_ablation.sh
set -eu

PROJECT_ROOT="$(pwd)"
RESULT_DIR="$PROJECT_ROOT/results/Ablation/fr3_office"
CONFIG_DIR="$PROJECT_ROOT/configs/rgbd/tum/ablation_fr3"

mkdir -p "$RESULT_DIR"

GPU_ID="${GPU_ID:-1}"

echo "=============================================="
echo "Ablation: fr3_office"
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
