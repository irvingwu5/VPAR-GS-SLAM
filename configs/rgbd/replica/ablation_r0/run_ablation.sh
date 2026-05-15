#!/bin/bash
# Ablation study: Baseline vs +Normal vs +Normal+Dist on replica room0
# Usage: cd VPAR-GS-SLAM && bash configs/rgbd/replica/ablation_r0/run_ablation.sh
set -eu

PROJECT_ROOT="$(pwd)"
RESULT_DIR="$PROJECT_ROOT/results/Ablation/room0"
CONFIG_DIR="$PROJECT_ROOT/configs/rgbd/replica/ablation_r0"

mkdir -p "$RESULT_DIR"

GPU_ID="${GPU_ID:-1}"

echo "=============================================="
echo "Ablation: Replica room0 -- Normal & Dist Loss"
echo "  GPU: $GPU_ID"
echo "  Results: $RESULT_DIR"
echo "=============================================="

echo ""
# echo "=== [A] Baseline: no normal/dist loss ==="
# CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
#   --config "$CONFIG_DIR/A_baseline.yaml" --eval \
#   2>&1 | tee "$RESULT_DIR/A_baseline.log"

echo ""
echo "=== [B] +Normal: lambda_normal=0.05 ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/B_normal_only.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/B_normal_only.log"

echo ""
echo "=== [C] +Normal+Dist: lambda_normal=0.05 lambda_dist=1.0 ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/C_normal_dist.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/C_normal_dist.log"

echo ""
echo "=== [D] +Normal+Dist+VO: lambda_normal=0.05 lambda_dist=1.0 VO=on ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/D_normal_dist_vo.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/D_normal_dist_vo.log"

echo ""
echo "=============================================="
echo "Ablation complete. Logs:"
echo "  $RESULT_DIR/A_baseline.log"
echo "  $RESULT_DIR/B_normal_only.log"
echo "  $RESULT_DIR/C_normal_dist.log"
echo "  $RESULT_DIR/D_normal_dist_vo.log"
echo "=============================================="
