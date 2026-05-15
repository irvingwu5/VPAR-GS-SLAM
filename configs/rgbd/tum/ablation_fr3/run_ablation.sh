#!/bin/sh
# Ablation study: Baseline vs VO-prior on fr3_office
# Usage: bash configs/rgbd/tum/ablation_fr3/run_ablation.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RESULT_DIR="$PROJECT_ROOT/results/Ablation/fr3_office"

mkdir -p "$RESULT_DIR"

GPU_ID="${GPU_ID:-1}"

echo "=============================================="
echo "Ablation: fr3_office"
echo "  GPU: $GPU_ID"
echo "  Results: $RESULT_DIR"
echo "=============================================="

echo ""
echo "=== [A] Baseline: VO=off, tracking_itr=100 ==="
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$SCRIPT_DIR/A_baseline.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/A_baseline.log"

echo ""
echo "=== [B] A+VO: VO=on ==="
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$SCRIPT_DIR/B_AplusVO.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/B_AplusVO.log"

echo ""
echo "=============================================="
echo "Ablation complete. Logs:"
echo "  $RESULT_DIR/A_baseline.log"
echo "  $RESULT_DIR/B_AplusVO.log"
echo "=============================================="
