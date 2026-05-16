#!/bin/bash
# Ablation study: 6-step FGS-SLAM mechanisms on TUM fr1_desk
# A_baseline -> B_Aplus_normal -> C_Bplus_dist -> D_Cplus_fftmask -> E_Dplus_errormask -> F_Eplus_dpvo
# Usage: cd VPAR-GS-SLAM && bash configs/rgbd/tum/ablation_fr1/run_ablation.sh
set -eu

PROJECT_ROOT="$(pwd)"
RESULT_DIR="$PROJECT_ROOT/results/Ablation/fr1_desk"
CONFIG_DIR="$PROJECT_ROOT/configs/rgbd/tum/ablation_fr1"

mkdir -p "$RESULT_DIR"

GPU_ID="${GPU_ID:-1}"
# With CUDA_VISIBLE_DEVICES=1,0: cuda:0=GPU1 (main), cuda:1=GPU0 (DPVO via dpvo_device_id=1)

 echo ""
 echo "=== [A] Baseline ==="
 CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
   --config "$CONFIG_DIR/A_baseline.yaml" --eval \
   2>&1 | tee "$RESULT_DIR/A_baseline.log"
# CUDA_VISIBLE_DEVICES=1 python slam.py --config "configs/rgbd/tum/ablation_fr1/A_baseline.yaml" --eval 2>&1 | tee "./results/Ablation/fr1_desk/A_baseline_sa_depth.log"
echo ""
echo "=== [B] A + Normal: lambda_normal=0.001 ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/B_normal_only.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/B_normal_only.log"
# CUDA_VISIBLE_DEVICES=1 python slam.py --config "configs/rgbd/tum/ablation_fr1/B_normal_only.yaml" --eval 2>&1 | tee "./results/Ablation/fr1_desk/B_normal_only.log"
echo ""
echo "=== [C] B + Dist: lambda_normal=0.05 lambda_dist=1.0 ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/C_normal_dist.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/C_normal_dist.log"

echo ""
echo "=== [D] C + FFT Mask: frequency-based adaptive density/scale ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/D_Cplus_fftmask.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/D_Cplus_fftmask.log"

echo ""
echo "=== [E] D + Error Mask: error-based densification (replaces gradient) ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" python slam.py \
  --config "$CONFIG_DIR/E_Dplus_errormask.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/E_Dplus_errormask.log"
# CUDA_VISIBLE_DEVICES=1 python slam.py --config "configs/rgbd/tum/ablation_fr1/E_Dplus_errormask.yaml" --eval 2>&1 | tee "./results/Ablation/fr1_desk/E_Dplus_errormask.log"
echo ""
echo "=== [F] E + DPVO: full FGS-SLAM mechanisms + VO prior ==="
CUDA_VISIBLE_DEVICES=1,0 python slam.py \
  --config "$CONFIG_DIR/F_Eplus_dpvo.yaml" --eval \
  2>&1 | tee "$RESULT_DIR/F_Eplus_dpvo.log"
# CUDA_VISIBLE_DEVICES=1,0 python slam.py --config "configs/rgbd/tum/ablation_fr1/F_Eplus_dpvo.yaml" --eval 2>&1 | tee "./results/Ablation/fr1_desk/F_Eplus_dpvo.log"
echo ""
echo "=============================================="
echo "Ablation complete. Logs:"
echo "  $RESULT_DIR/A_baseline.log"
echo "  $RESULT_DIR/B_normal_only.log"
echo "  $RESULT_DIR/C_normal_dist.log"
echo "  $RESULT_DIR/D_Cplus_fftmask.log"
echo "  $RESULT_DIR/E_Dplus_errormask.log"
echo "  $RESULT_DIR/F_Eplus_dpvo.log"
echo "=============================================="
