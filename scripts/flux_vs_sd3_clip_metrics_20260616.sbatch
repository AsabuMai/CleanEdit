#!/bin/bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -J fluxsd3met
#SBATCH -o logs/flux_vs_sd3_clip_metrics_%j.out
#SBATCH -e logs/flux_vs_sd3_clip_metrics_%j.err
set -euo pipefail
cd /cluster/users/grad/2025/25t8103/project
hostname
mkdir -p experiments/flux_vs_sd3_20260616 logs
export HF_HOME=/cluster/users/grad/2025/25t8103/project/.cache/huggingface
export TRANSFORMERS_CACHE=/cluster/users/grad/2025/25t8103/project/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
COMMON_TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2 pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir outputs/flux_vs_sd3_eval_tree_20260616 \
  --task-names "$COMMON_TASKS" \
  --method-names dece_rf_flux \
  --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 \
  --clip-device cuda \
  --dino-model facebook/dinov2-base \
  --dino-device cuda \
  --csv-output experiments/flux_vs_sd3_20260616/flux_clip_dino_seed10.csv \
  --json-output experiments/flux_vs_sd3_20260616/flux_clip_dino_seed10.json
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir outputs/pretty_matrix \
  --task-names "$COMMON_TASKS" \
  --method-names support_v3_controller_rmsgap \
  --seeds 10 \
  --clip-model openai/clip-vit-large-patch14 \
  --clip-device cuda \
  --dino-model facebook/dinov2-base \
  --dino-device cuda \
  --csv-output experiments/flux_vs_sd3_20260616/sd3_clip_dino_seed10.csv \
  --json-output experiments/flux_vs_sd3_20260616/sd3_clip_dino_seed10.json
