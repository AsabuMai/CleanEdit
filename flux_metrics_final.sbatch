#!/bin/bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -J flux-met-final
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/flux_metrics_final_%j.log
set -euo pipefail
cd /cluster/users/grad/2025/25t8103/project
hostname; date
mkdir -p experiments/flux_dece_final_20260618 logs
export HF_HOME=/cluster/users/grad/2025/25t8103/project/.cache/huggingface
export TRANSFORMERS_CACHE=/cluster/users/grad/2025/25t8103/project/.cache/huggingface/hub
export HF_HUB_OFFLINE=1

TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2 pillow_same_color_cable_knit pillow_same_color_cable_knit_grey pillow_same_color_cable_knit_armchair"

.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir outputs/phase2_flux_paper_final_20260618 \
  --task-names "$TASKS" \
  --method-names dece_rf_flux \
  --seeds "10 11 12" \
  --clip-model openai/clip-vit-large-patch14 --clip-device cuda \
  --dino-model facebook/dinov2-base --dino-device cuda \
  --compute-lpips --lpips-device cuda \
  --csv-output experiments/flux_dece_final_20260618/dece_rf_flux_metrics.csv \
  --json-output experiments/flux_dece_final_20260618/dece_rf_flux_metrics.json
date; echo flux_metrics_final_complete
