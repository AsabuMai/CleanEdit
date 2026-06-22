#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J t2a-common-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/table2a_common_metrics_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/table2a_common_metrics_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run Table 2a metric completion outside a100-01" >&2
  exit 2
fi

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/table2a_e4_common_subset_clipdino_metrics.csv" \
  --json-output "$EXP/table2a_e4_common_subset_clipdino_metrics.json" \
  --task-names "$TASKS" \
  --method-names "support_v3_fixed support_v3_controller_rmsgap" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/e4_t1_t4_reconstruction_floor_metrics.csv" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --allow-download

"$PY" scripts/build_table2a_common_subset.py
