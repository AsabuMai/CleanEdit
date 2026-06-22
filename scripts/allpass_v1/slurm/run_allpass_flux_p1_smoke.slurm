#!/bin/bash
#SBATCH -J allpass_flux_p1
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:8
#SBATCH -t 06:00:00
#SBATCH -o logs/allpass_flux_p1_%j.out
#SBATCH -e logs/allpass_flux_p1_%j.err

set -euo pipefail
cd /cluster/users/grad/2025/25t8103/project
test "$(hostname)" = "h100-01.gpu01.cis.k.hosei.ac.jp" || { hostname; exit 1; }
source .venv/bin/activate
python scripts/allpass_v1/run_allpass_batch.py \
  --backend flux \
  --keys fe_046_cat_crown_1_black_top_hat,fe_180_milk_4_whipped_cream,fe_198_piece_of_cake_1_cherry_on_top,fe_067_cat_stone_1_black_kitten,fe_037_castle_1_lego_castle,fe_142_iguana_2_blue_lizard_top_hat \
  --overwrite
