#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=06:00:00
#SBATCH -J e2-4-medit
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/e2_support_matched_medit_gate_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/e2_support_matched_medit_gate_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
OUT="$PROJECT/outputs/e2_support_matched_medit_gate"
PY="$PROJECT/.venv/bin/python"
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"
SEEDS="10 11 12"
METHOD="direct_target_medit_gate"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run E2.4 same-M_edit gating outside a100-01" >&2
  exit 2
fi

export MODEL_OFFLOAD=0
export RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1
export ALLOW_MASK_DOWNLOAD=1

"$PY" scripts/prepare_e2_support_matched_medit_gate.py --tasks $TASKS --seeds $SEEDS
"$PY" scripts/run_sd3_batch.py --manifest "$OUT/command_manifest.txt" --skip-existing --summary-output "$OUT/sd3_batch_summary.json" --stop-on-failure
"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$OUT" \
  --csv-output "$EXP/e2_support_matched_medit_gate_metrics.csv" \
  --json-output "$EXP/e2_support_matched_medit_gate_metrics.json" \
  --task-names "$TASKS" \
  --method-names "$METHOD" \
  --seeds "$SEEDS" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/e4_t1_t4_reconstruction_floor_metrics.csv"
"$PY" - <<'PY'
import csv
from pathlib import Path
exp = Path("experiments/support_v3_2026-06-02")
rows = list(csv.DictReader((exp / "e2_support_matched_medit_gate_metrics.csv").open(newline="", encoding="utf-8")))
rows = [r for r in rows if r.get("method") == "direct_target_medit_gate"]
def avg(key):
    vals=[]
    for r in rows:
        try:
            if r.get(key,"") != "": vals.append(float(r[key]))
        except ValueError:
            pass
    return "" if not vals else f"{sum(vals)/len(vals):.4f}"
summary = [{
    "method": "direct_target_medit_gate",
    "n": str(len(rows)),
    "outside_mask_l1_mean": avg("outside_mask_l1"),
    "inside_mask_l1_mean": avg("inside_mask_l1"),
    "source_ssim_luma_mean": avg("source_ssim_luma"),
    "edit_score_mean": avg("edit_score"),
}]
out = exp / "e2_support_matched_medit_gate_summary.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
print(f"wrote {out}")
PY
