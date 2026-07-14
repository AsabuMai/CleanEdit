#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=02:30:00
#SBATCH -J fe135-pgeval
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_sweep_eval_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/fe135_flux_pgsweep_metrics_20260623
MASK_DIR=$PROJ/data/flowedit_compatible_135/eval_masks
export HF_HOME=$PROJ/.cache/huggingface; export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub; export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1; export HF_HUB_OFFLINE=1
export LOCAL_PROMPTS_JSON=$PROJ/data/flowedit_compatible_135/local_target_prompts.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date; cd "$PROJ"; mkdir -p "$EXP"
M='fluxpg_s030 fluxpg_s060 ours_flux fluxpg_s160 fluxpg_s250'
python3 prepare_flowedit135_metric_runs.py --manifest data/flowedit_compatible_135/manifest_sweep_subset.json --methods "$M" --out-dir outputs/sweep_metric_runs --allow-missing
"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py --outputs-dir "$PROJ/outputs/sweep_metric_runs" --csv-output "$EXP/metrics.csv" --json-output "$EXP/metrics.json" --task-names all --method-names all --seeds 10 --clip-model openai/clip-vit-large-patch14 --dino-model facebook/dinov2-base --compute-lpips --bg-dilate-px 31 --eval-mask-dir "$MASK_DIR"
"$PROJ/.venv/bin/python" - <<'PY'
import csv,collections
r=list(csv.DictReader(open('experiments/fe135_flux_pgsweep_metrics_20260623/metrics.csv')))
g=collections.defaultdict(list)
for x in r: g[x['method']].append(x)
def m(items,k):
    v=[float(x[k]) for x in items if x.get(k) not in (None,'')]
    return sum(v)/len(v) if v else 0
print('%-16s %5s %10s %9s'%('method','n','local_clipT','bg_lpips'))
for meth in ['fluxpg_s030','fluxpg_s060','ours_flux','fluxpg_s160','fluxpg_s250']:
    if meth in g: print('%-16s %5d %10.4f %9.4f'%(meth,len(g[meth]),m(g[meth],'local_clip_t'),m(g[meth],'bg_lpips')))
PY
date; echo fe135_sweep_eval_complete
