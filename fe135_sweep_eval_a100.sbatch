#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH --time=02:30:00
#SBATCH -J fe135-sweepeval
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/fe135_sweep_eval_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
EXP=$PROJ/experiments/fe135_sweep_metrics_20260623
MASK_DIR=$PROJ/data/flowedit_compatible_135/eval_masks
export HF_HOME=$PROJ/.cache/huggingface; export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub; export HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1; export HF_HUB_OFFLINE=1
export LOCAL_PROMPTS_JSON=$PROJ/data/flowedit_compatible_135/local_target_prompts.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; date; cd "$PROJ"; mkdir -p "$EXP"
M='fluxsw_s070 fluxsw_s085 ours_flux fluxsw_s115 fluxsw_s130 sd3sw_s070 sd3sw_s085 ours_sd3 sd3sw_s115 sd3sw_s130'
python3 prepare_flowedit135_metric_runs.py --manifest data/flowedit_compatible_135/manifest_sweep_subset.json --methods "$M" --out-dir outputs/sweep_metric_runs --allow-missing
"$PROJ/.venv/bin/python" scripts/evaluate_paper_metrics.py --outputs-dir "$PROJ/outputs/sweep_metric_runs" --csv-output "$EXP/metrics.csv" --json-output "$EXP/metrics.json" --task-names all --method-names all --seeds 10 --clip-model openai/clip-vit-large-patch14 --dino-model facebook/dinov2-base --compute-lpips --bg-dilate-px 31 --eval-mask-dir "$MASK_DIR"
"$PROJ/.venv/bin/python" - <<'PY'
import csv,collections
r=list(csv.DictReader(open('experiments/fe135_sweep_metrics_20260623/metrics.csv')))
g=collections.defaultdict(list)
for x in r: g[x['method']].append(x)
def m(items,k):
    v=[float(x[k]) for x in items if x.get(k) not in (None,'')]
    return sum(v)/len(v) if v else 0
print('%-16s %5s %10s %9s'%('method','n','local_clipT','bg_lpips'))
for meth in ['sd3sw_s070','sd3sw_s085','ours_sd3','sd3sw_s115','sd3sw_s130','fluxsw_s070','fluxsw_s085','ours_flux','fluxsw_s115','fluxsw_s130']:
    if meth in g: print('%-16s %5d %10.4f %9.4f'%(meth,len(g[meth]),m(g[meth],'local_clip_t'),m(g[meth],'bg_lpips')))
PY
date; echo fe135_sweep_eval_complete
