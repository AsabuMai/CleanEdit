#!/usr/bin/env bash
#SBATCH -p h100
#SBATCH -w h100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -J pie-h100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie_run_h100_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
: "${MANIFEST:?set MANIFEST}"; : "${OUTROOT:?set OUTROOT}"
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DISABLE_BASELINE_NSFW=1
B=$PROJ/_baselines; E=$B/envs; hostname; date; echo "MANIFEST=$MANIFEST OUTROOT=$OUTROOT"
echo "### DeCE-FLUX ###"; MANIFEST=$MANIFEST FLUX_OUT=$OUTROOT $PROJ/.venv-h100/bin/python $PROJ/scripts/pie_batch_flux.py || echo deceflux_ERR; date
echo "### FireFlow ###"; (cd $B/src/FireFlow/src && MANIFEST=$MANIFEST OUT=$OUTROOT $E/fireflow-py310/bin/python fireflow_batch.py) || echo fireflow_ERR; date
echo "### RF-Solver ###"; (cd $B/src/RF-Solver-Edit/FLUX_Image_Edit/src && MANIFEST=$MANIFEST OUT=$OUTROOT $E/rf-solver-edit-py310/bin/python rfsolver_batch.py) || echo rfsolver_ERR; date
echo "### ReFlex ###"; MANIFEST=$MANIFEST OUT=$OUTROOT $E/reflex-py310/bin/python $PROJ/scripts/reflex_batch.py || echo reflex_ERR; date
echo "### Sam-Flow-flux ###"; MODE=flux MANIFEST=$MANIFEST OUT=$OUTROOT $E/sam-flow-py310/bin/python $PROJ/scripts/sam_flow_batch.py || echo samflowflux_ERR; date
echo pie_run_h100_complete
