#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -J pie-a100
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/pie_run_a100_%j.log
set -uo pipefail
PROJ=/cluster/users/grad/2025/25t8103/project
: "${MANIFEST:?set MANIFEST}"; : "${OUTROOT:?set OUTROOT}"
export HF_HUB_CACHE=$PROJ/.cache/huggingface/hub HUGGINGFACE_HUB_CACHE=$PROJ/.cache/huggingface/hub
export HF_HOME=$PROJ/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 DISABLE_BASELINE_NSFW=1
B=$PROJ/_baselines; E=$B/envs; hostname; date; echo "MANIFEST=$MANIFEST OUTROOT=$OUTROOT"
echo "### DeCE-SD3 ###"; MANIFEST=$MANIFEST OUT=$OUTROOT $PROJ/.venv/bin/python $PROJ/scripts/pie_sd3_batch.py || echo decesd3_ERR; date
echo "### gen config yamls ###"; MANIFEST=$MANIFEST $PROJ/.venv/bin/python $PROJ/scripts/pie_flowedit_gen.py; MANIFEST=$MANIFEST $PROJ/.venv/bin/python $PROJ/scripts/pie_splitflow_gen.py; MANIFEST=$MANIFEST $PROJ/.venv/bin/python $PROJ/scripts/pie_rf2_gen.py
echo "### FlowEdit ###"; (cd $B/src/FlowEdit && $E/flowedit-py310/bin/python run_script.py --exp_yaml pie15_exp.yaml) || echo flowedit_ERR; date
echo "### SplitFlow ###"; (cd $B/src/SplitFlow && $E/splitflow-py310/bin/python run_script.py --exp_yaml pie15_exp.yaml) || echo splitflow_ERR; date
echo "### OT-RF ###"; (cd $B/src/OT-RF && $E/ot-rf-otip-py310/bin/python run_script.py --exp_yaml pie15_exp.yaml) || echo otrf_ERR; date
echo "### DRFS ###"; (cd $B/src/DeltaRectifiedFlowSampling && $E/dvrf-py310/bin/python edit.py --exp_yaml pie15_exp.yaml) || echo drfs_ERR; date
echo "### Sam-Flow-sd3 ###"; MODE=sd3 MANIFEST=$MANIFEST OUT=$OUTROOT $E/sam-flow-py310/bin/python $PROJ/scripts/sam_flow_batch.py || echo samflowsd3_ERR; date
echo "### collect config-driven -> OUTROOT ###"; OUTROOT=$OUTROOT $PROJ/.venv/bin/python $PROJ/scripts/pie_collect_configdriven.py
echo pie_run_a100_complete
