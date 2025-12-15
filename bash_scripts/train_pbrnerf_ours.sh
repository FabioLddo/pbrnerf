#!/usr/bin/env bash
# Set the default scene
SCENE="data_rand"
# TAGS="debug"
TAGS="avg"
CONS_LOSS=0.0
SPEC_LOSS=0.0

# Load environment variables from dotenv file
ENV_FILE="/app/pbrnerf/.env"
if [ -f $ENV_FILE ]; then
  set -a
  source $ENV_FILE
  set +a
  echo "WANDB_MODE: WANDB_MODE"  # Add this line

fi

# Override with the first command-line argument, if provided
if [ -n "$1" ]; then
  SCENE="$1"
fi

if [ -n "$2" ]; then
  TAGS="$2"
fi

if [ -n "$3" ]; then
  CONS_LOSS="$3"
fi

if [ -n "$4" ]; then
  SPEC_LOSS="$4"
fi

echo "Using scene: $SCENE"

cd code
python training/train.py \
  $WORKSPACE_DIR/datasets/$SCENE \
  $WORKSPACE_DIR/outputs/$SCENE \
  --name $SCENE \
  --tags $TAGS \
  --override_cons_weighting $CONS_LOSS \
  --override_spec_weighting $SPEC_LOSS \
  --config_path configs/config.json


# Example for NeILF++ "city"
# python evaluation/evaluate.py \
#   /workspace/datasets/neilfpp_synthetic/synthetic_city \
#   /workspace/outputs \
#   --config_path configs/config_synthetic_data_pbrnerf_neilfpp.json \
#   --phase joint \
#   --eval_brdf \
#   --export_brdf \
#   --export_nvs \
#   --export_mesh \
#   --export_lighting

# python evaluation/evaluate.py \
#   /workspace/datasets/neilfpp_synthetic/synthetic_city \
#   /workspace/outputs \
#   --config_path configs/config_synthetic_data_pbrnerf_neilfpp.json \
#   --phase joint --export_mesh 


