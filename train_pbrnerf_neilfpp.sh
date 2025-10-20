#!/usr/bin/env bash
# Set the default scene
SCENE="city"
TAGS="debug"
CONS_LOSS=0.0
SPEC_LOSS=0.0

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
  /workspace/datasets/neilfpp_synthetic/synthetic_$SCENE \
  /workspace/outputs/neilfpp_synthetic/synthetic_$SCENE \
  --name pbrnerf_neilfpp_$SCENE \
  --tags $TAGS \
  --override_cons_weighting $CONS_LOSS \
  --override_spec_weighting $SPEC_LOSS \
  --config_path configs/config_synthetic_data_pbrnerf_neilfpp.json


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


