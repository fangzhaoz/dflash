#!/usr/bin/env bash
# prepare_data.sh — build the tiny gsm8k parquet files verl expects.
# Uses verl's own preprocessor so the schema matches main_ppo exactly.
# Run after setup_env.sh, in the same conda env.
set -eo pipefail

ENV_NAME=${ENV_NAME:-dflash-verl}
VERL_DIR=${VERL_DIR:-$HOME/verl}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_save_dir "$DATA_DIR"

echo "----------------------------------------------"
echo "gsm8k written to: $DATA_DIR"
ls -lh "$DATA_DIR"/*.parquet
echo "Next: bash rl_spec_decode/run0_smoke.sh"
