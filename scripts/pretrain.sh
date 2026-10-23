#!/usr/bin/env bash
# Pretrain one ResNet-18 encoder with the settings used for every run in
# results/pretraining/ (one GPU, 400 epochs, 4,096-key queue).
#
#   bash scripts/pretrain.sh caco
#   bash scripts/pretrain.sh caco --resume_from_checkpoint checkpoints/<run>/caco-epoch=0299.ckpt
#
# Writes checkpoints/<run>/resnet18_<mode>_geo_10k_400.pth (plain encoder state_dict)
# and logs/<run>/metrics.csv (per-epoch loss, subspace accuracy, LR, timing).
set -euo pipefail

MODE=${1:?usage: pretrain.sh <moco|seco|teco|caco> [extra train.py args]}
shift
DATA_DIR=${DATA_DIR:-data/clean_10k_geography}

python src/pretrain/train.py \
  --data_dir "$DATA_DIR" \
  --data_mode "$MODE" \
  --base_encoder resnet18 \
  --batch_size 256 \
  --num_negatives 4096 \
  --emb_dim 128 \
  --encoder_momentum 0.999 \
  --softmax_temperature 0.07 \
  --learning_rate 0.03 \
  --max_epochs 400 \
  --schedule 250 350 \
  --num_workers 2 \
  --gpus 1 \
  --save_dir checkpoints \
  --log_dir logs \
  -d "${MODE}_10k" \
  "$@"
