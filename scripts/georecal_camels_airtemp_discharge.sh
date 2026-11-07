#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
experiment_dir=${1:-results/camels_airtemp_discharge}
if [[ -e "$experiment_dir" ]]; then
  echo "Output already exists: $experiment_dir. Pass a new directory as the first argument." >&2
  exit 1
fi

# Shared data and model settings.
common=(
  --experiment_dir "$experiment_dir"
  --data camels_crossvar
  --root_path ./dataset/camels_ch/
  --site_meta_path ./dataset/camels_ch/data_sources.csv
  --x_data_path air_temperature.csv
  --y_data_path discharge.csv
  --seq_len 512
  --patch_len 128
  --stride 64
  --d_model 256
  --e_layers 3
  --d_layers 3
  --domain_len 128
  --support_ratio 0.66
  --learning_rate 1e-3
  --num_workers 4
)

# Relation pretraining.
python run.py "${common[@]}" \
  --is_relation_pretraining \
  --is_training 0 \
  --train_epochs 500 \
  --relation_early_stop_epoch 15 \
  --batch_size 512 \
  --weight_decay 0.05 \
  --relation_pretrain_samples_per_site 128 \
  --relation_loss_site_weight 1.0 \
  --relation_loss_aug_weight 0.5 \
  --relation_loss_xy_weight 0.1 \
  --relation_loss_proto_weight 0.2 \
  --num_prototypes 5 \
  --prototype_warmup_epochs 5 \
  --prototype_cluster_interval 2 \
  --prototype_momentum 0.5 \
  --prototype_temperature 0.2

# Train with MAML, then evaluate the best checkpoint.
torchrun --nnodes=1 --nproc_per_node=1 --master_port=${MASTER_PORT:-29522} run.py "${common[@]}" \
  --is_training 1 \
  --window_stride 8 \
  --train_epochs 100 \
  --batch_size 6 \
  --patience 6 \
  --inner_steps 2 \
  --inner_lr 5e-3 \
  --meta_lr 5e-5 \
  --static_emb_dim 64
