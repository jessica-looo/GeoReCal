#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4
experiment_dir=${1:-results/pwl_ms}
if [[ -e "$experiment_dir" ]]; then
  echo "Output already exists: $experiment_dir. Pass a new directory as the first argument." >&2
  exit 1
fi

# Shared data and model settings.
common=(
  --experiment_dir "$experiment_dir"
  --root_path ./dataset/sediment/
  --site_meta_path dataset/sediment/data_sources.csv
  --data pwl_ms_crossvar
  --seq_len 32
  --patch_len 16
  --stride 8
  --e_layers 3
  --d_layers 3
  --domain_len 128
  --d_model 256
  --num_workers 0
  --static_emb_dim 64
  --support_ratio 0.66
  --x_data_path PWL.csv
  --y_data_path MS.csv
  --itr 1
)

# Relation pretraining.
python run.py "${common[@]}" \
  --is_relation_pretraining \
  --is_training 0 \
  --model_id sediment_32_relation_pretrain \
  --train_epochs 300 \
  --relation_early_stop_epoch 15 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --weight_decay 0.05 \
  --relation_pretrain_samples_per_site 24 \
  --relation_loss_site_weight 1.0 \
  --relation_loss_aug_weight 0.5 \
  --relation_loss_xy_weight 0.1 \
  --relation_loss_proto_weight 0.2 \
  --relation_aug_noise_std 0.05 \
  --relation_aug_mask_ratio 0.1 \
  --num_prototypes 3 \
  --prototype_warmup_epochs 5 \
  --prototype_cluster_interval 2 \
  --prototype_momentum 0.5 \
  --prototype_temperature 0.2

# Train and evaluate each fold.
maml=(
  --is_training 1
  --model_id sediment_32
  --train_epochs 100
  --batch_size 6
  --learning_rate 1e-3
  --patience 6
  --cluster_result_path cluster/pwl/pwl_cluster.csv
  --inner_steps 2
  --inner_lr 1e-3
  --meta_lr 5e-5
  --maml_loss mae
  --selection_metric mae
)
for fold in 0 1 2 3 4; do
  torchrun --nnodes=1 --nproc_per_node=1 --master_port=${MASTER_PORT:-29503} run.py "${common[@]}" "${maml[@]}" --fold_id "$fold" --skip_test
  torchrun --nnodes=1 --nproc_per_node=1 --master_port=${MASTER_PORT:-29503} run.py "${common[@]}" "${maml[@]}" --fold_id "$fold" \
    --is_training 0 --eval_split test --inner_steps 2
done

# Pool predictions across all five test folds.
python summarize_fivefold.py "$experiment_dir/maml_baseline"
