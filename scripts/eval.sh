#!/usr/bin/env bash

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate cdformer
export CUDA_VISIBLE_DEVICES=0
fewshot_seed=01
num_shot=10

python -u main.py \
--dataset_file uodd \
--backbone dinov2 \
--num_feature_levels 1 \
--enc_layers 6 \
--dec_layers 6 \
--hidden_dim 256 \
--num_queries 300 \
--batch_size 2 \
--resume base_train.pth \
--fewshot_finetune \
--fewshot_seed ${fewshot_seed} \
--num_shots ${num_shot} \
--eval \
2>&1 | tee ./log_inference_base_0.txt
```