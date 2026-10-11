#!/usr/bin/env bash

# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate cdformer

EXP_DIR=exps/artaxor
BASE_TRAIN_DIR=${EXP_DIR}/epoch160
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

fewshot_seed=01
num_shot=10
epoch=160
save_every_epoch=5
lr_drop1=80
lr_drop2=130
lr=5e-5
lr_backbone=5e-6
mixstyle_p=0.8
mixstyle_alpha=0.3
clip_max_norm=0.1
dropout=0.1
FS_FT_DIR=${BASE_TRAIN_DIR}/seed${fewshot_seed}_${num_shot}shot_01_copy_p${mixstyle_p}_a${mixstyle_alpha}_clip${clip_max_norm}_dropout${dropout}_random_sam_augmentation_2
mkdir ${FS_FT_DIR}

python -u main.py \
    --dataset_file ArTaxOr \
    --backbone dinov2 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 2 \
    --lr ${lr} \
    --lr_backbone ${lr_backbone} \
    --resume model_pt/base_train.pth \
    --fewshot_finetune \
    --fewshot_seed ${fewshot_seed} \
    --num_shots ${num_shot} \
    --epoch ${epoch} \
    --lr_drop_milestones ${lr_drop1} ${lr_drop2} \
    --warmup_epochs 50 \
    --save_every_epoch ${save_every_epoch} \
    --eval_every_epoch 5 \
    --output_dir ${FS_FT_DIR} \
    --category_codes_cls_loss \
    --clip_max_norm ${clip_max_norm} \
    --dropout ${dropout} \
    --use_mixstyle \
    --mixstyle_p ${mixstyle_p} \
    --mixstyle_alpha ${mixstyle_alpha} \
2>&1 | tee ${FS_FT_DIR}/log.txt
