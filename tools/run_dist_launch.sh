#!/usr/bin/env bash

# source /opt/anaconda3/etc/profile.d/conda.sh
# conda info --env
# conda activate cdformer

# 设置要使用的 GPU 为 GPU 4
export CUDA_VISIBLE_DEVICES=0

set -x

GPUS=$1
RUN_COMMAND=${@:2}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.10"}
# MASTER_PORT=${MASTER_PORT:-"29505"}
MASTER_PORT=${MASTER_PORT:-"29505"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python ./tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}
