#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/mnt/sfs_turbo/hx/ModelCenter"
VERSION="6b"
DATASET="CB"

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config gptj-${VERSION}"
OPTS+=" --batch-size 8"
OPTS+=" --train-iters 400"
OPTS+=" --save-iters 1000"
OPTS+=" --max-decoder-length 512"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name finetune-gptj-ckpt"
OPTS+=" --lr 0.00001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 40"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 128"
# OPTS+=" --load ${BASE_PATH}/results/GPTj-${VERSION}.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gptj/finetune_gptj.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/logs/gptj_superglue/finetune-gptj-${VERSION}-${DATASET}.log
