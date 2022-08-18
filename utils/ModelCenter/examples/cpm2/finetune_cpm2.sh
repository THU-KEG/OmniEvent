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

BASE_PATH="/home/hx/ModelCenter"
DATASET="LCQMC"

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config cpm2-large"
OPTS+=" --batch-size 64"
OPTS+=" --train-iters 3000"
OPTS+=" --save-iters 1000"
OPTS+=" --max-encoder-length 256"
OPTS+=" --max-decoder-length 2"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name finetune-cpm2-ckpt"
OPTS+=" --lr 0.002"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 200"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 1048576"
# OPTS+=" --load ${BASE_PATH}/results/CPM2-0.25-0.005-checkpoint-110000.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/cpm2/finetune_cpm2.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/logs/cpm2/${DATASET}.log
