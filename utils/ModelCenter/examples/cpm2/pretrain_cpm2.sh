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

BASE_PATH="/mnt/sfs_turbo/ModelCenter"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${BASE_PATH}/configs/cpm2/cpm2-large"
OPTS+=" --batch-size 4"
OPTS+=" --train-iters 3000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name cpm2-checkpoint"
OPTS+=" --max-encoder-length 512"
OPTS+=" --max-decoder-length 256"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --lr 0.25"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 5e-3"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 110000"
# OPTS+=" --load ${BASE_PATH}/results/CPM2-0.25-0.005-checkpoint-110000.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/cpm2/pretrain_cpm2.py ${OPTS}"
echo ${CMD}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} | tee ${BASE_PATH}/logs/cpm2.log
else
    ${CMD}
fi
