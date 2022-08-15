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

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${BASE_PATH}/configs/cpm1/cpm1-large"
OPTS+=" --batch-size 64"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 1000"
OPTS+=" --save-name noam-1e-3-0.1-checkpoint"
OPTS+=" --max-length 512"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --lr 0.1"
OPTS+=" --inspect-iters 1000"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.001"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"
# OPTS+=" --load ${BASE_PATH}/results/noam-1e-3-0.05-checkpoint-1000.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/cpm1/pretrain_cpm1.py ${OPTS}"
echo ${CMD}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee ${BASE_PATH}/logs/cpm1-new.log
else
    ${CMD}
fi
