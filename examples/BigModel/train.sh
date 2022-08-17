#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=1234
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=$PWD
VERSION="large"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config configs/mt5-${VERSION}/"
OPTS+=" --epochs 10"
OPTS+=" --do_train"
OPTS+=" --do_test"
OPTS+=" --batch-size 16"
OPTS+=" --train-iters 1500"
OPTS+=" --save-iters 1000"
OPTS+=" --save ${BASE_PATH}/results/finetune-mt5-${VERSION}"
OPTS+=" --save-name pytorch_model.pt"
OPTS+=" --lr 0.00001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr-decay-style linear"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 100.0"
OPTS+=" --loss-scale 128"
OPTS+=" --train_file ../../data/processed/all-eae/train.unified.jsonl"
OPTS+=" --validation_file ../../data/processed/all-eae/dev.unified.jsonl"
OPTS+=" --test_file ../../data/processed/all-eae/test.unified.jsonl"
OPTS+=" --language English"
OPTS+=" --golden_trigger"
OPTS+=" --max_seq_length 160"
OPTS+=" --max_out_length 128" 
OPTS+=" --truncate_in_batch"
OPTS+=" --truncate_seq2seq_output"
# OPTS+=" --load ${BASE_PATH}/results/t5-v1_1-${VERSION}.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/logs/t5-v1_1_superglue/finetune-t5-v1_1-${VERSION}-${DATASET}.log