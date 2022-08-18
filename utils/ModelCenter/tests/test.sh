MASTER_ADDR=localhost
MASTER_PORT=12347
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_vit.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_bert_pkv.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_bert.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_roberta.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_t5.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_t5v1_1.py
python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_mt5.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_gpt2.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_gptj.py
# python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} test_glm.py
