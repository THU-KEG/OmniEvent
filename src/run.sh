CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
    --nnodes=1 \
    --nproc_per_node=7 \
    EAE_main.py ./config/zh/eae/duee.yaml
