CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py config.yaml
