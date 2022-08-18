# CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/ed/s2s/dt.yaml 
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/ed/s2s/ace.yaml 
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/ed/s2s/ace-zh.yaml 
deepspeed --include localhost:0,1,2,3,4,5,6,7 seq2seq.py \
        ../../config/ed/s2s/dt.yaml \
        --deepspeed ../../config/deepspeed_zero_2.json 