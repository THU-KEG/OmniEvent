# CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/ed/s2s/dt.yaml 
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/all-datasets/ed/s2s/ace-dygie.yaml 
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/ed/s2s/ace-zh.yaml 
# deepspeed --include localhost:0,1,2,3,4,5,6,7 seq2seq.py \
#         ../../config/ed/s2s/dt.yaml \
#         --deepspeed ../../config/deepspeed_zero_2.json 
CUDA_VISIBLE_DEVICES=$1 python token_classification.py \
    ../../config/all-models/ed/tc/cnn/dm.yaml
# CUDA_VISIBLE_DEVICES=$1 python sequence_labeling.py \
#     ../../config/all-models/ed/sl/lstm/wo-crf.yaml
