# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/ace-zh.yaml
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/ace.yaml
# CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/eae/s2s/dt.yaml
# CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/eae/s2s/ace.yaml
# CUDA_VISIBLE_DEVICES=$1 python sequence_labeling.py ../../config/eae/sl/ace-en.yaml
# CUDA_VISIBLE_DEVICES=$1 python token_classification.py ../../config/eae/tc/ace-en.yaml 
# CUDA_VISIBLE_DEVICES=2,3 deepspeed --num_gpus=2 delta_tuning.py \
#         ../../config/eae/s2s/dt.yaml \
#         --deepspeed ../../config/deepspeed.json 
# deepspeed --include localhost:2,3 seq2seq.py \
#         ../../config/eae/s2s/ace.yaml \
#         --deepspeed ../../config/deepspeed.json 
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/fewfc.yaml
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/duee.yaml
# CUDA_VISIBLE_DEVICES=$1 python token_classification.py ../../config/eae/tc/ace-en.yaml 
deepspeed --include localhost:4,5,6,7 delta_tuning.py \
        ../../config/eae/s2s/dt.yaml \
        --deepspeed ../../config/deepspeed.json 
# CUDA_VISIBLE_DEVICES=4,5,6,7 python delta_tuning.py \
#         ../../config/eae/s2s/dt.yaml 