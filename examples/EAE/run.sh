# CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed prompt_tuning.py ../../config/eae/s2s/pt.yaml \
#     --deepspeed ../../config/deepspeed.json 

CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/ace.yaml
