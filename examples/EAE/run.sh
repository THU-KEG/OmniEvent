# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/ace.yaml
CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/eae/s2s/dt.yaml
