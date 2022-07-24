# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/ace.yaml
CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/eae/s2s/ace-cn.yaml
# CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/eae/s2s/dt.yaml
# CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/eae/s2s/ace.yaml
# CUDA_VISIBLE_DEVICES=$1 python sequence_labeling.py ../../config/eae/sl/ace-en.yaml
# CUDA_VISIBLE_DEVICES=$1 python token_classification.py ../../config/eae/tc/ace-en.yaml 
