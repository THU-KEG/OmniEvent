# CUDA_VISIBLE_DEVICES=$1 python delta_tuning.py ../../config/ed/s2s/dt.yaml 
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/ed/s2s/ace.yaml 
# CUDA_VISIBLE_DEVICES=$1 python seq2seq.py ../../config/ed/s2s/maven.yaml 
CUDA_VISIBLE_DEVICES=$1 python sequence_labeling.py ../../config/ed/sl/lstm.yaml