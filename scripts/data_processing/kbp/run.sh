# Install requirements
pip install -r requirements.txt
# First, you need to download nltk model
# >>> import nltk
# >>> nltk.download("punkt")
# Or you can download manually here: https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.zip
# to your ~/nltk_data/tokenizers/. And then unzip it.

python kbp2014-ed.py \
    --data_dir ../../../data/original/tac_kbp_eng_event_nugget_detect_coref_2014-2015 \
    --save_dir ../../../data/processed/TAC-KBP2014

python kbp2015-ed.py \
    --data_dir ../../../data/original/tac_kbp_eng_event_nugget_detect_coref_2014-2015 \
    --save_dir ../../../data/processed/TAC-KBP2015

python kbp2016.py \
    --data_dir ../../../data/original/tac_kbp_event_arg_comp_train_eval_2016-2017 \
    --source_dir ../../../data/original/tac_kbp_eval_src_2016-2017 \
    --save_dir ../../../data/processed/TAC-KBP2016

python kbp2017.py \
    --data_dir ../../../data/original/tac_kbp_event_arg_comp_train_eval_2016-2017 \
    --source_dir ../../../data/original/tac_kbp_eval_src_2016-2017 \
    --save_dir ../../../data/processed/TAC-KBP2017
