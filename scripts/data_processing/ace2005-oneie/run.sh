# Install requirements
pip install -r requirements.txt
# First, you need to download nltk model 
# >>> import nltk
# >>> nltk.download("punkt")
# Or you can download manually here: https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.zip
# to your ~/nltk_data/tokenizers/. And then unzip it.
ACE_DATA_FOLDER=./../../../data/original/ace_2005_td_v7
python process_ace.py -i ${ACE_DATA_FOLDER}/data -o data/ -s splits/ACE05-E -b bert-base-cased -l english

python convert_to_openee.py \
    --save_dir ../../../data/processed/ace2005-oneie