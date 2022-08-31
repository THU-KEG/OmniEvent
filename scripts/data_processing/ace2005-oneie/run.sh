ACE_DATA_FOLDER=./../../../data/original/ace_2005_td_v7
python process_ace.py -i ${ACE_DATA_FOLDER}/data -o data/ -s splits/ACE05-E -b bert-base-cased -l english

python convert_to_openee.py \
    --save_dir ../../../data/processed/ace2005-oneie