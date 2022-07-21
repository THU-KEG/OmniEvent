if [ ! -d "stanford-corenlp-4.4.0" ]; then
    wget https://cloud.tsinghua.edu.cn/f/e2e3a37b30b14c869988/?dl=1 --content-disposition 
    unzip stanford-corenlp-4.4.0.zip 
    rm stanford-corenlp-4.4.0.zip 
fi

python ace2005.py \
    --data_dir ../../../data/ace_2005_td_v7 \
    --ACE_SPLITS splits \
    --ACE_DUMP ../../../data/processed/ace2005 \
    --corenlp_path stanford-corenlp-4.4.0 