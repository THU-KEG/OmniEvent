python LDC2015E29.py \
    --data_dir ../../../data/original/LDC2015E29 \
    --save_dir ../../../data/processed/LDC2015E29

python LDC2015E68.py \
    --data_dir ../../../data/original/LDC2015E68 \
    --save_dir ../../../data/processed/LDC2015E68

python LDC2015E78.py \
    --data_dir ../../../data/original/LDC2015E78 \
    --save_dir ../../../data/processed/LDC2015E78

python split.py \
    --ldc2015e29 ../../../data/processed/LDC2015E29/LDC2015E29.unified.jsonl \
    --ldc2015e68 ../../../data/processed/LDC2015E68/LDC2015E68.unified.jsonl \
    --ldc2015e78 ../../../data/processed/LDC2015E78/LDC2015E78.unified.jsonl \
    --split_dir ./split \
    --save_dir ../../../data/processed/ERE
