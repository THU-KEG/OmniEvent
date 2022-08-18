#! /bin/bash
WORKING_DIR=/mnt/sfs_turbo/hx/cpm3-pretrain/transfer
cd ${WORKING_DIR}
echo "Current working directory ${WORKING_DIR}"
# python3 cpm1_oldffn2newffn.py
# python3 cpm2_oldffn2newffn.py
# python3 hugGPTj_bmtrainGPTj.py
CMD="python3 cpm1_old2new.py"
echo ${CMD}

${CMD} 2>&1 | tee /mnt/sfs_turbo/hx/cpm3-pretrain/logs/test-new.log