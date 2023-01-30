
DATA_DIR="../../NIKL_WRITTEN(v1.1)/"

TRAIN_INPUT="${DATA_DIR}/train_final_mecab.txt"
TRAIN_DATA="${DATA_DIR}/train_preprocessed_final_mecab.txt"
DEV_INPUT="${DATA_DIR}/dev_final_mecab.txt"
DEV_DATA="${DATA_DIR}/dev_preprocessed_final_mecab.txt"
TEST_INPUT="${DATA_DIR}/test_final_mecab.txt"
TEST_DATA="${DATA_DIR}/test_preprocessed_final_mecab.txt"


# Preprocessing
python preprocessing.py \
    --input ${TRAIN_INPUT} \
    --output ${TRAIN_DATA} \
    --save_noised_output

python preprocessing.py \
    --input ${DEV_INPUT} \
    --output ${DEV_DATA} \
    --save_noised_output

python preprocessing.py \
    --input ${TEST_INPUT} \
    --output ${TEST_DATA} \
    --save_noised_output
