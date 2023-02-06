GPU_ID=$1
TAG=$2
LENGTH=64
BATCH_SIZE=1  # 32... batch size per each GPU
LR="6.25e-5"
ACCUM=1 # 4
N_EPOCHS=8
SAVE_STEPS=300  # Validating and save checkpoints
RANDOM_SEED=1223734

TRAIN_DATA="../NIKL_WRITTEN(v1.1)/train_preprocessed_final_mecab_shuffle_0.3_replace_0.2.txt"
DEV_DATA="../NIKL_WRITTEN(v1.1)/dev_preprocessed_final_mecab_shuffle_0.3_replace_0.2.txt"

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_trf3.py \
    --train_data_path ${TRAIN_DATA}\
    --dev_data_path ${DEV_DATA} \
    --max_length ${LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${ACCUM} \
    --num_epochs ${N_EPOCHS} \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LR} \
    --tag ${TAG} \
    --seed ${RANDOM_SEED} --debug --toy
