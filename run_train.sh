NUM_GPUS=1

TIME=$(date "+%Y%m%d%H%M%S")
MODEL_NAME=DFRSC
EXP_ID=BR000

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=12356 train.py \
    -opt configs/DFRSC_3F_CarlsRS.yml \
    --launcher pytorch --auto_resume  # 2> ./logs/${EXP_ID}_${MODEL_NAME}_${TIME}.log
