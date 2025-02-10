#!/bin/bash

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=2

# Default hyperparameters from the paper
EXP_NAME="diffusion_default"
BATCH_SIZE=32
LEARNING_RATE=1e-5
EPOCHS=5
SPATIAL_WIDTH=28
N_COLORS=1
N_TEMPORAL_BASIS=10
TRAJECTORY_LENGTH=1000
HIDDEN_CHANNELS=128
NUM_LAYERS=200
DEVICE="cuda"
VAL_INTERVAL=10
SAVE_DIR="checkpoints"
BETA_START=0.01
BETA_END=0.05
MIN_T=100

# Run training script with default arguments and any additional arguments passed
python ./scripts/train.py \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --spatial_width ${SPATIAL_WIDTH} \
    --n_colors ${N_COLORS} \
    --n_temporal_basis ${N_TEMPORAL_BASIS} \
    --trajectory_length ${TRAJECTORY_LENGTH} \
    --hidden_channels ${HIDDEN_CHANNELS} \
    --num_layers ${NUM_LAYERS} \
    --device ${DEVICE} \
    --val_interval ${VAL_INTERVAL} \
    --save_dir ${SAVE_DIR} \
    --beta_start ${BETA_START} \
    --beta_end ${BETA_END} \
    --min_t ${MIN_T} \
    "$@" 

# # 기본 설정으로 학습
# bash ./train.sh

# # 체크포인트에서 계속 학습
# bash ./train.sh --continue_train --checkpoint_path /home/yuhan/test/Deep-Unsupervised-Learning-using-Nonequilibrium-Thermodynamics/checkpoints/diffusion_default_epoch1_batch710.pth

# # 특정 파라미터 변경
# bash ./train.sh --batch_size 64 --learning_rate 1e-4