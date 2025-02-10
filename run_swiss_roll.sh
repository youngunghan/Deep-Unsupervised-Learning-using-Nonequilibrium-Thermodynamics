#!/bin/bash

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=2

# Default hyperparameters
EXP_NAME="swiss_roll_diffusion"
NUM_TIMESTEPS=40
HIDDEN_DIM=256
DATA_DIM=2
DEVICE="cuda"
BETA_MIN=1e-5
BETA_MAX=3e-1
BETA_SCHEDULE_MIN=-18
BETA_SCHEDULE_MAX=10
LEARNING_RATE=2e-4
BATCH_SIZE=128000
NUM_EPOCHS=300000
EVAL_INTERVAL=3000

# Directory setup
SAVE_DIR="test/checkpoints/${EXP_NAME}"
LOG_DIR="test/logs/${EXP_NAME}"

# Run training
python scripts/test_swiss_roll.py \
    --exp_name ${EXP_NAME} \
    --num_timesteps ${NUM_TIMESTEPS} \
    --hidden_dim ${HIDDEN_DIM} \
    --data_dim ${DATA_DIM} \
    --device ${DEVICE} \
    --beta_min ${BETA_MIN} \
    --beta_max ${BETA_MAX} \
    --beta_schedule_min ${BETA_SCHEDULE_MIN} \
    --beta_schedule_max ${BETA_SCHEDULE_MAX} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --eval_interval ${EVAL_INTERVAL}