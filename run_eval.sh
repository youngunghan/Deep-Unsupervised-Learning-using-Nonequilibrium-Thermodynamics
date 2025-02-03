#!/bin/bash

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA 디바이스 설정 (옵션)
export CUDA_VISIBLE_DEVICES=0

# 평가 실행
python scripts/eval.py \
    --model_path "/home/yuhan/test/TBH/result/cifar10/model/20250130/model-epoch10-batch200.pth" \
    --dataset "cifar10" \
    --batch_size 32 