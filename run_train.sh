#!/bin/bash

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 필요한 디렉토리 생성
mkdir -p result
mkdir -p data

# CUDA 디바이스 설정 (옵션)
export CUDA_VISIBLE_DEVICES=3

# 학습 실행
python scripts/train.py --resume /home/yuhan/test/TBH/result/cifar10/model/20250130/model-epoch4-batch200.pth

# # 처음부터 학습
# python scripts/train.py

# # 체크포인트에서 이어서 학습
#python scripts/train.py --resume /home/yuhan/test/TBH/result/cifar10/model/20250130/model-epoch4-batch200.pth