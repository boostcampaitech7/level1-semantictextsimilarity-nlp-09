#!/bin/bash

# 함수로 각 하이퍼파라미터를 설정하도록 구성
set_hyperparameters() {
    sed -i "s/seed: .*/seed: $1/" baselines/baseline_config.yaml
    sed -i "s/model_name: .*/model_name: $2/" baselines/baseline_config.yaml
    sed -i "s/batch_size: .*/batch_size: $3/" baselines/baseline_config.yaml
    sed -i "s/epoch: .*/epoch: $4/" baselines/baseline_config.yaml
    sed -i "s/LR: .*/LR: $5/" baselines/baseline_config.yaml
    sed -i "s/LossF: .*/LossF: $6/" baselines/baseline_config.yaml
    sed -i "s/optim: .*/optim: $7/" baselines/baseline_config.yaml
    sed -i "s/shuffle: .*/shuffle: $8/" baselines/baseline_config.yaml
}

# 예시: klue/roberta-base 모델, batch_size=16, epoch=5, LR=3e-5, L1Loss, AdamW 옵티마이저, shuffle=True

# set_hyperparameters 42 "upskyy\/kf-deberta-multitask" 16 1 0.00003 "L1Loss" "AdamW" "True"
# python3 train.py || true

# set_hyperparameters 42 "intfloat\/multilingual-e5-large-instruct" 16 5 0.00003 "L1Loss" "AdamW" "True"
# python3 train.py || true

# set_hyperparameters 42 "intfloat\/multilingual-e5-large" 16 5 0.00003 "L1Loss" "AdamW" "True"
# python3 train.py || true

# set_hyperparameters 42 "jhgan\/ko-sroberta-multitask" 16 5 0.00003 "L1Loss" "AdamW" "True"
# python3 train.py || true

# set_hyperparameters 42 "upskyy\/bge-m3-korean" 16 5 0.00003 "L1Loss" "AdamW" "True"
# python3 train.py || true