#!/bin/bash

# 1. hyperparameter 수정: model_name을 klue/roberta-base로 변경
sed -i 's/model_name: .*/model_name: klue\/roberta-base/' baselines/baseline_config.yaml
python3 train.py || true

sed -i 's/model_name: .*/model_name: intfloat\/multilingual-e5-large-instruct/' baselines/baseline_config.yaml
python3 train.py || true

sed -i 's/model_name: .*/model_name: intfloat\/multilingual-e5-large/' baselines/baseline_config.yaml
python3 train.py || true

sed -i 's/model_name: .*/model_name: jhgan\/ko-sroberta-multitask/' baselines/baseline_config.yaml
python3 train.py || true

sed -i 's/model_name: .*/model_name: upskyy\/bge-m3-korean/' baselines/baseline_config.yaml
python3 train.py || true
