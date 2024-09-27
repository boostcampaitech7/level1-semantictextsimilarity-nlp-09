#!/bin/bash

# 모델 이름을 저장한 리스트
models=("snunlp/KR-SBERT-V40K-klueNLI-augSTS" "snunlp/KR-ELECTRA-discriminator" )

# 설정 파일 경로
config_file="/data/ephemeral/home/donghyuk/STS/baselines/baseline_config.yaml"

# 모델 이름 리스트에서 하나씩 순차적으로 선택하여 config 파일 수정 및 train.py 실행
for model in "${models[@]}"
do
    echo "Training with model: $model"
    
    # YAML 파일에서 model_name 수정
    sed -i "s|model_name: .*|model_name: $model|" "$config_file"
    
    # train.py 실행
    python /data/ephemeral/home/donghyuk/STS/train.py --config "$config_file"
    
    # 학습이 성공적으로 완료되었는지 확인 (옵션)
    if [ $? -eq 0 ]; then
        echo "Training for $model completed successfully."
    else
        echo "Training for $model failed. Stopping script."
        exit 1
    fi
    
    echo "----------------------------------------"
done

echo "All models have been trained."
