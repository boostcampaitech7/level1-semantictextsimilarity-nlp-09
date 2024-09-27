#!/bin/bash

# python3 inference.py --target_folder 0.8613058_klue-roberta-base-attention_masked_seed_0 || true

# python3 inference.py --target_folder 0.8613058_klue-roberta-base-attention_masked || true

# python3 inference.py --target_folder 0.8813921_klue-roberta-base || true


# TARGET_FOLDER="0.8964779_upskyy-kf-deberta-multitask"

# python3 inference.py --target_folder "$TARGET_FOLDER" || true
# python3 inference_valid.py --target_folder "$TARGET_FOLDER" || true
# python3 EDA/vis_scatter_plot.py --target_folder "$TARGET_FOLDER" || true


set_target_folders() {
    python3 inference.py --target_folder $1 || true
    python3 inference_valid.py --target_folder $1 || true
    python3 EDA/vis_scatter_plot.py --target_folder $1 || true
}

set_target_folders "0.9273497_klue-roberta-large"

# set_target_folders "0.8707963_klue-roberta-base"

# set_target_folders "0.8057685_klue-roberta-base"
# set_target_folders "0.8901988_klue-roberta-large"
# set_target_folders "0.8150103_klue-roberta-base"
# set_target_folders "0.8945142_klue-roberta-large"
# set_target_folders ""