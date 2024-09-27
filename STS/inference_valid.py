import argparse
import yaml
import pandas as pd
import os
from tqdm.auto import tqdm

import torch

import pytorch_lightning as pl
#import wandb

from src import data_pipeline
import shutil
import numpy as np


def find_closest_label(predicted_value):
    if predicted_value < 0:
        value = 0.0
    
    elif predicted_value == -0.0:
        value = 0.0
        
    elif predicted_value > 5:
        value = 5.0
        
    elif (predicted_value*10) % 2 != 0:
        labels = np.array([0. , 0.2, 0.4, 0.5, 0.6, 0.8, 1. , 1.2, 1.4, 1.5, 1.6, 1.8, 2. ,
            2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.4, 3.5, 3.6, 3.8, 4. , 4.2, 4.4,
            4.5, 4.6, 4.8, 5. ])
        counts = np.array([21,  7, 16,  2, 20, 22, 22, 27, 17,  4, 18, 22, 22, 23, 21, 22, 22,
            22, 22, 22,  3, 19, 22, 22, 25, 19,  7, 15, 22, 22])
        
        distances = np.abs(labels - predicted_value)
        weighted_distances = distances / counts
        
        # 가장 가까운 라벨 선택
        value = labels[np.argmin(weighted_distances)]
    
    else:
        value = predicted_value
        
    return float(value)

def process_list(input_list):
    return [find_closest_label(x) for x in input_list]



if __name__ == '__main__':
    
    # if there is version_0 directory, remove it
    if os.path.exists('lightning_logs/version_0'):
        shutil.rmtree('lightning_logs/version_0')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../data/dev_preprop_v2.csv')
    parser.add_argument('--dev_path', default='../data/dev_preprop_v2.csv')
    parser.add_argument('--test_path', default='../data/dev_preprop_v2.csv')
    # parser.add_argument('--predict_path', default='../data/test_preprop_v2.csv')
    parser.add_argument('--predict_path', default='../data/dev_preprop_v2_no_label.csv')
    parser.add_argument('--target_folder', default='models')
    parser.add_argument('--model_name', default='model.pt')
    args = parser.parse_args()

    hparams = 'baselines/baseline_config_large_final_v1.yaml'
    with open(hparams) as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    dataloader = data_pipeline.Dataloader(CFG, args.train_path, args.dev_path, args.test_path, args.predict_path)
    model = torch.load(f'{args.target_folder}/{args.model_name}')
    
    # upload model with .ckpt
    # model = pl.load_from_checkpoint(f'./lightning_logs/{args.target_folder}/model.ckpt')

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=CFG['train']['epoch'], log_every_n_steps=1)

    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    predictions = process_list(predictions)
    

    output = pd.read_csv('../data/sample_submission_550.csv')
    output['target'] = predictions
    # output.to_csv(f'./lightning_logs/{args.target_folder}/output.csv', index=False)
    output.to_csv(f'{args.target_folder}/output_valid_2-{args.model_name}.csv', index=False)
