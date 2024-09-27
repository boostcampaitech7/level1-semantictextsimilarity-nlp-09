import argparse
import yaml
import pandas as pd
import os
from tqdm.auto import tqdm

import torch

import pytorch_lightning as pl
#import wandb

from src import data_pipeline



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../../data/train.csv')
    parser.add_argument('--dev_path', default='../../data/dev.csv')
    parser.add_argument('--test_path', default='../../data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    parser.add_argument('--target_folder')
    args = parser.parse_args()

    baseline_config = './STS/baselines/baseline_config.yaml'
    with open(baseline_config) as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    dataloader = data_pipeline.Dataloader(CFG, args.train_path, args.dev_path, args.test_path, args.predict_path)
    # model = torch.load(f'./lightning_logs/version_2/checkpoints/epoch=9-step=3850.ckpt')
    # model = pl.load_from_checkpoint(f'./lightning_logs/{args.target_folder}/model.ckpt')
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=CFG['train']['epoch'], log_every_n_steps=1)

    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    output = pd.read_csv('../../data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('./output.csv', index=False)