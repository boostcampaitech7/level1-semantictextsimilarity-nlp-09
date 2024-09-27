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



if __name__ == '__main__':
    
    # if there is version_0 directory, remove it
    if os.path.exists('lightning_logs/version_0'):
        shutil.rmtree('lightning_logs/version_0')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../donghyuk/data/train_even.csv')
    parser.add_argument('--dev_path', default='../donghyuk/data/dev.csv')
    parser.add_argument('--test_path', default='../donghyuk/data/dev.csv')
    # parser.add_argument('--predict_path', default='../../data/test.csv')
    parser.add_argument('--predict_path', default='../donghyuk/data/valid_dropped_label.csv')
    parser.add_argument('--target_folder', default='0.978852_beomi-KcELECTRA-base')
    args = parser.parse_args()

    baseline_config = f'./lightning_logs/{args.target_folder}/baseline_config.yaml'
    with open(baseline_config) as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # admin, seed, model, batch_size, epoch, LR, lossf, optim, shuffle
    print('\n'*2)
    print('*'*50)
    print(f'Admin: {CFG["admin"]}')
    print(f'Seed: {CFG["seed"]}')
    print(f'Model: {CFG["train"]["model_name"]}')
    print(f'Batch Size: {CFG["train"]["batch_size"]}')
    print(f'Epoch: {CFG["train"]["epoch"]}')
    print(f'Learning Rate: {CFG["train"]["LR"]}')
    print(f'Loss Function: {CFG["train"]["LossF"]}')
    print(f'Optimizer: {CFG["train"]["optim"]}')
    print(f'Shuffle: {CFG["train"]["shuffle"]}')
    print('*'*50)

    dataloader = data_pipeline.Dataloader(CFG, args.train_path, args.dev_path, args.test_path, args.predict_path)
    model = torch.load(f'./lightning_logs/{args.target_folder}/model.pt')

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=CFG['train']['epoch'], log_every_n_steps=1)

    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output = pd.read_csv('../../data/sample_submission.csv')
    output = pd.read_csv('../donghyuk/data/sample_submission_550.csv')
    output['target'] = predictions
    # output.to_csv(f'./lightning_logs/{args.target_folder}/output_dev550.csv', index=False)
    output.to_csv(f'./lightning_logs/{args.target_folder}/output_{args.target_folder}.csv', index=False)
    