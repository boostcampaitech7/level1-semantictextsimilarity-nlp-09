import argparse
import yaml
import os
import shutil

import torch

import pytorch_lightning as pl
#import wandb

from src import data_pipeline
from src.model import Model

from pytorch_lightning.callbacks import EarlyStopping
import random

if __name__ == "__main__":
    
    # if there is version_0 directory, remove it
    if os.path.exists('lightning_logs/version_0'):
        shutil.rmtree('lightning_logs/version_0')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default="../data/concat2.csv")
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args()

    with open('baselines/baseline_config.yaml') as f:
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
    
    
    # seed 고정
    torch.manual_seed(CFG['seed'])
    torch.cuda.manual_seed(CFG['seed'])
    torch.cuda.manual_seed_all(CFG['seed'])
    random.seed(CFG['seed'])

    dataloader = data_pipeline.Dataloader(CFG, args.train_path, args.dev_path, args.test_path, args.predict_path)
    model = Model(CFG)

    ''' early stopping callback
    early_stopping_callback = EarlyStopping(
    monitor='val_loss',          # 검증 손실을 모니터링
    patience=3,                  # 성능 향상이 없을 경우 몇 에폭 후에 조기 종료할지 설정
    verbose=True,                # 조기 종료시 로그 출력 여부
    mode='min',                  # 손실이 최소화되어야 함
    )
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=CFG['train']['epoch'], log_every_n_steps=1, callbacks=[early_stopping_callback])
    '''
    
    # train
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=CFG['train']['epoch'], log_every_n_steps=1, gradient_clip_val=1.0)

    trainer.fit(model=model, datamodule=dataloader)
    test_result = trainer.test(model=model, datamodule=dataloader)
    
    # save
    pearson = round(test_result[0]['test_pearson'], 7)
    model_name = CFG['train']['model_name']
    model_name = model_name.replace('/', '-')
    new_log_dir = f"{pearson}_{model_name}"
    os.rename('lightning_logs/version_0', f'lightning_logs/{new_log_dir}')
    model_path = f"lightning_logs/{new_log_dir}/model.pt"
    torch.save(model, model_path)
    
    src_file = 'baselines/baseline_config.yaml'
    dst_file = os.path.join(f'lightning_logs/{new_log_dir}', 'baseline_config.yaml')
    shutil.copy(src_file, dst_file)
    
    print(f"Model saved at {model_path}")