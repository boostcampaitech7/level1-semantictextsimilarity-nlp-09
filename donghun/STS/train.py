import argparse
import yaml
import os
import shutil

import torch
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger  # Import TensorBoard Logger

from src import data_pipeline
from src.model import Model

from pytorch_lightning.callbacks import EarlyStopping
import random

if __name__ == "__main__":
    
    # if there is version_0 directory, remove it
    # if os.path.exists('./STS/lightning_logs/version_0'):
    #     shutil.rmtree('./STS/lightning_logs/version_0')
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_path', default='../data/train.csv')
    # parser.add_argument('--train_path', default='./data/train2.csv')
    # parser.add_argument('--train_path', default='./data/SR_train_3.csv')
    # parser.add_argument('--train_path', default='./data/sr5.csv') 
    # parser.add_argument('--train_path', default='./data/train_even.csv')
    parser.add_argument('--train_path', default='./data/train_even_updated_ko2en.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args()

    with open('./STS/baselines/baseline_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # seed 고정
    torch.manual_seed(CFG['seed'])
    torch.cuda.manual_seed(CFG['seed'])
    torch.cuda.manual_seed_all(CFG['seed'])
    random.seed(CFG['seed'])

    dataloader = data_pipeline.Dataloader(CFG, args.train_path, args.dev_path, args.test_path, args.predict_path)
    model = Model(CFG)

    # Train the model with gradient clipping and TensorBoard logging
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=CFG['train']['epoch'], 
        log_every_n_steps=1, 
        gradient_clip_val=1.0,
        # logger=logger  # Add the TensorBoard logger here
    )

    # train
    trainer.fit(model=model, datamodule=dataloader)
    test_result = trainer.test(model=model, datamodule=dataloader)
    
    # save the model based on test result
    pearson = round(test_result[0]['test_pearson'], 7)
    model_name = CFG['train']['model_name']
    model_name = model_name.replace('/', '-')
    new_log_dir = f"{pearson}_{model_name}"
    os.rename('./STS/lightning_logs/version_0', f'./STS/lightning_logs/{new_log_dir}')
    model_path = f"./model.pt"
    torch.save(model, model_path)
    
    src_file = 'baselines/baseline_config.yaml'
    dst_file = os.path.join(f'./STS/lightning_logs/{new_log_dir}', 'baseline_config.yaml')
    shutil.copy(src_file, dst_file)
    
    print(f"Model saved at {model_path}")
    
    if pearson < 0.7:
        # delete the directory if model is not good enough
        print('\n'*2)
        print(f"Model is not good enough. Deleting the directory {new_log_dir}")
        shutil.rmtree(f'./STS/lightning_logs/{new_log_dir}')
        # make an empty folder named pearson
        os.makedirs(f'./STS/lightning_logs/{new_log_dir}', exist_ok=True)
        src_file = 'baselines/baseline_config.yaml'
        dst_file = os.path.join(f'./STS/lightning_logs/{new_log_dir}', 'baseline_config.yaml')
        shutil.copy(src_file, dst_file)
