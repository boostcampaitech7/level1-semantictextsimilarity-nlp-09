import argparse
import yaml
import os
import shutil

import torch

import pytorch_lightning as pl
import wandb

from src import data_pipeline

from src.model import Model
from src.model_lstm import ModelLSTM
from src.model_gru import ModelGRU

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import random
from pytorch_lightning.loggers import WandbLogger


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_model_sweep(config=None):
    with wandb.init(config=config):  # Wandb sweep에서 설정 전달
        config = wandb.config  # sweep에서 전달된 설정
        wandb_logger = WandbLogger(project="large_final_sweep")  # 여기에 WandB 프로젝트 이름을 지정


        # if there is version_0 directory, remove it
        if os.path.exists('lightning_logs/version_0'):
            shutil.rmtree('lightning_logs/version_0')
        
        # args
        parser = argparse.ArgumentParser()
        parser.add_argument('--dev_path', default='../data/dev_preprop_v2.csv')
        parser.add_argument('--test_path', default='../data/dev_preprop_v2.csv')
        parser.add_argument('--predict_path', default='../data/test_preprop_v2.csv')
        # parser.add_argument('--dev_path', default='../data/dev.csv')
        # parser.add_argument('--test_path', default='../data/dev.csv')
        # parser.add_argument('--predict_path', default='../data/test.csv')
        args = parser.parse_args()

        # open CFG
        with open('baselines/baseline_config.yaml') as f:
            CFG = yaml.load(f, Loader=yaml.FullLoader)
        
        # CFG에 sweep 값으로 덮어쓰기
        CFG['admin'] = config.admin
        CFG['seed'] = config.seed
        CFG['train_data'] = config.train_data
        CFG['train']['model_name'] = config.model_name
        CFG['train']['model_custom'] = config.model_custom
        CFG['train']['LR'] = config.learning_rate
        CFG['train']['batch_size'] = config.batch_size
        # CFG['train']['dropout'] = config.dropout
        CFG['train']['freeze_layers'] = config.freeze_layers
        CFG['train']['epoch'] = config.epochs
        CFG['train']['LossF'] = config.loss_function
        CFG['train']['optim'] = config.optimizer

        # print: admin, seed, model, batch_size, epoch, LR, lossf, optim, shuffle
        print('\n'*2)
        print('*'*50)
        print(f'Admin: {CFG["admin"]}')
        print(f'Seed: {CFG["seed"]}')
        print(f'Model: {CFG["train"]["model_name"]}')
        print(f'Custom: {CFG["train"]["model_custom"]}')
        print(f'Batch Size: {CFG["train"]["batch_size"]}')
        print(f'Epoch: {CFG["train"]["epoch"]}')
        print(f'Learning Rate: {CFG["train"]["LR"]}')
        # print(f'Dropout: {CFG["train"]["dropout"]}')
        print(f'Freeze Layers: {CFG["train"]["freeze_layers"]}')
        print(f'Loss Function: {CFG["train"]["LossF"]}')
        print(f'Optimizer: {CFG["train"]["optim"]}')
        print(f'Train Data: {CFG["train_data"]}')
        print('*'*50)
        
        # seed
        torch.manual_seed(CFG['seed'])
        torch.cuda.manual_seed(CFG['seed'])
        torch.cuda.manual_seed_all(CFG['seed'])
        random.seed(CFG['seed'])        

        # data pipeline, model, trainer
        if config.train_data == 'train.csv':
            print('\n'*2)
            print('*'*50)
            print(f"Using original train.csv")
            print('*'*50)
            print('\n'*2)
            dataloader = data_pipeline.Dataloader(CFG, f'../data/{config.train_data}', '../data/dev.csv', '../data/dev.csv', '../data/test.csv')
        else:
            print('\n'*2)
            print('*'*50)
            print(f"Using {config.train_data}")
            print('*'*50)
            print('\n'*2)
            dataloader = data_pipeline.Dataloader(CFG, f'../data/{config.train_data}', args.dev_path, args.test_path, args.predict_path)

        if config.model_custom == 'basic':
            model = Model(CFG)
        elif config.model_custom == 'lstm':
            model = ModelLSTM(CFG)
        elif config.model_custom == 'gru':
            model = ModelGRU(CFG)

        early_stopping = EarlyStopping(
            monitor='val_pearson',     # val_pearson을 모니터링
            patience=5,                # 3 epoch 동안 개선이 없으면 멈춤
            mode='max',                # val_pearson을 최대화하는 방향으로
            verbose=True               # 로그 출력
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_pearson',   # 모니터링할 메트릭
            mode='max',              # pearson을 최대화하는 방향으로 저장
            save_top_k=1,            # 가장 좋은 모델 하나만 저장
            filename='{epoch}-{val_pearson:.4f}',  # 저장 파일명 포맷
            verbose=True
        )
        
        # from scratch
        # trainer = pl.Trainer(
        #     accelerator="gpu",
        #     logger=wandb_logger,  # WandB Logger 추가
        #     devices=1, 
        #     max_epochs=CFG['train']['epoch'], 
        #     log_every_n_steps=1, 
        #     gradient_clip_val=1.0,
        #     callbacks=[early_stopping, checkpoint_callback]  # 콜백에 ModelCheckpoint 추가
        # )
        # trainer.fit(model=model, datamodule=dataloader)

        # from checkpoint
        checkpoint_path = "/data/ephemeral/home/heejun/STS/large_final_sweep/bhw94ili/checkpoints/epoch=4-val_pearson=0.9354.ckpt"  # 체크포인트 경로 설정        
        trainer = pl.Trainer(
            accelerator="gpu",
            logger=wandb_logger,  # WandB Logger 추가
            devices=1, 
            max_epochs=4,
            log_every_n_steps=1, 
            gradient_clip_val=1.0,
            callbacks=[early_stopping, checkpoint_callback],  # 콜백에 ModelCheckpoint 추가
        )
        trainer.fit(model=model, datamodule=dataloader, ckpt_path=checkpoint_path)


        
        # train
        test_result = trainer.test(model=model, datamodule=dataloader)
        
        # 성능 로그 기록
        wandb.log({"test_pearson": test_result[0]['test_pearson']})

        # save model
        pearson = round(test_result[0]['test_pearson'], 7)
        model_name = CFG['train']['model_name']
        model_name = model_name.replace('/', '-')
        model_path = f"models/{pearson}_{model_name}.pt"
        
        if pearson > 0.9:
            torch.save(model, model_path)
            print(f"Model saved at {model_path}")
        else:
            print(f"Model is not good enough. Pearson: {pearson}")            
        # src_file = 'baselines/baseline_config.yaml'
        # dst_file = os.path.join(new_log_dir, 'baseline_config.yaml')
        # shutil.copy(src_file, dst_file)
        
        # delete the directory
        
        # if pearson < 0.9:
        #     print('\n'*2)
        #     print(f"Model is not good enough. Deleting '.ckpt' file {new_log_dir}")
        #     shutil.rmtree(f'{new_log_dir}/checkpoints')
        #     with open(f'{new_log_dir}/dummy_file.txt', 'w') as f:
        #         f.write(".ckpt file deleted because the model is not good enough.")
        #     # make an empty folder named pearson
        #     os.makedirs(new_log_dir, exist_ok=True)
        #     src_file = 'baselines/baseline_config.yaml'
        #     dst_file = os.path.join(new_log_dir, 'baseline_config.yaml')
        #     shutil.copy(src_file, dst_file)


if __name__ == "__main__":
    # wandb sweep
    
    sweep_config = {
        'method': 'grid',  # 'grid', 'random', 'bayes'
        'metric': {
            'name': 'test_pearson',  # 최적화할 목표 메트릭 (예: 검증 손실)
            'goal': 'maximize'  # 'minimize'는 손실 최소화, 'maximize'는 정확도 최대화
        },
        
        'parameters': {

            'learning_rate': {'values': [1e-5]},
            'freeze_layers': {'values': [6]},

            'model_name': {'values': ['klue/roberta-large']},
            'epochs': {'values': [30]}, # 얘만 바꾸는건 어차피 그래프 겹쳐서 맨 마지막에 해도 되는구나
            'batch_size': {'values': [32]},    # base: 16이 제일 좋더라

            # fixed
            'admin': {'value': 'heejun'},
            'seed': {'value': 42},
            'model_custom': {'value': 'basic'}, # custom 안한게 제일 좋더라
            'optimizer': {'value': 'AdamW'},
            'loss_function': {'values': ['L1Loss']}, # 주관적인 유사도 측정이므로 L1Loss가 더 적합할 것 같다.
            'train_data': {'values': ['train_preprop_v2_oversampled.csv']},
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="large_final_sweep")
    wandb.agent(sweep_id, function=train_model_sweep) # , count=4