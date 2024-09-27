# hyperparam 고정값
admin: heejun
patience: 5 
seed: 42
train:
  model_name: klue/roberta-base
  batch_size: 16
  epoch: 1
  LR: 0.00003
  LossF: L1Loss
  optim: AdamW
  shuffle: True

# 실행시 필요한 args
python3 train.py
python3 inference.py --target_folder <folder name in lightning_logs>
















# baseline code
--model_name
--batch_size
--max_epoch
--shuffle
--learning_rate

# train.py 실행:
python3 train.py --model_name 'klue/roberta-large' --batch_size 16 --max_epoch 15 --learning_rate 3e-6

# inference.py 실행: --model_path 폴더명 추가해야함
python3 inference.py --model_name 'klue/roberta-large' --batch_size 16 --model_path 'lightning_logs/0.9251_klue-roberta-large'
python3 inference.py --model_name 'klue/roberta-large' --batch_size 16 --model_path 'lightning_logs/0.9153_klue-roberta-large'

# restore_model.py 실행:
python3 restore_model.py --model_name 'klue/roberta-large' --batch_size 16 --max_epoch 15 --learning_rate 3e-6





