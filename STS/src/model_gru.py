import torch
import torchmetrics
import transformers
import pytorch_lightning as pl

# from transformers import ElectraModel
from transformers import AutoTokenizer, AutoModel

import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        # Attention 가중치를 계산하기 위한 선형 변환
        self.attention = nn.Linear(hidden_size, 1)  # 각 time step에 대해 1개의 가중치 값 계산

    def forward(self, rnn_output):
        # rnn_output: [batch_size, seq_len, hidden_size]
        attn_weights = self.attention(rnn_output)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # 시퀀스 차원에 대해 softmax 적용하여 가중치 합을 1로 만듦

        # 각 time step에 대해 가중치를 곱한 후 합산
        weighted_output = torch.sum(attn_weights * rnn_output, dim=1)  # [batch_size, hidden_size]
        
        return weighted_output

class ModelWithAttention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ModelWithAttention, self).__init__()
        
        # self.rnn = nn.LSTM(
        #     input_size=hidden_size, hidden_size=hidden_size // 2, 
        #     num_layers=2, bidirectional=True, batch_first=True, dropout=0.2
        # )  # BiLSTM 추가
        self.rnn = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size // 2, 
            num_layers=2, bidirectional=True, batch_first=True, dropout=dropout
        )  # BiGRU 추가
        
        self.attention = AttentionLayer(hidden_size)  # Attention Layer 추가
        self.fc = nn.Linear(hidden_size, 1)  # 최종 출력 레이어 (유사도 예측)

    def forward(self, x, attention_mask):
        rnn_output, _ = self.rnn(x)  # LSTM 출력 [batch_size, seq_len, hidden_size]
        
        # Attention Layer 적용
        weighted_output = self.attention(rnn_output)  # [batch_size, hidden_size]

        # 최종 유사도 예측
        logits = self.fc(weighted_output)
        return logits


class ModelGRU(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = CFG['train']['model_name']
        self.lr = CFG['train']['LR']
        loss_name = "torch.nn." + CFG['train']['LossF']
        optim_name = "torch.optim." + CFG['train']['optim']
        self.dropout = CFG['train']['dropout']
        
        # 사용할 모델을 호출
        # self.plm = transformers.AutoModel.from_pretrained(
        #     pretrained_model_name_or_path=self.model_name, num_labels=1)
        self.plm = transformers.AutoModel.from_pretrained(self.model_name)
        self.plm.config.hidden_dropout_prob = self.dropout
        self.plm.config.attention_probs_dropout_prob = self.dropout
        
        self.hidden_size = self.plm.config.hidden_size
        self.num_layers = self.plm.config.num_hidden_layers
        self.rnn_with_attention = ModelWithAttention(self.hidden_size, self.dropout)
        
        # self.freeze_range = self.num_layers // 2
        self.freeze_range = (self.num_layers // 4) * 3

        for name, param in self.plm.named_parameters():
            if 'layer' in name:
                layer_num = int(name.split('.')[2])  # layer.{n}에서 n 추출
                if layer_num < self.freeze_range:
                    param.requires_grad = False
        
        self.loss_func = eval(loss_name)()
        self.optim = eval(optim_name)

    def forward(self, x, attention_mask):    
        x = self.plm(input_ids=x, attention_mask=attention_mask)['last_hidden_state']  # shape: [batch_size, seq_len, hidden_size]
        
        x = self.rnn_with_attention(x, attention_mask)  # shape: [batch_size, 1]
        
        return x

    def training_step(self, batch, batch_idx):
        x, attention_mask, y = batch
        logits = self(x, attention_mask)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, attention_mask, y = batch
        logits = self(x, attention_mask)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, attention_mask, y = batch
        logits = self(x, attention_mask)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x, attention_mask = batch
        logits = self(x, attention_mask)

        return logits.squeeze()

    def configure_optimizers(self):
        # 파라미터 그룹 설정 (Layer-wise Learning Rate Decay 적용)
        layerwise_lr = []
        lr = self.lr  # 기본 학습률
        
        
        # 6번째 레이어부터 학습
        for layer_idx in range(self.freeze_range, self.plm.config.num_hidden_layers):
            layerwise_lr.append({
                'params': self.plm.encoder.layer[layer_idx].parameters(),
                'lr': lr  # 각 레이어에 대해 학습률 설정
            })
            lr *= 0.95  # 학습률을 95%로 감소시키며 설정   


        # LSTM 레이어는 고정된 학습률 적용
        lstm_params = {
            'params': self.rnn_with_attention.rnn.parameters(),
            'lr': self.lr  # LSTM에 고정된 학습률 적용
        }
        layerwise_lr.append(lstm_params)

        optimizer = torch.optim.AdamW(layerwise_lr, lr=self.lr, weight_decay=0.01)

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda epoch: 0.95 ** epoch,
        #     last_epoch=-1,
        #     verbose=False)
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer=optimizer,
        #     step_size=10,
        #     gamma=0.7,
        #     verbose=True)

        # lr_scheduler = {
        # 'scheduler': scheduler,
        # 'name': 'LR_schedule'
        # }

        return [optimizer]#, [lr_scheduler]