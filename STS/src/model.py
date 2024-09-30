import torch
import torchmetrics
import transformers
import pytorch_lightning as pl

# from transformers import ElectraModel
from transformers import AutoTokenizer, AutoModel


class Model(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = CFG['train']['model_name']
        self.lr = CFG['train']['LR']
        loss_name = "torch.nn." + CFG['train']['LossF']
        optim_name = "torch.optim." + CFG['train']['optim']
        
        # 사용할 모델을 호출
        # self.plm = transformers.AutoModel.from_pretrained(
        #     pretrained_model_name_or_path=self.model_name, num_labels=1)
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1)
    
        self.freeze_layers = CFG['train']['freeze_layers']
                                    
        for name, param in self.plm.named_parameters():
            if 'layer' in name:
                layer_num = int(name.split('.')[3])  # layer.{n}에서 n 추출
                if layer_num < self.freeze_layers:
                    param.requires_grad = False
        
        
        self.loss_func = eval(loss_name)()
        self.optim = eval(optim_name)

    def forward(self, x, attention_mask):    
        x = self.plm(input_ids=x, attention_mask=attention_mask)['logits']

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
        optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=0.01)  # Weight Decay 추가

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,       # 첫 주기(epoch 길이)
            T_mult=2,    # 각 주기 이후 몇 배로 늘릴지
            eta_min=1e-6 # 최소 학습률
        )

        return [optimizer], [scheduler]