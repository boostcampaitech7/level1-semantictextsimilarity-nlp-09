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
        self.loss_func = eval(loss_name)()
        self.optim = eval(optim_name)

    def forward(self, x):    
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=0.01)  # Weight Decay 추가
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