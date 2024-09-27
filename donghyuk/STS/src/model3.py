import torch
import torchmetrics
import transformers
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = CFG['train']['model_name']
        self.lr = CFG['train']['LR']
        loss_name = "torch.nn." + CFG['train']['LossF']
        optim_name = "torch.optim." + CFG['train']['optim']
        
        # 사용할 모델을 호출
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1)
        
        # Loss 함수 정의 (기본 MSELoss)
        self.loss_func = eval(loss_name)()
        self.optim = eval(optim_name)

    def forward(self, x):    
        x = self.plm(x)['logits']
        return x

    def weighted_mse_loss(self, logits, y):
        # Residuals (예측값과 실제값 차이)
        residuals = torch.abs(logits - y.float())
        
        # 잔차에 따른 가중치 부여 (exponential scaling)
        weights = torch.exp(residuals)  # 가중치가 잔차에 비례하게 적용됨
        
        # 기본 MSE 손실 계산
        losses = torch.nn.functional.mse_loss(logits, y.float(), reduction='none')
        
        # 가중치를 곱한 손실
        weighted_losses = losses * weights
        
        # 가중된 손실의 평균 계산
        return torch.mean(weighted_losses)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Weighted MSE Loss 적용
        loss = self.weighted_mse_loss(logits, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Weighted MSE Loss 적용
        loss = self.weighted_mse_loss(logits, y)
        self.log("val_loss", loss)

        # Pearson correlation coefficient
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Pearson correlation coefficient
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=0.01)  # Weight Decay 추가
        return [optimizer]
