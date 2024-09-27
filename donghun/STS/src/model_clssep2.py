import torch
import torchmetrics
import transformers
import pytorch_lightning as pl
import torch.nn as nn

# from transformers import ElectraModel
from transformers import AutoTokenizer, AutoModel


class Model(pl.LightningModule):
    def __init__(self, CFG, num_labels=1):
        super(Model, self).__init__()
        self.save_hyperparameters()
        
        self.model_name = CFG['train']['model_name']
        self.lr = CFG['train']['LR']
        loss_name = "torch.nn." + CFG['train']['LossF']
        optim_name = "torch.optim." + CFG['train']['optim']
        
        # Load the tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=160)
        
        # Load the model
        self.plm = transformers.AutoModel.from_pretrained(self.model_name)
        self.optim = eval(optim_name)
        self.loss_func = eval(loss_name)()
        # Modify the out_proj layer for regression
        self.out_proj = nn.Linear(self.plm.config.hidden_size * 2, num_labels)  # Adjusted for concatenated output

    def forward(self, input_ids, attention_mask=None):
     outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)

     # [CLS] token representation
     cls_output = outputs.last_hidden_state[:, 0, :]  

     # Find the first occurrence of [SEP] token
     sep_token_index = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)
     
     if len(sep_token_index) > 0:
         sep_output = outputs.last_hidden_state[sep_token_index[:, 0], sep_token_index[:, 1], :]
         combined_output = torch.cat((cls_output, sep_output), dim=1)  # Concatenating [CLS] and [SEP] token representations
         logits = self.out_proj(combined_output)
     else:
         logits = self.out_proj(cls_output)

     return logits


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