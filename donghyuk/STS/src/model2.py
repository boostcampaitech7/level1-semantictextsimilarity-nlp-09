import torch
import torchmetrics
import transformers
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
class Model(pl.LightningModule):
    
    
    # def __init__(self, CFG):
    #     super().__init__()
    #     self.save_hyperparameters()
    #     self.model_name = CFG['train']['model_name']
    #     self.lr = CFG['train']['LR']
    #     loss_name = "torch.nn." + CFG['train']['LossF']
    #     optim_name = "torch.optim." + CFG['train']['optim']
    #     # 사용할 모델을 호출
    #     # self.plm = transformers.AutoModel.from_pretrained(
    #     #     pretrained_model_name_or_path=self.model_name, num_labels=1)
    #     self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
    #         pretrained_model_name_or_path=self.model_name, num_labels=1)
    #     self.loss_func = eval(loss_name)()
    #     self.optim = eval(optim_name)
    # def forward(self, x):
    #     x = self.plm(x)['logits']
    #     return x
    
    
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
        # Get the outputs from the base model
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        # Take the hidden state corresponding to the [CLS] token (the first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        # Get the index of [SEP] token (ID is typically 2)
        sep_token_index = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[1]
        if sep_token_index.size(0) > 0:
            # Average the hidden states after the first [SEP] token
            following_tokens_output = outputs.last_hidden_state[:, sep_token_index[0] + 1:, :]
            following_tokens_output = following_tokens_output.mean(dim=1)  # Shape: (batch_size, hidden_size)
            # Combine cls_output with following tokens (concatenate)
            combined_output = torch.cat((cls_output, following_tokens_output), dim=1)  # Shape: (batch_size, 2 * hidden_size)
            # Project the combined output
            logits = self.out_proj(combined_output)  # Shape: (batch_size, num_labels)
        else:
            # If no following tokens, just use cls_output
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
        return [optimizer]#, [lr_scheduler]