# %%
import os
import math
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
import numpy as np
import tqdm
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional
from sklearn import random_projection
import pytorch_lightning as pl

from torchvision.datasets import CIFAR10
from torchvision import transforms

import sys
sys.path.append('/workspace/WorldCrops/py_files')
sys.path.append('..')

#import model
#from model import *
from processing import *
#from tsai.all import *

import torch
import breizhcrops as bc

# %%
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from sklearn.metrics import classification_report,accuracy_score

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Max(nn.Module):
    def forward(self, x): return x.max(1)[0]

class Attention_LM(pl.LightningModule):

    def __init__(self, input_dim = 13, seq_length = 14, num_classes = 7, d_model = 64, n_head = 2, d_ffn = 128, nlayers = 2, dropout = 0.018, activation="relu", lr = 0.0002, batch_size  = 3, seed=42, PositonalEncoding = False):
        super().__init__()
        """
        Args:
            input_dim: amount of input dimensions -> Sentinel2 has 13 bands
            num_classes: amount of target classes
            dropout: default = 0.018
            d_model: default = 64 #number of expected features
            n_head: default = 2 #number of heads in multiheadattention models
            d_ff: default = 128 #dim of feedforward network 
            nlayers: default = 2 #number of encoder layers
            + : https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        Input:
            batch size(N) x T x D
        Output
            batch size(N) x Targets
        """

        self.model_type = 'Transformer_LM'
        pl.seed_everything(seed)
        self.PositionalEncoding = PositionalEncoding
        self.seq_length = seq_length

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.ce = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        # Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout = dropout, activation=activation, batch_first=True)

        if self.PositionalEncoding:

            self.backbone = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.ReLU(),
                PositionalEncoding(d_model = d_model, dropout = 0),
                nn.TransformerEncoder(encoder_layers, nlayers, nn.LayerNorm(d_model)),
                Max(),
                nn.ReLU()
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.ReLU(),
                nn.TransformerEncoder(encoder_layers, nlayers, nn.LayerNorm(d_model)),
                Max(),
                nn.ReLU()
            )

        self.outlinear = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )


    def forward(self,x):
        # N x T x D -> N x T x d_model / Batch First!
        embedding = self.backbone(x)
        x = self.outlinear(embedding)
        #torch.Size([N,num_classes ])
        x = F.log_softmax(x, dim=-1)
        #torch.Size([N, num_classes])
        return {'x' : x, 'embedding' : embedding}

    def training_step(self, batch, batch_idx):
        _,x, y = batch
        data = self.forward(x)
        y_pred = data['x']
        embedding = data['embedding']

        loss = self.ce(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y, 'embedding': embedding}

    def training_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()
        embedding_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])
            embedding_list.append(item['embedding'])

        acc = accuracy_score(torch.cat(y_true_list).cpu(),torch.cat(y_pred_list).cpu())
        #overall accuracy
        self.log('OA',round(acc,2)) 

        self.logger.experiment.add_embedding(
            torch.cat(embedding_list), # Encodings per epoch
            metadata=torch.cat(y_pred_list)) # Adding the labels per image to the plot
            #label_img= ken plan grad

    def validation_step(self, val_batch, batch_idx):
        _,x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}


    def validation_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list).cpu(),torch.cat(y_pred_list).cpu())
        #overall accuracy
        self.log('OA',round(acc,2))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, test_batch, batch_idx):
        _,x, y = test_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        #self.log('test_results', loss,on_step=True,prog_bar=True)

        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        #gets all results from test_steps
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list).cpu(),torch.cat(y_pred_list).cpu())
        #Overall accuracy
        self.log('OA',round(acc,2))
        


# %%

# %%
test
# %%
train = pd.read_excel(
    "/workspace/WorldCrops/data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")

train = utils.clean_bavarian_labels(train)
#delete class 0
train = train[train.NC != 0]

#rewrite the 'id' as we deleted one class
newid = 0
groups = train.groupby('id')
for id, group in groups:
    train.loc[train.id == id, 'id'] = newid
    newid +=1

years = [2016,2017,2018]
train = utils.augment_df(train, years)

#train = utils.clean_bavarian_labels(bavaria_train)
feature_list = train.columns[train.columns.str.contains('B')]
ts_dataset = TimeSeriesPhysical(train, feature_list.tolist(), 'NC')
dataloader_train = torch.utils.data.DataLoader(
    ts_dataset, batch_size=32, shuffle=True,drop_last=False, num_workers=2)

# %%
train.columns[train.columns.str.contains('B')]
# %%
dataiter = iter(dataloader_train)
(x1,x2),x, y = next(dataiter)

# %%
from pytorch_lightning.loggers import TensorBoardLogger

test = Attention_LM(num_classes=7)
logger = TensorBoardLogger("tb_logs", name="my_model")

trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5,logger=logger)
trainer.fit(test, dataloader_train)
