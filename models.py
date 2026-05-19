import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score, auroc
import numpy as np
from sklearn.metrics import roc_auc_score

class LightningDMC(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-5, batch_size=128, weight=None, num_classes=8, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = model.__class__.__name__

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.weight = weight
        self.num_classes = num_classes

        self.save_hyperparameters(ignore=['model'])
        self.predictions = []
        self.targets = []

    def forward(self, *args):
        return self.model(*args)

    def parse_batch(self, batch):
        if self.model_name == "DMF_tfidf":
            x, tfidf, y = batch
            return x, tfidf, y
        else:
            x, y = batch
            return x, None, y

    def training_step(self, batch, batch_idx):
        x, tfidf, y = self.parse_batch(batch)
        y_hat = self.model(x, tfidf) if tfidf is not None else self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.log('train_acc_epoch', accuracy(y_hat, y.int(), task='multiclass', average='macro', num_classes=self.num_classes), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, tfidf, y = self.parse_batch(batch)
        y_hat = self.model(x, tfidf) if tfidf is not None else self.model(x)
        val_loss = F.cross_entropy(y_hat, y)

        probs = torch.softmax(y_hat, dim=1)
        self.predictions.append(probs.detach().cpu())
        self.targets.append(y.detach().cpu())

        self.log("val_loss", val_loss, prog_bar=True)
        self.log('val_acc', accuracy(y_hat, y.int(), average='macro', task='multiclass', num_classes=self.num_classes), prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.predictions, dim=0).numpy()
        targets = torch.cat(self.targets, dim=0).numpy()
        self.predictions.clear()
        self.targets.clear()

        y_true_onehot = np.eye(self.num_classes)[targets]
        try:
            aucs = roc_auc_score(y_true_onehot, preds, average=None)
            for i, auc in enumerate(aucs):
                self.log(f"val_auc_class_{i}", auc, prog_bar=True)
        except Exception as e:
            print(f"Validation AUC calculation failed: {e}")

class DeepMicroClass(nn.Module):
    def __init__(self, num_classes=8) -> None:
        super().__init__()

        self.codon_transformer = CodonTransformer()

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

        self.base_channel = nn.Sequential(
            nn.Conv2d(1, 64, (6, 4)),
            nn.PReLU(),
            nn.Flatten(start_dim=2),
            nn.AvgPool1d(3),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3),
            nn.PReLU(),
            nn.AvgPool1d(3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.codon_channel = nn.Sequential(
            nn.Conv2d(1, 64, (2, 64)),
            nn.PReLU(),
            nn.Flatten(start_dim=2),
            nn.AvgPool1d(3),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3),
            nn.PReLU(),
            nn.AvgPool1d(3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x):
        interm_forward = self.base_channel(x)
        codon = self.codon_transformer(x)
        codon = self.codon_channel(codon)
        rev = torch.flip(x, dims=[-1, -2])
        interm_backward = self.base_channel(rev)
        codon_backward = self.codon_transformer(rev)
        codon_backward = self.codon_channel(codon_backward)
        z = torch.cat((interm_forward, codon, interm_backward, codon_backward, ), dim=1)
        z = self.dropout(z)
        z = self.fc(z)
        return z

class CodonTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codon_channel_num = 64
        self.codon_transformer = torch.zeros(64, 1, 3, 4)
        indicies = itertools.product(range(4), repeat=3)
        for i in range(self.codon_transformer.shape[0]):
            index = next(indicies)
            for j in range(3):
                self.codon_transformer[i, 0, j, index[j]] = 1
        self.codon_transformer = nn.Parameter(self.codon_transformer, requires_grad=False)
        self.padding_layers = [nn.ZeroPad2d((0, 0, 0, 2)), nn.ZeroPad2d((0, 0, 0, 1))]

    def forward(self, x):
        mod_len = int(x.shape[2] % 3)
        if mod_len != 2:
            x = self.padding_layers[mod_len](x)
        x = F.conv2d(x, self.codon_transformer) - 2
        x = F.relu(x)
        x = x.flatten(start_dim=2)

        x = x.view(-1, self.codon_channel_num, int(x.shape[2]//3), 3)
        x = x.transpose(2, 3)
        x = x.reshape(-1, 1, self.codon_channel_num, x.shape[-1]*3).transpose(2, 3)
        return x
