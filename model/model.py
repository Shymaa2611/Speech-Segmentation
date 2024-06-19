import math
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import (
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
    
)
from utils import permutate,binary_cross_entropy
from functools import lru_cache
from asteroid_filterbanks import Encoder, ParamSincFB
from utils import *

from torchmetrics import AUROC, MetricCollection

from model.utils import multi_conv_num_frames, multi_conv_receptive_field_center, multi_conv_receptive_field_size

class SincNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=16000):
        super(SincNet, self).__init__()
       # if sample_rate != 16000:
        #    raise NotImplementedError("SincNet only supports 16kHz audio .")
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_rate=sample_rate
        self.coeffs = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.b = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data.normal_(0, 1)
        self.b.data.zero_()

    def forward(self, x):
        sinc = torch.sin(self.coeffs * self.sample_rate) / (self.coeffs * self.sample_rate)
        h = sinc.sum(2) + self.b.unsqueeze(1)
        return nn.functional.conv1d(x, h.unsqueeze(-1), stride=self.kernel_size // 2)
    
class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(13, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_size // 4), 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64,2)

    def get_num_frames(self):
        x = torch.randn(
            (
                self.batch_size,
                1,  
                int(self.hparams.sample_rate * self.duration),
            ),
        )
        with torch.no_grad():
            self.num_frames = self.sincnet(x).shape[-1]
        return self.num_frames

    def forward(self, waveforms):
        outputs = self.sincnet(waveforms)
        outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
        outputs, _ = self.core(outputs)
        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))

    def loss_func(
        self, prediction, target,
    ):
       

        return binary_cross_entropy(prediction, target.float())

    def training_step(self, batch, batch_idx):
        target = batch["y"]
        waveform = batch["X"]
        prediction = self.forward(waveform)
        permutated_prediction, _ = permutate(target, prediction)
        loss = self.loss_func(permutated_prediction, target)
        self.log(
            "TrainLoss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        X, y = batch["X"], batch["y"]
        y_pred = self.forward(X)
        target = y
        permutated_prediction, _ = permutate(target, y_pred)
        loss = self.loss_func(permutated_prediction, target)
        self.log(
            "ValLoss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )

        if target.shape[-1] != y_pred.shape[-1]:
            pad_func = nn.ConstantPad1d((0, y_pred.shape[-1] - target.shape[-1]), 0)
            target = pad_func(y)

        self.validation_metric(
            y_pred.squeeze()== torch.transpose(y_pred, 1, 2),
            target.squeeze()== torch.transpose(target, 1, 2),
        )

        self.log_dict(
            self.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if batch_idx == 3:
            X = X.cpu().numpy()
            y = y.float().cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            num_samples = min(self.batch_size, 9)
            nrows = math.ceil(math.sqrt(num_samples))
            ncols = math.ceil(num_samples / nrows)
            fig, axes = plt.subplots(
                nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
            )

            y[y == 0] = np.NaN
            if len(y.shape) == 2:
                y = y[:, :, np.newaxis]
            y *= np.arange(y.shape[2])
            for sample_idx in range(num_samples):
                row_idx = sample_idx // nrows
                col_idx = sample_idx % ncols
                ax_ref = axes[row_idx * 2 + 0, col_idx]
                sample_y = y[sample_idx]
                ax_ref.plot(sample_y)
                ax_ref.set_xlim(0, len(sample_y))
                ax_ref.set_ylim(-1, sample_y.shape[1])
                ax_ref.get_xaxis().set_visible(False)
                ax_ref.get_yaxis().set_visible(False)
                ax_hyp = axes[row_idx * 2 + 1, col_idx]
                sample_y_pred = y_pred[sample_idx]

                ax_hyp.plot(sample_y_pred)
                ax_hyp.set_ylim(-0.1, 1.1)
                ax_hyp.set_xlim(0, len(sample_y))
                ax_hyp.get_xaxis().set_visible(False)

            plt.tight_layout()
            plt.savefig("figure.png")
            self.logger.log_image("ValSamples", ["figure.png"], self.current_epoch)

            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=12, factor=0.5, min_lr=1e-8
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "ValLoss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
