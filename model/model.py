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
    def __init__(self, sample_rate: int = 16000, stride: int = 1):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("SincNet only supports 16kHz audio for now.")
        self.sample_rate = sample_rate
        self.stride = stride

        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        self.conv1d.append(
            Encoder(
                ParamSincFB(
                    80,
                    251,
                    stride=self.stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
            )
        )
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(80, affine=True))

        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def receptive_field_center(self, frame: int = 0) -> int:

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

 



class VSegmentationModel(pl.LightningModule):
 
    def __init__(
        self,
        sincnet={"stride": 10},
        lstm={
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "dropout": 0.5,
            "batch_first": True,
        },
        sample_rate: int = 16000,
        batch_size=32,
        duration=5,
        num_classes=4,
    ):
        super().__init__()

        self.duration = duration
        self.batch_size = batch_size
        self.num_classes = num_classes
        sincnet["sample_rate"] = sample_rate
        self.save_hyperparameters()
        self.sincnet = SincNet(**sincnet)
        self.core = nn.LSTM(input_size=60, **lstm)
        num_out_features = lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1)
        num_out_features_halved = num_out_features // 2
        self.linear = nn.ModuleList(
            [
                nn.Linear(
                    in_features=num_out_features,
                    out_features=num_out_features_halved,
                    bias=True,
                ),
                nn.Linear(
                    in_features=num_out_features_halved,
                    out_features=num_out_features_halved,
                    bias=True,
                ),
            ]
        )
        self.classifier = nn.Linear(
            in_features=num_out_features_halved, out_features=self.num_classes
        )
        self.activation = nn.Sigmoid()
        self.get_num_frames()

        self.validation_metric = MetricCollection(
            [
                AUROC(
                    self.num_frames ==self.num_classes,
                    pos_label=1,
                    average="macro",
                    compute_on_step=False,
                ),
                OptimalDiarizationErrorRate(),
                OptimalDiarizationErrorRateThreshold(),
                OptimalSpeakerConfusionRate(),
                OptimalMissedDetectionRate(),
                OptimalFalseAlarmRate(),
            ]
        )

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
