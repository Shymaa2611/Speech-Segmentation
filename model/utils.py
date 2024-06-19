from typing import List
from numbers import Number
from typing import Optional, Tuple, Union
import networkx as nx
import math
import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple
from functools import singledispatch, partial
from scipy.optimize import linear_sum_assignment
from typing import Optional
import numpy as np
import torch
from pyannote.core import SlidingWindowFeature
from torchmetrics import Metric
@singledispatch
def permutate(y1, y2, cost_func: Optional[Callable] = None, return_cost: bool = False):
     pass
    
def mse_cost_func(Y, y, **kwargs):
    return torch.mean(F.mse_loss(Y, y, reduction="none"), dim=0)


def mae_cost_func(Y, y, **kwargs):
    return torch.mean(torch.abs(Y - y), axis=0)


@permutate.register
def permutate_torch(
    y1: torch.Tensor,
    y2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:

    batch_size, num_samples, num_classes_1 = y1.shape

    if len(y2.shape) == 2:
        y2 = y2.expand(batch_size, -1, -1)

    if len(y2.shape) != 3:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_classes)."
        raise ValueError(msg)

    batch_size_, num_frames, num_classes_2 = y2.shape
    if batch_size != batch_size_:
        msg = f"Batch size mismatch: {batch_size} != {batch_size_}."
        raise ValueError(msg)

    if cost_func is None:
        cost_func = mse_cost_func

    permutations = []
    permutated_y2 = []

    if return_cost:
        costs = []

    permutated_y2 = torch.zeros(y1.shape, device=torch.device('cpu'), dtype=torch.float32)

    for b, (y1_, y2_) in enumerate(zip(y1, y2)):
        cost = torch.stack(
            [
              cost_func(y2_[:, np.newaxis], y1_[:, i : i + 1])

                for i in range(num_classes_1)
            ],
            dim=1,
        )

        # Find the permutation that minimizes the total cost using the Hungarian algorithm
        _, perm = linear_sum_assignment(cost.mean(dim=0))

        for i, j in enumerate(perm):
            permutated_y2[b, :, i] = y2_[:, j]

        permutations.append(tuple(perm))

        if return_cost:
            costs.append(cost)

    if return_cost:
        return permutated_y2, permutations, torch.stack(costs)

    return permutated_y2, permutations

@permutate.register
def permutate_numpy(
    y1: np.ndarray,
    y2: np.ndarray,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int]]]:

    output = permutate(
        torch.from_numpy(y1),
        torch.from_numpy(y2),
        cost_func=cost_func,
        return_cost=return_cost,
    )

    if return_cost:
        permutated_y2, permutations, costs = output
        return permutated_y2.numpy(), permutations, costs.numpy()

    permutated_y2, permutations = output
    return permutated_y2.numpy(), permutations


def build_permutation_graph(
    segmentations: SlidingWindowFeature,
    onset: float = 0.5,
    cost_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mae_cost_func,
) -> nx.Graph:
    cost_func = partial(cost_func, onset=onset)

    chunks = segmentations.sliding_window
    num_chunks, num_frames, _ = segmentations.data.shape
    max_lookahead = math.floor(chunks.duration / chunks.step - 1)
    lookahead = 2 * (max_lookahead,)

    permutation_graph = nx.Graph()

    for C, (chunk, segmentation) in enumerate(segmentations):
        for c in range(max(0, C - lookahead[0]), min(num_chunks, C + lookahead[1] + 1)):

            if c == C:
                continue
            shift = round((C - c) * num_frames * chunks.step / chunks.duration)

            if shift < 0:
                shift = -shift
                this_segmentations = segmentation[shift:]
                that_segmentations = segmentations[c, : num_frames - shift]
            else:
                this_segmentations = segmentation[: num_frames - shift]
                that_segmentations = segmentations[c, shift:]

            # find the optimal one-to-one mapping
            _, (permutation,), (cost,) = permutate(
                this_segmentations[np.newaxis],
                that_segmentations,
                cost_func=cost_func,
                return_cost=True,
            )

            for this, that in enumerate(permutation):

                this_is_active = np.any(this_segmentations[:, this] > onset)
                that_is_active = np.any(that_segmentations[:, that] > onset)

                if this_is_active:
                    permutation_graph.add_node((C, this))

                if that_is_active:
                    permutation_graph.add_node((c, that))

                if this_is_active and that_is_active:
                    permutation_graph.add_edge(
                        (C, this), (c, that), cost=cost[this, that]
                    )

    return permutation_graph


def _der_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[torch.Tensor, float] = 0.5,
    reduce: str = "batch",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  
    prd_batch_size, prd_num_speakers, prd_num_frames = preds.shape
    tgt_batch_size, tgt_num_speakers, tgt_num_frames = target.shape

    if prd_batch_size != tgt_batch_size:
        raise ValueError(f"Batch size mismatch: {prd_batch_size} != {tgt_batch_size}.")

    if prd_num_frames != tgt_num_frames:
        raise ValueError(
            f"Number of frames mismatch: {prd_num_frames} != {tgt_num_frames}."
        )

    if prd_num_speakers > tgt_num_speakers:
        target = F.pad(target, (0, 0, 0, prd_num_speakers - tgt_num_speakers))
    elif prd_num_speakers < tgt_num_speakers:
        preds = F.pad(preds, (0, 0, 0, tgt_num_speakers - prd_num_speakers))

    scalar_threshold = isinstance(threshold, Number)
    if scalar_threshold:
        threshold = torch.tensor([threshold], dtype=preds.dtype, device=preds.device)

    permutated_preds, _ = permutate(
        torch.transpose(target, 1, 2), torch.transpose(preds, 1, 2)
    )
    permutated_preds = torch.transpose(permutated_preds, 1, 2)
    hypothesis = (permutated_preds.unsqueeze(-1) > threshold).float()
    speech_total = 1.0 * torch.sum(target, 1)
    target = target.unsqueeze(-1)
    detection_error = torch.sum(hypothesis, 1) - torch.sum(target, 1)
    false_alarm = torch.maximum(detection_error, torch.zeros_like(detection_error))
    missed_detection = torch.maximum(
        -detection_error, torch.zeros_like(detection_error)
    )
    speaker_confusion = torch.sum((hypothesis != target) * hypothesis, 1) - false_alarm

    if reduce == "frame":
        if scalar_threshold:
            return (
                false_alarm[:, :, 0],
                missed_detection[:, :, 0],
                speaker_confusion[:, :, 0],
                speech_total,
            )
        return false_alarm, missed_detection, speaker_confusion, torch.sum(target, 1)

    speech_total = torch.sum(speech_total, 1)
    
    false_alarm = torch.sum(false_alarm, 1)
    missed_detection = torch.sum(missed_detection, 1)
    speaker_confusion = torch.sum(speaker_confusion, 1)
   
    if reduce == "chunk":
        if scalar_threshold:
            return (
                false_alarm[:, 0],
                missed_detection[:, 0],
                speaker_confusion[:, 0],
                speech_total,
            )
        return false_alarm, missed_detection, speaker_confusion, speech_total

    speech_total = torch.sum(speech_total, 0)
    false_alarm = torch.sum(false_alarm, 0)
    missed_detection = torch.sum(missed_detection, 0)
    speaker_confusion = torch.sum(speaker_confusion, 0)

    if scalar_threshold:
        return (
            false_alarm[0],
            missed_detection[0],
            speaker_confusion[0],
            speech_total,
        )

    return false_alarm, missed_detection, speaker_confusion, speech_total


def _der_compute(
    false_alarm: torch.Tensor,
    missed_detection: torch.Tensor,
    speaker_confusion: torch.Tensor,
    speech_total: torch.Tensor,
) -> torch.Tensor:
   
    return (false_alarm + missed_detection + speaker_confusion) / (speech_total + 1e-8)



class DiarizationErrorRate(Metric):
  
    higher_is_better = False
    is_differentiable = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()

        self.threshold = threshold

        self.add_state("false_alarm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "missed_detection", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "speaker_confusion", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("speech_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
            preds, target, threshold=self.threshold
        )
        self.false_alarm += false_alarm
        self.missed_detection += missed_detection
        self.speaker_confusion += speaker_confusion
        self.speech_total += speech_total

    def compute(self):
        return _der_compute(
            self.false_alarm,
            self.missed_detection,
            self.speaker_confusion,
            self.speech_total,
        )


class SpeakerConfusionRate(DiarizationErrorRate):
    def compute(self):
        return self.speaker_confusion / (self.speech_total + 1e-8)


class FalseAlarmRate(DiarizationErrorRate):
    def compute(self):
        return self.false_alarm / (self.speech_total + 1e-8)


class MissedDetectionRate(DiarizationErrorRate):
    def compute(self):
        return self.missed_detection / (self.speech_total + 1e-8)


class OptimalDiarizationErrorRate(Metric):
    higher_is_better = False
    is_differentiable = False

    def __init__(self, threshold: Optional[torch.Tensor] = None):
        super().__init__()

        threshold = threshold or torch.linspace(0.0, 1.0, 51)
        self.add_state("threshold", default=threshold, dist_reduce_fx="mean")
        (num_thresholds,) = threshold.shape
        self.add_state(
            "FalseAlarm",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "MissedDetection",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "SpeakerConfusion",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state("speech_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
    
        false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
            preds, target, threshold=self.threshold
        )
        self.FalseAlarm += false_alarm
        self.MissedDetection += missed_detection
        self.SpeakerConfusion += speaker_confusion
        self.speech_total += speech_total

    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        opt_der, _ = torch.min(der, dim=0)

        return opt_der


class OptimalDiarizationErrorRateThreshold(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        opt_threshold = self.threshold[opt_threshold_idx]

        return opt_threshold


class OptimalSpeakerConfusionRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.SpeakerConfusion[opt_threshold_idx] / (self.speech_total + 1e-8)


class OptimalFalseAlarmRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.FalseAlarm[opt_threshold_idx] / (self.speech_total + 1e-8)


class OptimalMissedDetectionRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.MissedDetection[opt_threshold_idx] / (self.speech_total + 1e-8)


def conv1d_num_frames(
    num_samples, kernel_size=5, stride=1, padding=0, dilation=1
) -> int:
    return 1 + (num_samples + 2 * padding - dilation * (kernel_size - 1) - 1) // stride


def multi_conv_num_frames(
    num_samples: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    num_frames = num_samples
    for k, s, p, d in zip(kernel_size, stride, padding, dilation):
        num_frames = conv1d_num_frames(
            num_frames, kernel_size=k, stride=s, padding=p, dilation=d
        )

    return num_frames


def conv1d_receptive_field_size(num_frames=1, kernel_size=5, stride=1, dilation=1):
  
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return effective_kernel_size + (num_frames - 1) * stride


def multi_conv_receptive_field_size(
    num_frames: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    receptive_field_size = num_frames

    for k, s, d in reversed(list(zip(kernel_size, stride, dilation))):
        receptive_field_size = conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=k,
            stride=s,
            dilation=d,
        )
    return receptive_field_size


def conv1d_receptive_field_center(
    frame=0, kernel_size=5, stride=1, padding=0, dilation=1
) -> int:
    
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return frame * stride + (effective_kernel_size - 1) // 2 - padding


def multi_conv_receptive_field_center(
    frame: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    receptive_field_center = frame
    for k, s, p, d in reversed(list(zip(kernel_size, stride, padding, dilation))):
        receptive_field_center = conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )

    return receptive_field_center
 
def interpolate(target: torch.Tensor, weight: Optional[torch.Tensor] = None):
    num_frames = target.shape[1]
    if weight is not None and weight.shape[1] != num_frames:
        weight = F.interpolate(
            weight.transpose(1, 2),
            size=num_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return weight

def binary_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(target.shape) == 2:
        target = target.unsqueeze(dim=2)

    if weight is None and target.shape==prediction.shape:
        return F.binary_cross_entropy(prediction, target.float())

    else:
        weight = interpolate(target, weight=weight)

        return F.binary_cross_entropy(
            prediction, target.float(), 
        )

