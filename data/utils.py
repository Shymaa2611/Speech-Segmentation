import math
import random
import warnings
from io import IOBase
from pathlib import Path
from typing import Mapping, Optional, Text, Tuple, Union
from pyannote.core import Segment
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from pyannote.database import FileFinder, Protocol, get_annotated
torchaudio.set_audio_backend("soundfile")

AudioFile = Union[Text, Path, IOBase, Mapping]

def get_torchaudio_info(file: AudioFile):
    info = torchaudio.info(file["audio"])
    if isinstance(file["audio"], IOBase):
        file["audio"].seek(0)

    return info
 
class Audio:
    PRECISION = 0.001

    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor:
        rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
        return waveform / (rms + 1e-8)

    @staticmethod
    def validate_file(file: AudioFile) -> Mapping:
        if isinstance(file, Mapping):
            pass

        elif isinstance(file, (str, Path)):
            file = {"audio": str(file), "uri": Path(file).stem}

        elif isinstance(file, IOBase):
            return {"audio": file, "uri": "stream"}

        else:
            pass
            #raise ValueError(AudioFileDocString)

        if "waveform" in file:
            waveform: Union[np.ndarray, Tensor] = file["waveform"]
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]:
                raise ValueError(
                    "'waveform' must be provided as a (channel, time) torch Tensor."
                )

            sample_rate: int = file.get("sample_rate", None)
            if sample_rate is None:
                raise ValueError(
                    "'waveform' must be provided with their 'sample_rate'."
                )

            file.setdefault("uri", "waveform")

        elif "audio" in file:
            if isinstance(file["audio"], IOBase):
                return file

            path = Path(file["audio"])
            if not path.is_file():
                raise ValueError(f"File {path} does not exist")

            file.setdefault("uri", path.stem)

        else:
            raise ValueError(
                "Neither 'waveform' nor 'audio' is available for this file."
            )

        return file

    def __init__(self, sample_rate=None, mono=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(self, waveform: Tensor, sample_rate: int) -> Tensor:
      
        num_channels = waveform.shape[0]
        if num_channels > 1:
            if self.mono == "random":
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix":
                waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            sample_rate = self.sample_rate

        return waveform, sample_rate

    def get_duration(self, file: AudioFile) -> float:

        file = self.validate_file(file)

        if "waveform" in file:
            frames = len(file["waveform"].T)
            sample_rate = file["sample_rate"]

        else:
            if "torchaudio.info" in file:
                info = file["torchaudio.info"]
            else:
                info = get_torchaudio_info(file)

            frames = info.num_frames
            sample_rate = info.sample_rate

        return frames / sample_rate

    def get_num_samples(
        self, duration: float, sample_rate: Optional[int] = None
    ) -> int:
        
        sample_rate = sample_rate or self.sample_rate

        if sample_rate is None:
            raise ValueError(
                "`sample_rate` must be provided to compute number of samples."
            )

        return math.floor(duration * sample_rate)

    def __call__(self, file: AudioFile) -> Tuple[torch.Tensor, int]:

        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]

        elif "audio" in file:
            waveform, sample_rate = torchaudio.load(file["audio"])

            # rewind if needed
            if isinstance(file["audio"], IOBase):
                file["audio"].seek(0)

        channel = file.get("channel", None)

        if channel is not None:
            waveform = waveform[channel : channel + 1]

        return self.downmix_and_resample(waveform, sample_rate)

    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        duration: Optional[float] = None,
        mode="raise",
    ) -> Tuple[Tensor, int]:
  
        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            frames = waveform.shape[1]
            sample_rate = file["sample_rate"]

        elif "torchaudio.info" in file:
            info = file["torchaudio.info"]
            frames = info.num_frames
            sample_rate = info.sample_rate

        else:
            info = get_torchaudio_info(file)
            frames = info.num_frames
            sample_rate = info.sample_rate

        channel = file.get("channel", None)

        # infer which samples to load from sample rate and requested chunk
        start_frame = math.floor(segment.start * sample_rate)

        if duration:
            num_frames = math.floor(duration * sample_rate)
            end_frame = start_frame + num_frames

        else:
            end_frame = math.floor(segment.end * sample_rate)
            num_frames = end_frame - start_frame

        if mode == "raise":
            if num_frames > frames:
                raise ValueError(
                    f"requested fixed duration ({duration:6f}s, or {num_frames:d} frames) is longer "
                    f"than file duration ({frames / sample_rate:.6f}s, or {frames:d} frames)."
                )

            if end_frame > frames + math.ceil(self.PRECISION * sample_rate):
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0., {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )
            else:
                end_frame = min(end_frame, frames)
                start_frame = end_frame - num_frames

            if start_frame < 0:
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0, {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )

        elif mode == "pad":
            pad_start = -min(0, start_frame)
            pad_end = max(end_frame, frames) - frames
            start_frame = max(0, start_frame)
            end_frame = min(end_frame, frames)
            num_frames = end_frame - start_frame

        if "waveform" in file:
            data = file["waveform"][:, start_frame:end_frame]

        else:
            try:
                data, _ = torchaudio.load(
                    file["audio"], frame_offset=start_frame, num_frames=num_frames
                )
                # rewind if needed
                if isinstance(file["audio"], IOBase):
                    file["audio"].seek(0)
            except RuntimeError:
                if isinstance(file["audio"], IOBase):
                    msg = "torchaudio failed to seek-and-read in file-like object."
                    raise RuntimeError(msg)

                msg = (
                    f"torchaudio failed to seek-and-read in {file['audio']}: "
                    f"loading the whole file instead."
                )

                warnings.warn(msg)
                waveform, sample_rate = self.__call__(file)
                data = waveform[:, start_frame:end_frame]

                # storing waveform and sample_rate for next time
                # as it is very likely that seek-and-read will
                # fail again for this particular file
                file["waveform"] = waveform
                file["sample_rate"] = sample_rate

        if channel is not None:
            data = data[channel : channel + 1, :]

        # pad with zeros
        if mode == "pad":
            data = F.pad(data, (pad_start, pad_end))

        return self.downmix_and_resample(data, sample_rate)


def check_protocol(protocol: Protocol) -> Protocol:
    get_duration = Audio(mono="downmix").get_duration
    try:
        file = next(protocol.train())
    except (AttributeError, NotImplementedError):
        msg = f"Protocol {protocol.name} does not define a training set."
        raise ValueError(msg)
    if "audio" not in file:

        if "waveform" in file:
            if "sample_rate" not in file:
                msg = f'Protocol {protocol.name} provides audio with "waveform" key but is missing a "sample_rate" key.'
                raise ValueError(msg)

        else:

            file_finder = FileFinder()
            try:
                _ = file_finder(file)

            except (KeyError, FileNotFoundError):
                msg = (
                    f"Protocol {protocol.name} does not provide the path to audio files. "
                    f"See pyannote.database documentation on how to add an 'audio' preprocessor."
                )
                raise ValueError(msg)
            else:
                protocol.preprocessors["audio"] = file_finder
                msg = (
                    f"Protocol {protocol.name} does not provide the path to audio files: "
                    f"adding an 'audio' preprocessor for you. See pyannote.database documentation "
                    f"on how to do that yourself."
                )
                print(msg)

    if "waveform" not in file and "torchaudio.info" not in file:

        protocol.preprocessors["torchaudio.info"] = get_torchaudio_info
        msg = (
            f"Protocol {protocol.name} does not precompute the output of torchaudio.info(): "
            f"adding a 'torchaudio.info' preprocessor for you to speed up dataloaders. "
            f"See pyannote.database documentation on how to do that yourself."
        )
        print(msg)

    if "annotated" not in file:

        if "duration" not in file:
            protocol.preprocessors["duration"] = get_duration

        protocol.preprocessors["annotated"] = get_annotated

        msg = (
            f"Protocol {protocol.name} does not provide the 'annotated' regions: "
            f"adding an 'annotated' preprocessor for you. See pyannote.database documentation "
            f"on how to do that yourself."
        )
        print(msg)

    has_scope = "scope" in file
    has_classes = "classes" in file
    validation_method = "development"

    try:
        _ = next(getattr(protocol, validation_method)())
    except (AttributeError, NotImplementedError):
        has_validation = False
    else:
        has_validation = True

    checks = {
        "has_validation": has_validation,
        "has_scope": has_scope,
        "has_classes": has_classes,
    }

    return protocol, checks

