import itertools
from utils import Audio,AudioFile
import os
import random
import numpy as np
from collections import Counter
from pyannote.core import Segment, SlidingWindow
import torch
from torch.utils.data._utils.collate import default_collate

os.environ["PYANNOTE_DATABASE_CONFIG"] = "database.yml"


"""
input  => audio chunks
output => annotation of each chunks
##Training
   model accept audio that split into chunks as input 
   output of model is discrete data that accept by model
"""
def determine_max_num_speakers(dataset_part,duration):
        max_num_speakers=4,
        if max_num_speakers is None:
            num_speakers = []
            for file in dataset_part:
                start = file["annotated"][0].start
                end = file["annotated"][-1].end
                window = SlidingWindow(
                    start=start, end=end, duration=duration, step=1.0,
                )
                for chunk in window:
                    num_speakers.append(len(file["annotation"].crop(chunk).labels()))

            num_speakers, counts = zip(*list(Counter(num_speakers).items()))
            num_speakers, counts = np.array(num_speakers), np.array(counts)

            sorting_indices = np.argsort(num_speakers)
            num_speakers = num_speakers[sorting_indices]
            counts = counts[sorting_indices]

            max_num_speakers = max(
                2,
                num_speakers[np.where(np.cumsum(counts) / np.sum(counts) > 0.99)[0][0]],
            )

def prepare_data(protocol,duration):
      validation_samples=[]
      # Prepare training data
      train = []
      for f in protocol.train():
            file = dict()
            for key, value in f.items():
                if key == "annotated":
                    value = [
                        segment for segment in value if segment.duration > duration
                    ]
                    file["_annotated_duration"] = sum(
                        segment.duration for segment in value
                    )
                file[key] = value
       
                train.append(file)

      determine_max_num_speakers(train,duration)
      # Prepare validation data
      validation = []
      for f in protocol.development():
            for segment in f["annotated"]:
                if segment.duration < duration:
                    continue

                num_chunks = round(segment.duration //duration)

                for c in range(num_chunks):
                    start_time = segment.start + c * duration
                    chunk = Segment(start_time, start_time + duration)
                    validation.append((f, chunk))
                    validation_samples.append(prepare_chunk(f,chunk))
      return train,validation_samples

def prepare_chunk(file: AudioFile, chunk) -> dict:
        duration=5.0
        sample = dict()
        sample_rate=16000
        audio = Audio(sample_rate=sample_rate, mono=True)
        sample["X"], _ = audio.crop(file, chunk, duration=duration)
        num_frames=5
        resolution = duration /num_frames
        sample["y"] = file["annotation"].discretize(
            support=chunk, resolution=resolution, duration=duration
        )

        return sample

def preprocess(dataset_part):
          samples=[]
          duration=5.0
          count=0
          while True:
            file, *_ = random.choices(
                dataset_part, weights=[f["_annotated_duration"] for f in dataset_part], k=1,
            )

            segment, *_ = random.choices(
                file["annotated"], weights=[s.duration for s in file["annotated"]], k=1,
            )
            start_time = random.uniform(segment.start, segment.end - duration)
            chunk = Segment(start_time, start_time + duration)

            sample=prepare_chunk(file,chunk)
            samples.append(sample)
            count+=1
          return samples

def collate_fn(batch):
        samples=[]
        for b                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                in batch:
            if b["X"].size(0)==1:
                samples.append(b["X"])
            else:
             
                  print("Erro in tensor size")
        collate_X=default_collate(samples)
        labels = sorted(set(itertools.chain(*(b["y"].labels for b in batch))))

        batch_size, num_frames, num_labels = (
            len(batch),
            len(batch[0]["y"]),
            len(labels),
        )
        Y = np.zeros((batch_size, num_frames, num_labels), dtype=np.int64)

        for i, b in enumerate(batch):
            for local_idx, label in enumerate(b["y"].labels):
                global_idx = labels.index(label)
                Y[i, :, global_idx] = b["y"].data[:, local_idx]
        collate_y=torch.from_numpy(Y)
        batch_size, num_frames, _ = collate_y.shape
        max_num_speakers = torch.max(
            torch.sum(torch.sum(collate_y, dim=1) > 0.0, dim=1)
        )
        indices = torch.argsort(torch.sum(collate_y, dim=1), dim=1, descending=True)

        y = torch.zeros(
            (batch_size, num_frames, max_num_speakers), dtype=collate_y.dtype
        )
        for b, index in enumerate(indices):
            for k, i in zip(range(max_num_speakers), index):
                y[b, :, k] = collate_y[b, :, i.item()]
     
        return {"X": collate_X, "y": y}

