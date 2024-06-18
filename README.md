# VSegmentation
   - detect speech in any audio that contain more speakers and detect who and when   
     speak in audio .


# Dataset
 - AMI Meeting Corpus 
   <br/>○ 100 hours of meeting recordings<br/>
   ○ English <br/>
   ○ Recorded in three different rooms with different acoustic properties<br/>
   ○ Mostly non-native speakers<br/>
   ○ Split into train, test, and development sets<br/>
   ○ Groundtruth: RTTM files (one speech turn per line with start time and duration)<br/>

## PREPROCESSING
 -	Load data using pyannote.database
 -	Read RTTM files 
 -	Split into 293 frames

 ![AMI](AMI.jpg)


 ![Architecture](architecture.jpg)

 ## Research Paper
  - https://arxiv.org/abs/2104.04045

## Usage 
``` python
from pyannote.audio import Pipeline
audio_url="audio_2.wav"
diarization = pipeline(audio_url)
pipeline = Pipeline.from_pretrained("V-Segmentation_checkpoint\\config.yaml")
for turn, _,speaker in  diarization.itertracks(yield_label=True):
  print({"start":turn.start,"end":turn.end})
```
## OUTPUT 
    RTTM 
      SPEAKER ES2011a 1 34.27 10.12 <NA> <NA> FEE041 <NA> <NA>
      SPEAKER ES2011a 1 46.43 10.42 <NA> <NA> FEE041 <NA> <NA>




## Checkpoint 

