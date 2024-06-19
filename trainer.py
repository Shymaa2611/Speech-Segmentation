import os
from data.process import prepare_data,preprocess,collate_fn
from pyannote.database import FileFinder, get_protocol
from pytorch_lightning import Trainer
from model.model import VSegmentationModel
import torch

os.environ["PYANNOTE_DATABASE_CONFIG"] = "database.yml"

BATCH_SIZE = 128
protocol = get_protocol(
    "AMI.SpeakerDiarization.AMI", preprocessors={"audio": FileFinder()}
     )
duration=5.0
train,validation_sample=prepare_data(protocol,duration)
train_data=preprocess(train)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=32,collate_fn=collate_fn)
val_loader=torch.utils.data.DataLoader(validation_sample,batch_size=16,collate_fn=collate_fn) 
 


if __name__=="__main__":
    trainer = Trainer()
    model= VSegmentationModel(batch_size=BATCH_SIZE)
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)

