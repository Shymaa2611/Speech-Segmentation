import torch
from dataset import extract_mfcc
import numpy as np




def predict(model,file_path):
    model.load_state_dict(torch.load('classifier_model.pt'))
    model.eval()
    file_path = "D:\\MachineCourse\\MachineLearnig\\NLP\\NLP-main\\testCode\\speech _Recognition\\Audio_Classification\\data\\Forest Recordings\\recording_94.mp3"
    test_mfcc = extract_mfcc(file_path)
    test_mfcc = np.expand_dims(test_mfcc, axis=0)
    test_tensor = torch.tensor(test_mfcc, dtype=torch.float32)
    predicted_class = torch.argmax(model(test_tensor)).item()
    return predicted_class