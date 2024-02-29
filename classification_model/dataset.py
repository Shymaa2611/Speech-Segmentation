import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import *

def extract_mfcc(audio_path, max_pad_len=100):
    audio, sr = librosa.load(audio_path, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    pad_width = max_pad_len - mfccs.shape[1]
    
    # Ensure pad_width is n
    if pad_width < 0:
        mfccs = mfccs[:, :max_pad_len]
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0,pad_width)), mode='constant')
    
    return mfccs

def load_data(data_dir):
    labels = []
    mfccs = []
    max_pad_len = 0
    
    for label, folder in enumerate(os.listdir(data_dir)):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if filename.endswith(".wav") or filename.endswith(".mp3") :
                audio_path = os.path.join(data_dir, folder, filename)
                mfcc = extract_mfcc(audio_path)
                mfccs.append(mfcc)
                labels.append(label)
                
                if mfcc.shape[1] > max_pad_len:
                    max_pad_len = mfcc.shape[1]
    
    return np.array(mfccs), np.array(labels)

def convert_dataset_tensor(x_train,x_test,y_train,y_test):
    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor

def split_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)
    return X_train,X_test,y_train,y_test

def prepare_dataset(data_dir):
  X,y=load_data(data_dir)
  x_train,x_test,y_train,y_test=split_dataset(X,y)
  X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor=convert_dataset_tensor(x_train,x_test,y_train,y_test)
  num_classes=len(np.unique(y))
  return X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor,x_train,y_train,y_test,num_classes


