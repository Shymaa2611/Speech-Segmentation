import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x



"""
model consist of :- 
    SincNet 
       |
     LSTM
       |
    Feed forward
       | 
    Classifier

"""
class segmentationModel(nn.Module):
    def __init__(self):
        super(segmentationModel, self).__init__()
        self.sincnet = nn.Sequential(
            SincNet(1, 32, 251, 160),  
            nn.BatchNorm1d(32),
            nn.MaxPool1d(3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(3),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(3),
            nn.ReLU()
        )
        self.lstm1= nn.LSTM(256, 128, num_layers=3, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm2 = nn.LSTM(256, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.classifier=Classifier(input_size=0)

        

    def forward(self, x):
        x = self.sincnet(x)
        x, _ = self.lstm1(x.permute(0, 2, 1))
        x, _ = self.lstm2(x.permute(0, 2, 1))
        x = x[:, -1, :]
        x =  F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x=self.classifier(x.shape[2])
        return x
