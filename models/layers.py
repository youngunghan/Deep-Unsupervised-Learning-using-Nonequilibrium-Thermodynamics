import torch
import torch.nn as nn

class BinaryBottleneck(nn.Module):
    def __init__(self, in_features, hash_dim):
        super(BinaryBottleneck, self).__init__()
        self.fc = nn.Linear(in_features, hash_dim)
        self.bn = nn.BatchNorm1d(hash_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.tanh(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_features, bottleneck_dim):
        super(Bottleneck, self).__init__()
        self.fc = nn.Linear(in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x 