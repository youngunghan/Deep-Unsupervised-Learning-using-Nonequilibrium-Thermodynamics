import torch
import torch.nn as nn
import torchvision.models as models
from .layers import BinaryBottleneck, Bottleneck

class TBH(nn.Module):
    def __init__(self, hash_dim=64, feature_dim=2048, bottleneck_dim=256):
        super(TBH, self).__init__()
        
        # ResNet50에서 마지막 FC layer 제거
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        
        # 추가: Feature Adaptation Layer
        self.feature_adaptation = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Twin bottleneck structure
        self.bottleneck1 = Bottleneck(feature_dim, bottleneck_dim)
        self.binary_bottleneck = BinaryBottleneck(bottleneck_dim, hash_dim)
        self.bottleneck2 = Bottleneck(hash_dim, bottleneck_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.BatchNorm1d(bottleneck_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 3*224*224)
        )
        
        # 추가 필요: training_stage 초기화
        self.training_stage = 1  # 초기 stage 설정
        self.temperature = 0.1   # temperature 추가
        
        # 추가 필요: weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def encode(self, x):
        # Feature extraction
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.feature_adaptation(x)
        
        # Add temperature scaling here too
        if self.training and self.training_stage == 2:
            x = x / self.temperature
        
        # First bottleneck
        z1 = self.bottleneck1(x)
        
        # Binary bottleneck
        b = self.binary_bottleneck(z1)
        
        # Second bottleneck
        z2 = self.bottleneck2(b)
        
        return x, z1, b, z2
    
    def decode(self, z2):
        x_recon = self.decoder(z2)
        x_recon = x_recon.view(-1, 3, 224, 224)
        return x_recon
    
    def forward(self, x):
        x_orig, z1, b, z2 = self.encode(x)
        x_recon = self.decode(z2)
        return x_orig, z1, b, z2, x_recon 