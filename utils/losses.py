import torch
import torch.nn as nn
import torch.nn.functional as F

class TBHLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super(TBHLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.training_stage = 1
        
    def forward(self, x_orig, z1, b, z2, x_recon, x_input):
        # Reconstruction loss - 논문에서는 L2 normalization 후 MSE 사용
        x_input_norm = F.normalize(x_input.view(x_input.size(0), -1), p=2, dim=1)
        x_recon_norm = F.normalize(x_recon.view(x_recon.size(0), -1), p=2, dim=1)
        
        # Feature preservation loss - 논문에서는 normalized cosine similarity 사용
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        
        # Stage에 따라 다른 loss 계산
        if self.training_stage == 1:
            # Stage 1: Reconstruction + Quantization
            recon_loss = F.mse_loss(x_recon_norm, x_input_norm)
            quant_loss = F.mse_loss(torch.abs(b), torch.ones_like(b))
            total_loss = self.alpha * recon_loss + self.gamma * quant_loss
            return total_loss, recon_loss, 0, quant_loss
        else:
            # Stage 2: Feature Preservation + Quantization
            temperature = 0.1
            feat_loss = 1 - F.cosine_similarity(z2_norm/temperature, 
                                              z1_norm/temperature).mean()
            quant_loss = F.mse_loss(torch.abs(b), torch.ones_like(b))
            total_loss = self.beta * feat_loss + self.gamma * quant_loss
            return total_loss, 0, feat_loss, quant_loss 