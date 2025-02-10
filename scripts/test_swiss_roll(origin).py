import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import os
import argparse
from typing import List, Tuple, Optional
from networks.dpm import DiffusionModel
from networks.mlp import MLP, MultiscaleConvolution

class SwissRollDataset:
    @staticmethod
    def sample_batch(size: int) -> np.ndarray:
        """Generate Swiss Roll dataset samples"""
        x, _ = make_swiss_roll(size)
        return x[:, [2, 0]] / 10.0 * np.array([1, -1])

    @staticmethod
    def to_image_format(x: torch.Tensor) -> torch.Tensor:
        """Convert (batch_size, 2) to (batch_size, 1, 8, 8) format"""
        # Reshape to (batch_size, 1, 8, 8)
        x = x.unsqueeze(1)  # (batch_size, 1, 2)
        x = x.view(x.shape[0], 1, 2, 1)  # (batch_size, 1, 2, 1)
        x = torch.nn.functional.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)
        return x

    @staticmethod
    def from_image_format(x: torch.Tensor) -> torch.Tensor:
        """Convert (batch_size, 1, 8, 8) back to (batch_size, 2) format"""
        # Downsample back to original size
        x = torch.nn.functional.adaptive_avg_pool2d(x, (2, 1))
        return x.view(x.shape[0], 2)

class SwissRollMLP(MultiscaleConvolution):
    def __init__(self, num_channels=1, hidden_channels=256, n_temporal_basis=10):
        """Modified MultiscaleConvolution for Swiss Roll dataset"""
        super().__init__(
            num_channels=num_channels,
            num_filters=hidden_channels,
            num_scales=3,  # Reduce scales for 8x8 input
            filter_size=3,  # 3x3 convolution
            activation=lambda x: torch.nn.functional.softplus(x - 1)  # Shifted softplus
        )
        
        # Temporal encoding using Gaussian basis functions
        self.n_temporal_basis = n_temporal_basis
        self.register_buffer('temporal_basis', self._get_temporal_basis())
        
        # Final layers for mu and sigma prediction
        self.mu_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, 1)
        )
        self.sigma_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, 1)
        )

    def _get_temporal_basis(self) -> torch.Tensor:
        """Generate Gaussian basis functions for temporal encoding"""
        time = torch.linspace(0, 1, self.n_temporal_basis)
        centers = torch.linspace(0, 1, self.n_temporal_basis)
        width = 1.0 / self.n_temporal_basis
        # Shape: [n_temporal_basis]
        return torch.exp(-0.5 * (time.view(-1, 1) - centers).pow(2) / width ** 2)

    def temporal_encode(self, t: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        """Encode time using Gaussian basis functions"""
        # Normalize time to [0, 1]
        t_norm = (t.float() - 1) / (40 - 1)  # Assuming max timestep is 40
        
        # Get basis function values: [batch_size, n_temporal_basis]
        centers = torch.linspace(0, 1, self.n_temporal_basis, device=t_norm.device)
        width = 1.0 / self.n_temporal_basis
        
        # Calculate encoding for each timestep
        t_norm = t_norm.view(-1, 1)  # [batch_size or 1, 1]
        encoding = torch.exp(-0.5 * (t_norm - centers).pow(2) / width ** 2)  # [batch_size or 1, n_temporal_basis]
        
        # Expand to batch size if needed
        if batch_size is not None and encoding.size(0) == 1:
            encoding = encoding.expand(batch_size, -1)
        
        return encoding

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal encoding
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            t: Time tensor
        Returns:
            Output tensor of shape (batch_size, num_channels * 2)
        """
        # Get temporal encoding: [batch_size, n_temporal_basis]
        t_encoding = self.temporal_encode(t, batch_size=x.shape[0])
        
        # Process through multiscale convolution network
        features = super().forward(x)  # Shape: [batch_size, hidden_channels, H, W]
        
        # Apply temporal modulation (broadcasting over hidden_channels, H, W)
        features = features * t_encoding.mean(dim=1).view(-1, 1, 1, 1)
        
        # Get mu and sigma
        mu = self.mu_head(features)
        log_sigma = self.sigma_head(features)
        
        # Take mean over spatial dimensions and concatenate
        mu = mu.mean(dim=(2, 3))
        log_sigma = log_sigma.mean(dim=(2, 3))
        
        return torch.cat([mu, log_sigma], dim=1)

class SwissRollDiffusion(DiffusionModel):
    def __init__(
        self,
        spatial_width: int = 2,
        n_colors: int = 1,
        n_temporal_basis: int = 10,
        trajectory_length: int = 40,
        beta_start: float = 1e-5,
        beta_end: float = 3e-1,
        mlp_layers: int = 200,
        mlp_hidden_channels: int = 256,
        min_t: int = 2,
        device: torch.device = None
    ):
        super().__init__(
            spatial_width=spatial_width,
            n_colors=n_colors,
            n_temporal_basis=n_temporal_basis,
            trajectory_length=trajectory_length,
            beta_start=beta_start,
            beta_end=beta_end,
            mlp_layers=mlp_layers,
            mlp_hidden_channels=mlp_hidden_channels,
            min_t=min_t,
            device=device
        )
        
        # Replace MLP with Swiss Roll version
        self.mlp = SwissRollMLP(
            num_channels=1,  # 1D data
            hidden_channels=mlp_hidden_channels,
            n_temporal_basis=n_temporal_basis
        ).to(device)

        # Initialize noise schedule
        self._init_noise_schedule(beta_start, beta_end)

    def _init_noise_schedule(self, beta_start: float, beta_end: float):
        """Initialize noise schedule parameters"""
        betas = torch.linspace(-18, 10, self.trajectory_length)
        self.beta = (torch.sigmoid(betas) * (beta_end - beta_start) + beta_start).to(self.device)
        self.alpha = (1. - self.beta).to(self.device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)

    def get_mu_sigma(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and variance for reverse process"""
        # Get MLP output with temporal encoding
        mlp_out = self.mlp(x_t, t)
        
        # Split into mu and sigma
        mu, log_sigma = mlp_out.chunk(2, dim=1)
        sigma = torch.exp(log_sigma)
        
        # Reshape to match input format: (batch_size, 1) -> (batch_size, 1, 8, 8)
        mu = mu.view(x_t.shape[0], 1, 1, 1).expand(-1, -1, 8, 8)
        sigma = sigma.view(x_t.shape[0], 1, 1, 1).expand(-1, -1, 8, 8)
        
        return mu, sigma

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward diffusion process with posterior calculation"""
        t = t - 1  # Start indexing at 0
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        
        # Add noise
        noise = torch.randn_like(x0, device=self.device)
        xt = x0 * torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1) + noise * torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1)
        
        # Calculate posterior parameters
        mu1_scl = torch.sqrt(alpha_bar_t / alpha_t).view(-1, 1, 1, 1)
        mu2_scl = (1. / torch.sqrt(alpha_t)).view(-1, 1, 1, 1)
        cov1 = (1. - alpha_bar_t / alpha_t).view(-1, 1, 1, 1)
        cov2 = (beta_t / alpha_t).view(-1, 1, 1, 1)
        lam = 1. / cov1 + 1. / cov2
        mu = (x0 * mu1_scl / cov1 + xt * mu2_scl / cov2) / lam
        sigma = torch.sqrt(1. / lam)
        
        return mu, sigma, xt

    def train_step(self, x0: torch.Tensor) -> torch.Tensor:
        """Single training step with KL divergence loss"""
        # Convert to image format
        x0_img = SwissRollDataset.to_image_format(x0)
        
        # Sample random timestep
        t = torch.randint(self.min_t, self.trajectory_length + 1, (1,), device=self.device)
        
        # Forward process
        mu_posterior, sigma_posterior, xt = self.forward_process(x0_img, t)
        
        # Get model predictions
        mu, sigma = self.get_mu_sigma(xt, t)
        
        # Calculate KL divergence loss
        KL = (torch.log(sigma) - torch.log(sigma_posterior) + 
              (sigma_posterior ** 2 + (mu_posterior - mu) ** 2) / (2 * sigma ** 2) - 0.5)
        
        return KL.mean()

    def sample(self, size: int) -> torch.Tensor:
        """Generate samples using reverse process"""
        # Initialize with random noise
        x = torch.randn(size, 1, 8, 8).to(self.device)
        
        # Reverse diffusion process
        for t in range(self.trajectory_length):
            t_tensor = torch.tensor([self.trajectory_length - t - 1 + 1], device=self.device)
            
            # Get denoising parameters
            mu, sigma = self.get_mu_sigma(x, t_tensor)
            
            # Add noise scaled by predicted sigma unless it's the last step
            if t < self.trajectory_length - 1:
                noise = torch.randn_like(x)
                x = mu + sigma * noise
            else:
                x = mu
        
        # Convert back to vector format
        return SwissRollDataset.from_image_format(x)

class Visualizer:
    @staticmethod
    def plot(model: SwissRollDiffusion, device: str, save_dir: str):
        """Plot Swiss Roll samples and generated samples"""
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        # Plot original and noisy samples
        x0 = SwissRollDataset.sample_batch(5000)
        x0_tensor = torch.from_numpy(x0).float().to(device)
        x0_img = SwissRollDataset.to_image_format(x0_tensor)
        
        # Forward process
        x20_img = model.forward_process(x0_img, torch.tensor([20], device=device))[-1]
        x40_img = model.forward_process(x0_img, torch.tensor([40], device=device))[-1]
        
        # Convert back to vector format
        x20 = SwissRollDataset.from_image_format(x20_img)
        x40 = SwissRollDataset.from_image_format(x40_img)
        
        data = [x0, x20.detach().cpu().numpy(), x40.detach().cpu().numpy()]
        for i, t in enumerate([0, 20, 39]):
            plt.subplot(2, 3, 1 + i)
            plt.scatter(data[i][:, 0], data[i][:, 1], alpha=.1, s=1)
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.gca().set_aspect('equal')
            if t == 0: plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
            if i == 0: plt.title(r'$t=0$', fontsize=17)
            if i == 1: plt.title(r'$t=\frac{T}{2}$', fontsize=17)
            if i == 2: plt.title(r'$t=T$', fontsize=17)

        # Generate samples
        samples = model.sample(5000)
        for i, t in enumerate([0, 20, 40]):
            plt.subplot(2, 3, 4 + i)
            plt.scatter(
                samples[:, 0].detach().cpu().numpy(),
                samples[:, 1].detach().cpu().numpy(),
                alpha=.1, s=1, c='r'
            )
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.gca().set_aspect('equal')
            if t == 0: plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        
        plt.savefig(os.path.join(save_dir, "diffusion_model.png"), bbox_inches='tight')
        plt.close()

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = SwissRollDiffusion(
        spatial_width=2,
        n_colors=1,
        n_temporal_basis=10,
        trajectory_length=40,
        mlp_hidden_channels=args.hidden_dim,
        device=device
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    training_loss = []
    for _ in tqdm(range(args.epochs)):
        # Get data and convert to tensor
        x0 = torch.from_numpy(SwissRollDataset.sample_batch(args.batch_size)).float().to(device)
        
        # Training step
        loss = model.train_step(x0)
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
    
    # Plot results
    Visualizer.plot(model, device, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train diffusion model on Swiss Roll dataset')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory to save the results')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128_000,
                      help='Batch size')
    
    args = parser.parse_args()
    main(args) 