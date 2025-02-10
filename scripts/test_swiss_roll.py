import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from typing import List, Tuple, Optional
from dataclasses import dataclass
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import wasserstein_distance


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion model."""
    exp_name: str
    num_timesteps: int
    hidden_dim: int
    data_dim: int
    device: str
    beta_min: float
    beta_max: float
    beta_schedule_limits: Tuple[float, float]
    learning_rate: float
    batch_size: int
    num_epochs: int
    save_dir: str
    log_dir: str
    eval_interval: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'DiffusionConfig':
        """Create DiffusionConfig from command line arguments."""
        return cls(
            exp_name=args.exp_name,
            num_timesteps=args.num_timesteps,
            hidden_dim=args.hidden_dim,
            data_dim=args.data_dim,
            device=args.device,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            beta_schedule_limits=(args.beta_schedule_min, args.beta_schedule_max),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            eval_interval=args.eval_interval
        )


def create_swiss_roll_dataset(num_samples: int) -> np.ndarray:
    """Create Swiss Roll dataset samples."""
    data_points, _ = make_swiss_roll(num_samples)
    return data_points[:, [2, 0]] / 10.0 * np.array([1, -1])


class DiffusionEncoder(nn.Module):
    """Encoder network for diffusion model."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        
        self.encoder_head = nn.Sequential(
            nn.Linear(config.data_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        self.time_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.data_dim * 2)
            ) for _ in range(config.num_timesteps)
        ])

    def forward(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        """Forward pass through the encoder."""
        hidden = self.encoder_head(x)
        return self.time_encoders[timestep](hidden)


class DiffusionModel(nn.Module):
    """Diffusion model implementing forward and reverse processes."""

    def __init__(self, encoder: nn.Module, config: DiffusionConfig):
        super().__init__()
        
        self.encoder = encoder
        self.device = config.device
        self.num_timesteps = config.num_timesteps
        
        # Setup noise schedule
        beta_schedule = torch.linspace(config.beta_schedule_limits[0], 
                                     config.beta_schedule_limits[1], 
                                     config.num_timesteps)
        self.beta = torch.sigmoid(beta_schedule) * (config.beta_max - config.beta_min) + config.beta_min
        
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def forward_diffusion(self, x0: torch.Tensor, timestep: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward diffusion process."""
        timestep = timestep - 1  # Convert to 0-based indexing
        beta_t = self.beta[timestep]
        alpha_t = self.alpha[timestep]
        alpha_bar_t = self.alpha_bar[timestep]
        
        # Add noise
        noise = torch.randn_like(x0)
        xt = x0 * torch.sqrt(alpha_bar_t) + noise * torch.sqrt(1. - alpha_bar_t)
        
        # Calculate posterior parameters
        mu1_scale = torch.sqrt(alpha_bar_t / alpha_t)
        mu2_scale = 1. / torch.sqrt(alpha_t)
        var1 = 1. - alpha_bar_t / alpha_t
        var2 = beta_t / alpha_t
        precision = 1. / var1 + 1. / var2
        mu = (x0 * mu1_scale / var1 + xt * mu2_scale / var2) / precision
        sigma = torch.sqrt(1. / precision)
        
        return mu, sigma, xt

    def reverse_diffusion(self, xt: torch.Tensor, timestep: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Reverse diffusion process."""
        timestep = timestep - 1
        if timestep == 0:
            return None, None, xt
            
        mu, h = self.encoder(xt, timestep).chunk(2, dim=1)
        sigma = torch.sqrt(torch.exp(h))
        noise = torch.randn_like(xt)
        samples = mu + noise * sigma
        
        return mu, sigma, samples

    def sample(self, num_samples: int) -> List[torch.Tensor]:
        """Generate samples using the diffusion model."""
        noise = torch.randn((num_samples, 2)).to(self.device)
        samples = [noise]
        
        for t in range(self.num_timesteps):
            _, _, x = self.reverse_diffusion(samples[-1], self.num_timesteps - t - 1 + 1)
            samples.append(x)
            
        return samples


def calculate_mmd_distance(x: np.ndarray, y: np.ndarray, kernel_width: float = 0.1) -> float:
    """Calculate Maximum Mean Discrepancy between two samples."""
    def gaussian_kernel(x1, x2, width):
        return np.exp(-np.sum((x1[:, None] - x2[None, :]) ** 2, axis=-1) / (2 * width ** 2))
    
    nx, ny = x.shape[0], y.shape[0]
    kxx = gaussian_kernel(x, x, kernel_width)
    kyy = gaussian_kernel(y, y, kernel_width)
    kxy = gaussian_kernel(x, y, kernel_width)
    
    return (np.sum(kxx) - np.trace(kxx)) / (nx * (nx - 1)) + \
           (np.sum(kyy) - np.trace(kyy)) / (ny * (ny - 1)) - \
           2 * np.mean(kxy)


def calculate_manifold_coverage(real_data: np.ndarray, generated_data: np.ndarray, distance_threshold: float = 0.1) -> float:
    """Calculate coverage of the real data manifold by generated samples."""
    from sklearn.neighbors import NearestNeighbors
    neighbor_finder = NearestNeighbors(n_neighbors=1).fit(generated_data)
    distances, _ = neighbor_finder.kneighbors(real_data)
    return np.mean(distances.ravel() < distance_threshold)


def evaluate_model_metrics(model: DiffusionModel, num_eval_samples: int = 5000) -> dict:
    """Evaluate model using various metrics."""
    with torch.no_grad():
        samples = model.sample(num_eval_samples)
        generated_data = samples[-1].detach().cpu().numpy()
    
    real_data = create_swiss_roll_dataset(num_eval_samples)
    
    metrics = {
        'mmd_score': calculate_mmd_distance(real_data, generated_data),
        'manifold_coverage': calculate_manifold_coverage(real_data, generated_data),
        'wasserstein_distance': wasserstein_distance(
            real_data.reshape(-1), generated_data.reshape(-1)
        )
    }
    
    return metrics


def visualize_diffusion_steps(model: DiffusionModel, save_dir: str, writer: Optional[SummaryWriter] = None, step: int = 0):
    """Visualize diffusion process steps."""
    plt.figure(figsize=(10, 6))
    
    # Visualize forward process
    real_samples = create_swiss_roll_dataset(5000)
    mid_samples = model.forward_diffusion(torch.from_numpy(real_samples).to(model.device), 20)[-1].detach().cpu().numpy()
    noise_samples = model.forward_diffusion(torch.from_numpy(real_samples).to(model.device), 40)[-1].detach().cpu().numpy()
    process_samples = [real_samples, mid_samples, noise_samples]
    
    for i, t in enumerate([0, 20, 39]):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(process_samples[i][:, 0], process_samples[i][:, 1], alpha=.1, s=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0:
            plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        plt.title(f'$t={t if t != 39 else "T"}$' if t != 20 else r'$t=\frac{T}{2}$', fontsize=17)

    # Visualize reverse process
    with torch.no_grad():
        generated_samples = model.sample(5000)
    
    for i, t in enumerate([0, 20, 40]):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(generated_samples[40 - t][:, 0].detach().cpu().numpy(),
                   generated_samples[40 - t][:, 1].detach().cpu().numpy(),
                   alpha=.1, s=1, c='r')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0:
            plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
    
    # Save visualization
    save_path = os.path.join(save_dir, f"diffusion_process_{step}.png")
    plt.savefig(save_path, bbox_inches='tight')
    
    if writer is not None:
        writer.add_figure('Diffusion Process', plt.gcf(), step)
    
    plt.close()


def train_diffusion_model(
    model: DiffusionModel,
    optimizer: torch.optim.Optimizer,
    config: DiffusionConfig,
    writer: SummaryWriter,
) -> List[float]:
    """Train diffusion model."""
    loss_history = []
    best_mmd = float('inf')
    
    for epoch in tqdm(range(config.num_epochs)):
        # Training step
        batch_data = torch.from_numpy(create_swiss_roll_dataset(config.batch_size)).float().to(config.device)
        timestep = np.random.randint(2, config.num_timesteps + 1)
        
        mu_posterior, sigma_posterior, noisy_samples = model.forward_diffusion(batch_data, timestep)
        mu_predicted, sigma_predicted, _ = model.reverse_diffusion(noisy_samples, timestep)

        # Calculate loss
        kl_divergence = (torch.log(sigma_predicted) - torch.log(sigma_posterior) +
                        (sigma_posterior ** 2 + (mu_posterior - mu_predicted) ** 2) / 
                        (2 * sigma_predicted ** 2) - 0.5)
        loss = kl_divergence.mean()

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        writer.add_scalar('train/loss', loss.item(), epoch)
        loss_history.append(loss.item())
        
        # Evaluation
        if epoch % config.eval_interval == 0:
            metrics = evaluate_model_metrics(model)
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f'metrics/{metric_name}', metric_value, epoch)
            
            # Save best model
            if metrics['mmd_score'] < best_mmd:
                best_mmd = metrics['mmd_score']
                save_model_checkpoint(model, optimizer, config, metrics, epoch, is_best=True)
            
            # Visualization
            visualize_diffusion_steps(model, config.save_dir, writer, epoch)
        
    return loss_history


def save_model_checkpoint(
    model: DiffusionModel,
    optimizer: torch.optim.Optimizer,
    config: DiffusionConfig,
    metrics: dict,
    epoch: int,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    filename = f'{config.exp_name}_best.pth' if is_best else f'{config.exp_name}_final.pth'
    save_path = os.path.join(config.save_dir, filename)
    torch.save(checkpoint, save_path)


def parse_training_args() -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description='Train diffusion model on Swiss Roll dataset')
    
    # Experiment name
    parser.add_argument('--exp_name', type=str, default='swiss_roll_diffusion',
                      help='Experiment name')
    
    # Model architecture
    parser.add_argument('--num_timesteps', type=int, default=40,
                      help='Number of diffusion timesteps')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension size')
    parser.add_argument('--data_dim', type=int, default=2,
                      help='Data dimension')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    # Noise schedule
    parser.add_argument('--beta_min', type=float, default=1e-5,
                      help='Minimum beta value')
    parser.add_argument('--beta_max', type=float, default=3e-1,
                      help='Maximum beta value')
    parser.add_argument('--beta_schedule_min', type=float, default=-18,
                      help='Minimum value for beta schedule')
    parser.add_argument('--beta_schedule_max', type=float, default=10,
                      help='Maximum value for beta schedule')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128_000,
                      help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300_000,
                      help='Number of epochs')
    
    # Save directory
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save results')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for tensorboard logs')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=3000,
                      help='Number of epochs between evaluations')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_training_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"Results will be saved to: {args.save_dir}")
    print(f"Logs will be saved to: {args.log_dir}")
    
    # Initialize tensorboard writer
    writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name))
    
    # Create model and optimizer
    config = DiffusionConfig.from_args(args)
    encoder = DiffusionEncoder(config).to(config.device)
    model = DiffusionModel(encoder, config)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    
    # Log configuration
    writer.add_text('Model/Architecture', str(model))
    writer.add_text('Model/Config', str(config))
    
    # Training
    loss_history = train_diffusion_model(model, optimizer, config, writer)
    
    # Final evaluation
    final_metrics = evaluate_model_metrics(model)
    for metric_name, metric_value in final_metrics.items():
        writer.add_scalar(f'metrics/{metric_name}_final', metric_value, 0)
    
    print("\nFinal Metrics:")
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Save final model
    save_model_checkpoint(model, optimizer, config, final_metrics, config.num_epochs, is_best=False)
    
    writer.close()
    print(f"\nTraining completed. Results saved to: {config.save_dir}")
    print(f"View training progress with: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()