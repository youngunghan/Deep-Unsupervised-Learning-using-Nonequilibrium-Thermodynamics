import torch
import torch.nn as nn
from .mlp import MLP
from typing import Optional, Tuple

class DiffusionModel(nn.Module):
    """Diffusion Probabilistic Model from 'Deep Unsupervised Learning using Nonequilibrium Thermodynamics'
    
    Learns to reverse a forward diffusion process that gradually adds noise to data.
    Uses neural network to predict reverse process parameters μ_θ(x^(t),t) and σ_θ²(x^(t),t).
    
    Args:
        spatial_width (int): Input image width/height
        n_colors (int): Number of image channels  
        n_temporal_basis (int): Number of Gaussian basis functions for time encoding
        trajectory_length (int): Total diffusion steps T
        beta_start (float): Initial noise level β₁
        beta_end (float): Final noise level βₜ
        mlp_layers (int): Number of network layers
        mlp_hidden_channels (int): Hidden dimension size
        min_t (int): Minimum timestep for sampling
        eps (float): Small constant for numerical stability
        device (torch.device): Device to run model on
    
    Example:
        >>> model = DiffusionModel(spatial_width=28, n_colors=1, trajectory_length=1000)
        >>> loss = model.cost_single_t(x_batch)  # Training
        >>> samples = model.sample(16)  # Generation
    """
    def __init__(
        self,
        spatial_width: int,
        n_colors: int,
        n_temporal_basis: int = 10,
        trajectory_length: int = 2000,
        beta_start: float = 0.01,
        beta_end: float = 0.05,
        mlp_layers: int = 200,
        mlp_hidden_channels: int = 128,
        min_t: int = 100,
        eps: float = 1e-5,
        device: torch.device = None
    ):
        super(DiffusionModel, self).__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store configuration
        self.spatial_width = spatial_width
        self.n_colors = n_colors
        self.n_temporal_basis = n_temporal_basis
        self.trajectory_length = trajectory_length
        self.min_t = min_t
        self.eps = eps
        
        # Initialize components
        self._initialize_temporal_basis()
        self._initialize_diffusion_params(beta_start, beta_end)
        self.mlp = self._build_mlp(mlp_layers, mlp_hidden_channels)

    def _initialize_temporal_basis(self):
        """Initialize Gaussian temporal basis functions for time-dependent modeling (Section 2.2)
        
        Creates normalized basis functions: g_j(t) = exp(-1/(2w²)(t-τ_j)²) / ∑_k exp(-1/(2w²)(t-τ_k)²)
        
        Args:
            None
            
        Returns:
            None
            
        Example:
            >>> model._initialize_temporal_basis()  # Called automatically in __init__
            >>> model.temporal_basis.shape  # torch.Size([n_temporal_basis, trajectory_length])
        """
        self.register_buffer(
            'temporal_basis',
            self.generate_temporal_basis(self.trajectory_length, self.n_temporal_basis)
        )

    def _initialize_diffusion_params(self, beta_start: float, beta_end: float):
        """Setup noise schedule and derived parameters for forward diffusion process (Section 2.1)
        
        Args:
            beta_start (float): Initial noise level β₁
            beta_end (float): Final noise level βₜ
            
        Returns:
            None
            
        Example:
            >>> model._initialize_diffusion_params(0.0001, 0.02)  # Called automatically in __init__
            >>> model.beta.shape  # torch.Size([trajectory_length])
        """
        # Linear noise schedule: β_t from beta_start to beta_end
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, self.trajectory_length))
        
        # Derived parameters: α_t = 1 - β_t, ᾱ_t = ∏α_s
        self.register_buffer('alpha', 1.0 - self.beta)
        self.register_buffer('alpha_cum', torch.cumprod(self.alpha, dim=0))

    def _build_mlp(self, num_layers: int, hidden_channels: int) -> nn.Module:
        """Build neural network for predicting reverse process parameters (Section 2.2)
        
        Args:
            num_layers (int): Number of network layers
            hidden_channels (int): Hidden dimension size
            
        Returns:
            nn.Module: MLP that outputs coefficients for temporal basis functions
            
        Example:
            >>> mlp = model._build_mlp(200, 128)  # Called automatically in __init__
            >>> x = torch.randn(16, 1, 28, 28)
            >>> output = mlp(x)  # Shape: [16, 2*n_colors*n_temporal_basis, 28, 28]
        """
        return MLP(
            num_channels=self.n_colors,
            num_layers=num_layers,
            num_output_channels=2 * self.n_colors * self.n_temporal_basis,
            hidden_channels=hidden_channels
        )

    def generate_temporal_basis(
        self,
        trajectory_length: int,
        n_basis: int
    ) -> torch.Tensor:
        """Generate Gaussian temporal basis functions for time-dependent modeling (Appendix D.2.1)
        
        Args:
            trajectory_length (int): Total timesteps T
            n_basis (int): Number of basis functions
            
        Returns:
            torch.Tensor: Normalized basis functions [n_basis, trajectory_length]
            
        Example:
            >>> basis = model.generate_temporal_basis(1000, 10)
            >>> basis.shape  # torch.Size([10, 1000])
        """
        temporal_basis = torch.zeros((n_basis, trajectory_length), device=self.device)
        
        # Time points with log spacing
        t = torch.linspace(0, 1, trajectory_length, device=self.device)
        log_t = torch.log(t + self.eps)
        
        # Basis centers and widths
        centers = torch.exp(torch.linspace(log_t[0], log_t[-1], n_basis, device=self.device))
        widths = self._calculate_adaptive_widths(centers)
        
        # Generate Gaussian basis functions
        for i in range(n_basis):
            temporal_basis[i] = torch.exp(-(t - centers[i])**2 / (2 * widths[i]**2))
        
        # Normalize to sum to 1
        normalizer = temporal_basis.sum(dim=0, keepdim=True)
        return temporal_basis / (normalizer + self.eps)

    def _calculate_adaptive_widths(self, centers: torch.Tensor) -> torch.Tensor:
        """Calculate adaptive widths based on center spacing for smooth basis overlap
        
        Args:
            centers (torch.Tensor): Basis function center positions
            
        Returns:
            torch.Tensor: Width for each basis function
            
        Example:
            >>> centers = torch.tensor([0.1, 0.3, 0.7, 0.9])
            >>> widths = model._calculate_adaptive_widths(centers)
            >>> widths.shape  # torch.Size([4])
        """
        widths = torch.zeros_like(centers)
        
        # First and last centers
        widths[0] = centers[1] - centers[0]
        widths[-1] = centers[-1] - centers[-2]
        
        # Middle centers
        widths[1:-1] = (centers[2:] - centers[:-2]) / 2
        
        return widths

    def forward_process(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to clean images using closed-form forward diffusion (Section 2.1)
        
        Implements: x^(t) = √(ᾱ_t)x^(0) + √(1-ᾱ_t)ε where ε ~ N(0,I)
        
        Args:
            x (torch.Tensor): Clean images [batch_size, channels, height, width]
            t (Optional[torch.Tensor]): Timestep (random if None)
            
        Returns:
            Tuple: (noisy_images, noise, timestep, alpha_cumulative)
            
        Example:
            >>> x = torch.randn(32, 1, 28, 28)
            >>> noisy_x, noise, t, alpha = model.forward_process(x)
        """
        # Sample random timestep if not provided
        if t is None:
            t = torch.randint(self.min_t, self.trajectory_length, (1,), device=self.device)
        t = t.long()
        
        # Apply forward diffusion: x^(t) = √(ᾱ_t)x^(0) + √(1-ᾱ_t)ε
        noise = torch.randn_like(x, device=self.device)
        alpha_cum_t = self.alpha_cum[t].view(-1, 1, 1, 1)
        noisy_x = torch.sqrt(alpha_cum_t) * x + torch.sqrt(1 - alpha_cum_t) * noise
        
        return noisy_x, noise, t, alpha_cum_t

    def get_mu_sigma(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance for reverse diffusion step p(x^(t-1)|x^(t)) (Section 2.2)
        
        Args:
            x_t (torch.Tensor): Noisy data at timestep t
            t (torch.Tensor): Current timestep
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted mean μ and std dev σ
            
        Example:
            >>> x_t = torch.randn(16, 1, 28, 28)
            >>> t = torch.tensor([100])
            >>> mu, sigma = model.get_mu_sigma(x_t, t)
        """
        t = t.long()
        batch_size = x_t.shape[0]
        H, W = x_t.shape[2:]
        
        # Get network output and reshape
        mlp_out = self.mlp(x_t).view(
            batch_size, 2, self.n_colors,
            self.n_temporal_basis, H, W
        )
        
        # Project temporal basis for current timestep
        t_weights = self.get_t_weights(t)
        temporal_projection = torch.matmul(self.temporal_basis, t_weights)
        
        # Calculate parameters
        mu, sigma = self._calculate_mu_sigma(
            mlp_out, temporal_projection, t
        )
        
        return mu, sigma

    def _calculate_mu_sigma(
        self,
        mlp_out: torch.Tensor,
        temporal_projection: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate mean and variance using temporal basis projection and noise schedule
        
        Args:
            mlp_out (torch.Tensor): Network output coefficients [batch, 2, channels, basis, H, W]
            temporal_projection (torch.Tensor): Temporal weights for current timestep
            t (torch.Tensor): Current timestep
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted mean μ and variance σ
            
        Example:
            >>> mlp_out = torch.randn(8, 2, 1, 10, 28, 28)
            >>> temporal_proj = torch.randn(10, 1)
            >>> t = torch.tensor([100])
            >>> mu, sigma = model._calculate_mu_sigma(mlp_out, temporal_proj, t)
        """
        # Extract mean and variance coefficients
        mu_coeffs = mlp_out[:, 0]
        sigma_coeffs = mlp_out[:, 1]
        
        # Project with temporal basis
        mu = torch.sum(
            mu_coeffs * temporal_projection.view(1, 1, -1, 1, 1),
            dim=2
        )
        log_sigma = torch.sum(
            sigma_coeffs * temporal_projection.view(1, 1, -1, 1, 1),
            dim=2
        )
        
        # Apply noise schedule scaling
        alpha_t = self.alpha[t]
        beta_t = self.beta[t]
        
        mu = mu * torch.sqrt(1 / alpha_t).view(-1, 1, 1, 1)
        sigma = torch.exp(log_sigma).clamp(
            min=beta_t.sqrt().item(),
            max=1.0
        )
        
        return mu, sigma

    def cost_single_t(self, x: torch.Tensor) -> torch.Tensor:
        """Compute training loss using variational lower bound (Section 2.3)
        
        Args:
            x (torch.Tensor): Clean training images
            
        Returns:
            torch.Tensor: Negative log-likelihood loss
            
        Example:
            >>> x = torch.randn(32, 1, 28, 28)
            >>> loss = model.cost_single_t(x)
            >>> loss.backward()
        """
        # Add noise to clean images
        X_noisy, _, t, _ = self.forward_process(x)
        
        # Predict reverse parameters
        mu, sigma = self.get_mu_sigma(X_noisy, t)
        
        # Compute negative log likelihood
        return -torch.distributions.Normal(mu, sigma).log_prob(x).mean()

    def get_t_weights(self, t):
        """Create one-hot temporal weight vector for specific timestep
        
        Args:
            t (torch.Tensor): Target timestep
            
        Returns:
            torch.Tensor: One-hot weight vector [trajectory_length, 1]
            
        Example:
            >>> weights = model.get_t_weights(torch.tensor([100]))
            >>> weights[100]  # tensor([1.0])
        """
        t = t.long()
        weights = torch.zeros(self.trajectory_length, device=self.device)
        weights[t] = 1.0
        return weights.unsqueeze(-1)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Generate new samples using reverse diffusion process (Appendix A, Algorithm 1)
        
        Args:
            batch_size (int): Number of samples to generate
            
        Returns:
            torch.Tensor: Generated images
            
        Example:
            >>> samples = model.sample(16)  # Generate 16 images
            >>> samples.shape  # [16, 1, 28, 28]
        """
        # Start from random noise x^(T) ~ N(0,I)
        x = torch.randn(batch_size, self.n_colors, self.spatial_width, self.spatial_width, device=self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.min_t, self.trajectory_length)):
            t_tensor = torch.tensor([t], device=self.device)
            
            # Get reverse parameters
            mu, sigma = self.get_mu_sigma(x, t_tensor)
            
            # Sample x^(t-1)
            if t > 0:
                noise = torch.randn_like(x)
                x = mu + sigma * noise
            else:
                x = mu
        
        return x
