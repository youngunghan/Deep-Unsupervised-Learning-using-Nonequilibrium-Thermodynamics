import torch
import torch.nn as nn
from .mlp import MLP
from typing import Optional, Tuple

class DiffusionModel(nn.Module):
    """
    Args:
        spatial_width (int): Width/height of input images (e.g., 28 for MNIST)
        n_colors (int): Number of image channels (1 for grayscale, 3 for RGB)
        n_temporal_basis (int, optional): Number of Gaussian basis functions for time encoding. Default: 10
                                        More basis functions allow finer temporal control but increase complexity
        trajectory_length (int, optional): Total number of diffusion timesteps T. Default: 2000
                                         Longer trajectories give better quality but slower sampling
        beta_start (float, optional): Initial noise level β₁. Default: 0.01
                                    Controls how quickly noise is added at the start
        beta_end (float, optional): Final noise level βₜ. Default: 0.05
                                  Controls maximum amount of noise added
        mlp_layers (int, optional): Number of layers in denoising network. Default: 200
                                  Deeper networks can model more complex transformations
        mlp_hidden_channels (int, optional): Hidden dimension size. Default: 128
                                           Larger values increase model capacity
        min_t (int, optional): Minimum timestep for sampling. Default: 100
                              Can be increased to trade quality for speed
        device (torch.device, optional): Device to run model on. Default: None (auto-detect)

    Shape:
        - Input: (batch_size, n_colors, spatial_width, spatial_width)
        - Output: Depends on method called (see individual method documentation)

    Examples:
        >>> # Train on MNIST
        >>> model = DiffusionModel(spatial_width=28, n_colors=1)
        >>> x = torch.randn(32, 1, 28, 28)  # Batch of MNIST images
        >>> loss = model.cost_single_t(x)    # Training loss
        
        >>> # Generate samples
        >>> samples = model.sample(16)  # Generate 16 new images
        >>> samples.shape  # torch.Size([16, 1, 28, 28])
        
        >>> # Train on CIFAR-10
        >>> model = DiffusionModel(spatial_width=32, n_colors=3)
        >>> x = torch.randn(64, 3, 32, 32)  # Batch of CIFAR images
        >>> loss = model.cost_single_t(x)    # Training loss
    """
    def __init__(
        self,
        spatial_width: int,        # Width/height of input images
        n_colors: int,            # Number of image channels
        n_temporal_basis: int = 10,    # Number of basis functions for time encoding
        trajectory_length: int = 2000,  # Total number of diffusion steps T
        beta_start: float = 0.01,      # Initial noise level β₁
        beta_end: float = 0.05,        # Final noise level βₜ
        mlp_layers: int = 200,         # Number of layers in denoising network
        mlp_hidden_channels: int = 128, # Hidden dimension in denoising network
        min_t: int = 100,              # Minimum timestep for sampling
        eps: float = 1e-5,             # Small constant for numerical stability
        device: torch.device = None     # Device to run the model on
    ):
        super(DiffusionModel, self).__init__()
        
        # Device setup for model placement
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core model configuration
        self.spatial_width = spatial_width
        self.n_colors = n_colors
        self.n_temporal_basis = n_temporal_basis
        self.trajectory_length = trajectory_length  # T in the paper
        self.min_t = min_t
        self.eps = eps
        
        # Initialize model components
        self._initialize_temporal_basis()  # Implements time encoding from Section 2.2
        self._initialize_diffusion_params(beta_start, beta_end)  # Implements noise schedule from Section 2.1
        self.mlp = self._build_mlp(mlp_layers, mlp_hidden_channels)  # Neural network from Section 2.2

    def _initialize_temporal_basis(self):
        """Creates Gaussian basis functions for encoding timesteps (Section 2.2)
        
        The paper uses a set of Gaussian functions to encode time information, which is crucial for:
        1. Allowing the model to learn different behaviors for different noise levels
        2. Providing smooth interpolation between timesteps
        3. Enabling the reverse process to adapt its denoising strength
        
        The basis functions are evenly spaced in log-time to handle the exponential
        nature of the diffusion process. This spacing ensures better resolution
        for early timesteps where changes happen quickly.
        
        Example:
            >>> model = DiffusionModel(28, 1)
            >>> basis = model.temporal_basis  # Shape: [n_temporal_basis, trajectory_length]
            >>> basis.shape  # e.g., torch.Size([10, 2000])
        """
        self.register_buffer(
            'temporal_basis',
            self.generate_temporal_basis(self.trajectory_length, self.n_temporal_basis)
        )

    def _initialize_diffusion_params(self, beta_start: float, beta_end: float):
        """Initialize noise schedule parameters for the forward diffusion process (Section 2.1)
        
        Implements the forward process q(x_t|x_0) from the paper with:
        1. β_t: Linear schedule from beta_start to beta_end
           - Controls the rate of noise addition
           - Linear schedule found to work well empirically
        
        2. α_t = 1 - β_t: Signal scaling factor
           - Represents how much of the original signal remains
           - Decreases monotonically with t
        
        3. ᾱ_t = ∏α_s: Cumulative product for multi-step diffusion
           - Allows direct sampling of x_t given x_0
           - Used in the reparameterization trick for training
        
        Example:
            >>> model = DiffusionModel(28, 1, trajectory_length=1000)
            >>> model.beta[0]      # Initial noise level (β₁)
            tensor(0.0100)
            >>> model.beta[-1]     # Final noise level (βₜ)
            tensor(0.0500)
            >>> model.alpha[0]     # Initial signal scaling (α₁)
            tensor(0.9900)
            >>> model.alpha_cum[0] # Initial cumulative scaling (ᾱ₁)
            tensor(0.9900)
        """
        # Create noise schedule from beta_start to beta_end (Eq. 2)
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, self.trajectory_length))
        
        # Calculate signal scaling factors
        self.register_buffer('alpha', 1.0 - self.beta)  # α_t in paper
        self.register_buffer('alpha_cum', torch.cumprod(self.alpha, dim=0))  # ᾱ_t in paper

    def _build_mlp(self, num_layers: int, hidden_channels: int) -> nn.Module:
        """Build MLP for estimating mean and variance"""
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
        """Generate temporal basis functions"""
        # Initialize basis matrix
        temporal_basis = torch.zeros((n_basis, trajectory_length), device=self.device)
        
        # Generate time points
        t = torch.linspace(0, 1, trajectory_length, device=self.device)
        log_t = torch.log(t + self.eps)
        
        # Calculate centers and widths
        centers = torch.exp(torch.linspace(log_t[0], log_t[-1], n_basis, device=self.device))
        widths = self._calculate_adaptive_widths(centers)
        
        # Generate basis functions
        for i in range(n_basis):
            temporal_basis[i] = torch.exp(-(t - centers[i])**2 / (2 * widths[i]**2))
        
        # Normalize
        normalizer = temporal_basis.sum(dim=0, keepdim=True)
        return temporal_basis / (normalizer + self.eps)

    def _calculate_adaptive_widths(self, centers: torch.Tensor) -> torch.Tensor:
        """Calculate adaptive widths for temporal basis functions"""
        widths = torch.zeros_like(centers)
        
        # First and last centers
        widths[0] = centers[1] - centers[0]
        widths[-1] = centers[-1] - centers[-2]
        
        # Middle centers
        widths[1:-1] = (centers[2:] - centers[:-2]) / 2
        
        return widths

    def forward_process(
        self,
        x: torch.Tensor,          # Input images [batch_size, channels, height, width]
        t: Optional[torch.Tensor] = None  # Optional timestep
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements the forward diffusion process q(x_t|x_0) from Section 2.1
        
        Key equation (Eq. 2): x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε where ε ~ N(0,I)
        
        This process gradually adds noise to the data according to the variance schedule:
        1. The √(ᾱ_t) term scales down the original signal
        2. The √(1-ᾱ_t) term scales the noise
        3. As t increases, more of the original signal is replaced with noise
        4. At t=T, the distribution is close to N(0,I)
        
        Args:
            x: Input images x_0 from the data distribution
            t: Optional timestep (randomly sampled if None)
            
        Returns:
            noisy_x: x_t, the noisy version of input
            noise: ε, the random noise added
            t: The timestep used
            alpha_cum_t: ᾱ_t, the cumulative alpha at timestep t
            
        Example:
            >>> model = DiffusionModel(28, 1)
            >>> x = torch.randn(32, 1, 28, 28)  # Original images
            >>> t = torch.tensor([500])         # Timestep
            >>> noisy_x, noise, _, alpha = model.forward_process(x, t)
            >>> noisy_x.shape  # torch.Size([32, 1, 28, 28])
            >>> alpha         # Shows how much signal remains at t=500
        """
        # Sample random timestep if not provided
        if t is None:
            t = torch.randint(self.min_t, self.trajectory_length, (1,), device=self.device)
        t = t.long()
        
        # Generate Gaussian noise and scale both signal and noise components
        noise = torch.randn_like(x, device=self.device)
        alpha_cum_t = self.alpha_cum[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
        noisy_x = torch.sqrt(alpha_cum_t) * x + torch.sqrt(1 - alpha_cum_t) * noise
        
        return noisy_x, noise, t, alpha_cum_t

    def get_mu_sigma(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implements the reverse process p(x_{t-1}|x_t) from Section 2.2
        
        Predicts the parameters of the reverse diffusion distribution:
        μ_θ(x_t,t): The mean of x_{t-1} given x_t
        σ_θ(x_t,t): The standard deviation
        
        The neural network learns to predict these parameters to gradually
        denoise the data. The process:
        1. Encodes the current noisy image x_t
        2. Uses temporal encoding to handle different noise levels
        3. Predicts mean and variance for the denoising step
        4. Parameters are used in Eq. 5 for sampling
        
        Args:
            x_t: Noisy images at timestep t
            t: Current timestep
            
        Returns:
            mu: Predicted mean for denoising
            sigma: Predicted standard deviation
            
        Example:
            >>> model = DiffusionModel(28, 1)
            >>> x_t = torch.randn(16, 1, 28, 28)  # Noisy images
            >>> t = torch.tensor([100])           # Timestep
            >>> mu, sigma = model.get_mu_sigma(x_t, t)
            >>> mu.shape    # torch.Size([16, 1, 28, 28])
            >>> sigma.shape # torch.Size([16, 1, 28, 28])
        """
        t = t.long()
        batch_size = x_t.shape[0]
        H, W = x_t.shape[2:]
        
        # Get MLP output and reshape
        mlp_out = self.mlp(x_t).view(
            batch_size, 2, self.n_colors,
            self.n_temporal_basis, H, W
        )
        
        # Get temporal projection
        t_weights = self.get_t_weights(t)
        temporal_projection = torch.matmul(self.temporal_basis, t_weights)
        
        # Calculate mu and sigma
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
        """Calculate mean and variance from MLP output"""
        # Extract coefficients
        mu_coeffs = mlp_out[:, 0]
        sigma_coeffs = mlp_out[:, 1]
        
        # Calculate mu and sigma
        mu = torch.sum(
            mu_coeffs * temporal_projection.view(1, 1, -1, 1, 1),
            dim=2
        )
        log_sigma = torch.sum(
            sigma_coeffs * temporal_projection.view(1, 1, -1, 1, 1),
            dim=2
        )
        
        # Apply scaling
        alpha_t = self.alpha[t]
        beta_t = self.beta[t]
        
        mu = mu * torch.sqrt(1 / alpha_t).view(-1, 1, 1, 1)
        sigma = torch.exp(log_sigma).clamp(
            min=beta_t.sqrt().item(),
            max=1.0
        )
        
        return mu, sigma

    def cost_single_t(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the training objective from Section 2.3
        
        Minimizes the negative log-likelihood -log p(x_0) as described in Eq. 9.
        The training process:
        1. Sample timestep t uniformly to get unbiased gradients
        2. Run forward process to get x_t using reparameterization
        3. Predict reverse process parameters μ_θ and σ_θ
        4. Calculate likelihood of x_0 under these parameters
        5. Average negative log-likelihood across batch
        
        Args:
            x: Input images to train on
            
        Returns:
            loss: Negative log-likelihood loss
            
        Example:
            >>> model = DiffusionModel(28, 1)
            >>> x = torch.randn(32, 1, 28, 28)  # Training images
            >>> loss = model.cost_single_t(x)
            >>> loss.backward()  # Backpropagate
        """
        # Forward process to get x_t
        X_noisy, _, t, _ = self.forward_process(x)
        
        # Get reverse process parameters μ_θ and σ_θ
        mu, sigma = self.get_mu_sigma(X_noisy, t)
        
        # Calculate negative log likelihood (Eq. 9)
        return -torch.distributions.Normal(mu, sigma).log_prob(x).mean()

    def get_t_weights(self, t):
        """Calculate temporal weights for a given time step"""
        t = t.long()  # Convert to long tensor for indexing
        weights = torch.zeros(self.trajectory_length, device=self.device)
        weights[t] = 1.0
        return weights.unsqueeze(-1)  # Shape: [trajectory_length, 1]

    def sample(self, batch_size: int) -> torch.Tensor:
        """Implements the sampling procedure from Section 2.4
        
        Generates new samples by running the reverse diffusion process:
        1. Start with pure noise x_T ~ N(0,I)
        2. Gradually denoise by sampling x_{t-1} ~ p(x_{t-1}|x_t)
        3. Repeat until reaching x_0
        
        This implements Algorithm 1 from the paper:
        - Uses learned reverse process to progressively remove noise
        - Each step uses neural network to predict denoising parameters
        - Final sample x_0 is drawn from the learned data distribution
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            samples: Generated images
            
        Example:
            >>> model = DiffusionModel(28, 1)
            >>> samples = model.sample(batch_size=16)
            >>> samples.shape  # torch.Size([16, 1, 28, 28])
            
            >>> # Generate RGB images
            >>> model = DiffusionModel(32, 3)  # CIFAR-10 size
            >>> samples = model.sample(batch_size=8)
            >>> samples.shape  # torch.Size([8, 3, 32, 32])
        """
        # Start from random noise (x_T ~ N(0,I))
        x = torch.randn(batch_size, self.n_colors, self.spatial_width, self.spatial_width, device=self.device)
        
        # Reverse diffusion process (Algorithm 1)
        for t in reversed(range(self.min_t, self.trajectory_length)):
            t_tensor = torch.tensor([t], device=self.device)
            
            # Get p(x_{t-1}|x_t) parameters
            mu, sigma = self.get_mu_sigma(x, t_tensor)
            
            # Sample x_{t-1} (except for t=0)
            if t > 0:
                noise = torch.randn_like(x)
                x = mu + sigma * noise  # Eq. 5
            else:
                x = mu  # For t=0, just use the mean
        
        return x
