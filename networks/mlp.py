import torch
import torch.nn as nn

class MultiscaleConvolution(nn.Module):
    """Multi-scale convolution layer for extracting features at different resolutions
    
    Performs convolution at multiple scales to capture both fine details and global structure.

    
    Args:
        num_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        num_filters (int): Number of output filters/channels
        num_scales (int): Number of different scales to process
        filter_size (int): Convolution kernel size
        activation (callable): Activation function (defaults to shifted softplus)
        padding_mode (str): Padding mode for convolution
        upsample_mode (str): Interpolation mode for upsampling
        device (torch.device): Device to run computations on
    
    Example:
        >>> layer = MultiscaleConvolution(num_channels=1, num_filters=64)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> features = layer(x)  # Shape: [32, 64, 28, 28]
    """
    def __init__(
        self,
        num_channels: int,
        num_filters: int,
        num_scales: int = 4,
        filter_size: int = 5,
        activation: callable = None,
        padding_mode: str = 'reflect',
        upsample_mode: str = 'bilinear',
        device: torch.device = None
    ):
        """Initialize MultiscaleConvolution layer
        
        Args:
            num_channels (int): Number of input channels
            num_filters (int): Number of output filters/channels
            num_scales (int): Number of different scales to process
            filter_size (int): Convolution kernel size
            activation (callable): Activation function (defaults to shifted softplus)
            padding_mode (str): Padding mode for convolution
            upsample_mode (str): Interpolation mode for upsampling
            device (torch.device): Device to run computations on
            
        Returns:
            None
            
        Example:
            >>> layer = MultiscaleConvolution(num_channels=3, num_filters=64, num_scales=4)
            >>> layer.num_filters  # 64
        """
        super(MultiscaleConvolution, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configuration
        self.num_scales = num_scales
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.padding_mode = padding_mode
        self.upsample_mode = upsample_mode
        self.activation = activation if activation is not None else self._shifted_softplus
        
        # Initialize convolution layers
        self.conv_layers = self._build_conv_layers(num_channels)

    def _build_conv_layers(self, num_channels: int) -> nn.ModuleList:
        """Create convolution layers for each scale
        
        Args:
            num_channels (int): Number of input channels
            
        Returns:
            nn.ModuleList: List of Conv2d layers for different scales
            
        Example:
            >>> conv_layers = layer._build_conv_layers(3)  # For RGB input
            >>> len(conv_layers)  # Equal to num_scales
        """
        return nn.ModuleList([
            nn.Conv2d(
                num_channels,
                self.num_filters,
                kernel_size=self.filter_size,
                padding=self.filter_size // 2,
                padding_mode=self.padding_mode
            ).to(self.device) for _ in range(self.num_scales)
        ])

    @staticmethod
    def _shifted_softplus(x: torch.Tensor) -> torch.Tensor:
        """Smooth activation function: f(x) = log(1 + exp(x-1))
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Activated tensor with same shape as input
            
        Example:
            >>> x = torch.randn(2, 3)
            >>> activated = MultiscaleConvolution._shifted_softplus(x)
            >>> activated.shape  # torch.Size([2, 3])
        """
        return torch.nn.functional.softplus(x - 1)

    def downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Reduce resolution by a factor of 2^scale using average pooling
        
        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width]
            scale (int): Downsampling scale factor (0 means no downsampling)
            
        Returns:
            torch.Tensor: Downsampled tensor [batch, channels, height//2^scale, width//2^scale]
            
        Example:
            >>> x = torch.randn(1, 1, 32, 32)
            >>> y = layer.downsample(x, scale=2)  # Output: [1, 1, 8, 8]
        """
        if scale == 0:
            return x
        kernel_size = 2**scale
        return nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)

    def upsample(self, x: torch.Tensor, scale: int, size: int = None) -> torch.Tensor:
        """Increase resolution by a factor of 2^scale using interpolation
        
        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width]
            scale (int): Upsampling scale factor (0 means no upsampling)
            size (int): Target size (defaults to current_size * 2^scale)
            
        Returns:
            torch.Tensor: Upsampled tensor [batch, channels, size, size]
            
        Example:
            >>> x = torch.randn(1, 64, 8, 8)
            >>> y = layer.upsample(x, scale=2, size=32)  # Output: [1, 64, 32, 32]
        """
        if scale == 0:
            return x
        if size is None:
            size = x.shape[-1] * (2**scale)
        return nn.functional.interpolate(
            x,
            size=(size, size),
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode == 'bilinear' else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multiple scales and combine features
        
        Args:
            x (torch.Tensor): Input images [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Multi-scale features [batch, num_filters, height, width]
            
        Example:
            >>> layer = MultiscaleConvolution(1, 64)
            >>> x = torch.randn(8, 1, 64, 64)
            >>> out = layer(x)  # Shape: [8, 64, 64, 64]
        """
        x = x.to(self.device)
        
        # Get original size and initialize output
        original_size = x.shape[-1]
        output = torch.zeros(
            x.size(0),
            self.num_filters,
            original_size,
            original_size,
            device=self.device
        )
        
        # Process each scale
        for scale in range(self.num_scales):
            y = self.downsample(x, scale)
            y = self.conv_layers[scale](y)
            y = self.activation(y)
            y = self.upsample(y, scale, original_size)
            output += y
            
        return output / self.num_scales

    @property
    def output_size(self) -> int:
        """Get the number of output channels
        
        Args:
            None
            
        Returns:
            int: Number of output channels (filters)
            
        Example:
            >>> layer = MultiscaleConvolution(1, 64)
            >>> layer.output_size  # 64
        """
        return self.num_filters

# Define the MLP for estimating mean and variance
class MLP(nn.Module):
    """Neural network for predicting denoising parameters in diffusion models
    
    Uses two parallel processing paths to extract features:
    1. Multi-scale path: Captures spatial features across different scales
    2. Dense path: Processes pixel-wise channel relationships
    
    Args:
        num_channels (int): Number of input channels
        num_layers (int): Number of layers in each path
        num_output_channels (int): Number of output channels
        hidden_channels (int): Hidden dimension size
        activation (nn.Module): Activation function class
        reduction_factor (int): Factor to reduce channels in final layers
        
    Example:
        >>> mlp = MLP(num_channels=1, num_output_channels=20)
        >>> x = torch.randn(16, 1, 28, 28)
        >>> out = mlp(x)  # Shape: [16, 20, 28, 28]
    """
    def __init__(self, num_channels=1, num_layers=200, num_output_channels=20,
                 hidden_channels=128, activation=nn.Tanh, reduction_factor=2):
        """Initialize MLP with parallel processing paths
        
        Args:
            num_channels (int): Number of input channels
            num_layers (int): Number of layers in each processing path
            num_output_channels (int): Number of output channels
            hidden_channels (int): Hidden dimension size
            activation (nn.Module): Activation function class
            reduction_factor (int): Channel reduction factor for final layers
            
        Returns:
            None
            
        Example:
            >>> mlp = MLP(num_channels=3, num_layers=100, num_output_channels=10)
            >>> mlp.num_channels  # 3
        """
        super(MLP, self).__init__()
        
        # Core network parameters
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_output_channels = num_output_channels
        self.activation = activation()
        
        # Build parallel processing paths
        self.msc_path = self._build_msc_path()
        self.dense_path = self._build_dense_path()
        
        # Final layers for combining and reducing features
        self.final_conv = self._build_final_layers(reduction_factor)

    def _build_msc_path(self):
        """Create multi-scale convolutional path for spatial feature extraction
        
        Args:
            None
            
        Returns:
            nn.ModuleList: List of MultiscaleConvolution layers
            
        Example:
            >>> mlp = MLP(num_layers=3)
            >>> msc_path = mlp._build_msc_path()
            >>> len(msc_path)  # 3
        """
        layers = nn.ModuleList([
            MultiscaleConvolution(self.num_channels, self.hidden_channels)
        ])
        
        # Add remaining layers
        layers.extend([
            MultiscaleConvolution(self.hidden_channels, self.hidden_channels)
            for _ in range(self.num_layers - 1)
        ])
        
        return layers

    def _build_dense_path(self):
        """Create 1x1 convolution path for pixel-wise feature processing
        
        Args:
            None
            
        Returns:
            nn.ModuleList: List of Conv2d and activation layers
            
        Example:
            >>> mlp = MLP(num_layers=2)
            >>> dense_path = mlp._build_dense_path()
            >>> len(dense_path)  # 4 (2 layers * 2 components each)
        """
        layers = nn.ModuleList()
        
        # First layer
        layers.extend([
            nn.Conv2d(self.num_channels, self.hidden_channels, 1),
            self.activation
        ])
        
        # Remaining layers
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Conv2d(self.hidden_channels, self.hidden_channels, 1),
                self.activation
            ])
            
        return layers

    def _build_final_layers(self, reduction_factor):
        """Create final layers to reduce feature dimensions and generate output
        
        Args:
            reduction_factor (int): Factor to reduce hidden channels
            
        Returns:
            nn.Sequential: Sequential layers for dimension reduction and output
            
        Example:
            >>> mlp = MLP(hidden_channels=128, reduction_factor=2)
            >>> final_layers = mlp._build_final_layers(2)
            >>> # Reduces 128 -> 64 -> num_output_channels
        """
        reduced_channels = self.hidden_channels // reduction_factor
        return nn.Sequential(
            nn.Conv2d(self.hidden_channels, reduced_channels, 1),
            self.activation,
            nn.Conv2d(reduced_channels, self.num_output_channels, 1),
            self.activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through both parallel paths and combine features
        
        Args:
            x (torch.Tensor): Input images [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Combined features [batch, num_output_channels, height, width]
            
        Example:
            >>> net = MLP(num_channels=3, num_output_channels=6)
            >>> x = torch.randn(4, 3, 32, 32)
            >>> out = net(x)  # Shape: [4, 6, 32, 32]
        """
        # Process through parallel paths
        x_msc = self._forward_msc_path(x.clone())
        x_dense = self._forward_dense_path(x.clone())
        
        # Combine and reduce dimensions
        return self.final_conv(x_msc + x_dense)

    def _forward_msc_path(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multi-scale convolutional layers sequentially
        
        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Multi-scale features [batch, hidden_channels, height, width]
            
        Example:
            >>> mlp = MLP(num_channels=1, hidden_channels=64)
            >>> x = torch.randn(8, 1, 32, 32)
            >>> features = mlp._forward_msc_path(x)
            >>> features.shape  # torch.Size([8, 64, 32, 32])
        """
        for layer in self.msc_path:
            x = layer(x)
        return x

    def _forward_dense_path(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through 1x1 convolutional layers with activations
        
        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Dense features [batch, hidden_channels, height, width]
            
        Example:
            >>> mlp = MLP(num_channels=1, hidden_channels=64)
            >>> x = torch.randn(8, 1, 32, 32)
            >>> features = mlp._forward_dense_path(x)
            >>> features.shape  # torch.Size([8, 64, 32, 32])
        """
        for i in range(0, len(self.dense_path), 2):
            x = self.dense_path[i](x)      # Conv2d layer
            x = self.dense_path[i+1](x)    # Activation function
        return x
