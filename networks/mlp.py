import torch
import torch.nn as nn

class MultiscaleConvolution(nn.Module):
    """    
    Args:
        num_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        num_filters (int): Number of output channels for each convolution
        num_scales (int, optional): Number of different scales to process. Default: 4
                                  Each scale is a power of 2 downsampling
        filter_size (int, optional): Convolution kernel size. Default: 5
                                   Larger size captures more spatial context
        activation (callable, optional): Custom activation function. Default: shifted softplus
                                      f(x) = log(1 + exp(x-1))
        padding_mode (str, optional): Convolution padding type. Default: 'reflect'
                                    Helps handle image boundaries
        upsample_mode (str, optional): Interpolation method. Default: 'bilinear'
                                     Used when restoring original resolution
        device (torch.device, optional): Computing device. Default: None (auto-detect)
    
    Shape:
        - Input: (batch_size, num_channels, height, width)
        - Output: (batch_size, num_filters, height, width)
        
    Examples:
        >>> # Process MNIST images
        >>> layer = MultiscaleConvolution(num_channels=1, num_filters=64)
        >>> x = torch.randn(32, 1, 28, 28)  # Batch of MNIST images
        >>> out = layer(x)  # Shape: [32, 64, 28, 28]
        
        >>> # Process RGB images
        >>> layer = MultiscaleConvolution(num_channels=3, num_filters=128)
        >>> x = torch.randn(16, 3, 64, 64)  # Batch of RGB images
        >>> out = layer(x)  # Shape: [16, 128, 64, 64]
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
        """Build convolution layers for each scale
        
        Creates separate convolution layers for each scale to allow
        independent processing at different resolutions.
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
        """Shifted softplus activation: f(x) = log(1 + exp(x-1))
        
        This activation function:
        - Is smoother than ReLU
        - Has better gradient properties
        - Helps prevent saturation
        """
        return torch.nn.functional.softplus(x - 1)

    def downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample input using average pooling
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]
            scale (int): Scale factor (power of 2)
            
        Returns:
            torch.Tensor: Downsampled tensor [batch_size, channels, height/2^scale, width/2^scale]
            
        Example:
            >>> x = torch.randn(1, 1, 28, 28)
            >>> y = downsample(x, scale=2)  # Output shape: [1, 1, 7, 7]
        """
        if scale == 0:
            return x
        kernel_size = 2**scale
        return nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)

    def upsample(self, x: torch.Tensor, scale: int, size: int = None) -> torch.Tensor:
        """Upsample input using specified interpolation mode
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]
            scale (int): Scale factor (power of 2)
            size (int, optional): Target size. Default: None
            
        Returns:
            torch.Tensor: Upsampled tensor [batch_size, channels, size, size]
            
        Example:
            >>> x = torch.randn(1, 64, 7, 7)
            >>> y = upsample(x, scale=2, size=28)  # Output shape: [1, 64, 28, 28]
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
        """Forward pass of the multi-scale convolution
        
        Process steps:
        1. Downsample input to different scales
        2. Apply convolution at each scale
        3. Upsample results back to original size
        4. Average all scales to get final output
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Processed tensor [batch_size, num_filters, height, width]
            
        Example:
            >>> layer = MultiscaleConvolution(1, 64)
            >>> x = torch.randn(32, 1, 28, 28)
            >>> out = layer(x)  # Shape: [32, 64, 28, 28]
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
        """Get the number of output channels"""
        return self.num_filters

# Define the MLP for estimating mean and variance
class MLP(nn.Module):
    """
    Args:
        num_channels (int): Number of input channels (e.g., 1 for grayscale)
        num_layers (int): Number of layers in each processing path
        num_output_channels (int): Number of output channels (e.g., 2*n for mean/variance)
        hidden_channels (int, optional): Width of hidden layers. Default: 128
        activation (nn.Module, optional): Activation function. Default: nn.Tanh
        reduction_factor (int, optional): Channel reduction in final layers. Default: 2
    
    Shape:
        - Input: (batch_size, num_channels, height, width)
        - Output: (batch_size, num_output_channels, height, width)
    
    Examples:
        >>> # For MNIST-like images
        >>> mlp = MLP(num_channels=1, num_layers=200, num_output_channels=20)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> out = mlp(x)  # Shape: [32, 20, 28, 28]
        
        >>> # For RGB images with larger resolution
        >>> mlp = MLP(num_channels=3, num_layers=200, num_output_channels=30)
        >>> x = torch.randn(16, 3, 64, 64)
        >>> out = mlp(x)  # Shape: [16, 30, 64, 64]
        
        >>> # Split output into mean and variance
        >>> mean, var = out.chunk(2, dim=1)  # Each has half the channels
    """
    def __init__(self, num_channels=1, num_layers=200, num_output_channels=20,
                 hidden_channels=128, activation=nn.Tanh, reduction_factor=2):
        super(MLP, self).__init__()
        
        # Core network parameters
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_output_channels = num_output_channels
        self.activation = activation()
        
        # Build parallel processing paths
        self.msc_path = self._build_msc_path()  # Multi-scale convolution path
        self.dense_path = self._build_dense_path()  # 1x1 convolution path
        
        # Final layers for combining and reducing features
        self.final_conv = self._build_final_layers(reduction_factor)

    def _build_msc_path(self):
        """Build multi-scale convolution path
        
        Creates a sequence of MultiscaleConvolution layers that:
        1. Process input at multiple resolutions
        2. Capture spatial dependencies
        3. Maintain consistent channel dimensions
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
        """Build dense path with 1x1 convolutions
        
        Creates a sequence of 1x1 convolutions that:
        1. Process each pixel independently
        2. Allow channel-wise feature mixing
        3. Maintain spatial dimensions
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
        """Construct final layers that reduce feature dimensions
        
        Architecture:
        1. Reduce channels by reduction_factor to compress features
        2. Apply activation for non-linearity
        3. Project to final output channels (e.g., for mean and variance)
        4. Final activation for stable outputs
        """
        reduced_channels = self.hidden_channels // reduction_factor
        return nn.Sequential(
            nn.Conv2d(self.hidden_channels, reduced_channels, 1),
            self.activation,
            nn.Conv2d(reduced_channels, self.num_output_channels, 1),
            self.activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Network forward pass combining multi-scale and dense paths
        
        Process flow:
        1. Split input into two parallel paths
        2. Process through multi-scale convolutions for spatial features
        3. Process through 1x1 convolutions for channel mixing
        4. Combine results and reduce dimensions
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, num_output_channels, height, width]
            
        Example:
            >>> mlp = MLP(num_channels=1, num_output_channels=2)
            >>> x = torch.randn(32, 1, 28, 28)
            >>> out = mlp(x)  # Shape: [32, 2, 28, 28]
            >>> mu, sigma = out[:, 0], out[:, 1]  # Split into statistics
        """
        # Process through parallel paths independently
        x_msc = self._forward_msc_path(x.clone())  # Multi-scale features
        x_dense = self._forward_dense_path(x.clone())  # Dense features
        
        # Combine features and apply dimension reduction
        return self.final_conv(x_msc + x_dense)

    def _forward_msc_path(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multi-scale path
        
        Sequentially applies MultiscaleConvolution layers to:
        1. Capture features at different scales
        2. Process spatial dependencies
        3. Maintain original resolution
        """
        for layer in self.msc_path:
            x = layer(x)
        return x

    def _forward_dense_path(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through dense path
        
        Sequentially applies 1x1 convolutions to:
        1. Mix channel information
        2. Process each pixel independently
        3. Maintain spatial dimensions
        """
        for i in range(0, len(self.dense_path), 2):
            x = self.dense_path[i](x)  # Conv2d
            x = self.dense_path[i+1](x)  # Activation
        return x