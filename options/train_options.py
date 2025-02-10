import argparse
import os

def train_options():
    parser = argparse.ArgumentParser(description='Training options for Diffusion Model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Number of samples per batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    
    # Model parameters
    parser.add_argument('--spatial_width', type=int, default=28,
                      help='Spatial width of the input data')
    parser.add_argument('--n_colors', type=int, default=1,
                      help='Number of color channels')
    parser.add_argument('--n_temporal_basis', type=int, default=10,
                      help='Number of temporal basis functions')
    parser.add_argument('--trajectory_length', type=int, default=1000,
                      help='Length of diffusion trajectory')
    parser.add_argument('--hidden_channels', type=int, default=128,
                      help='Number of hidden channels in the network')
    parser.add_argument('--num_layers', type=int, default=200,
                      help='Number of layers in the network')
    
    # Diffusion parameters
    parser.add_argument('--beta_start', type=float, default=0.01,
                      help='Starting value for beta schedule')
    parser.add_argument('--beta_end', type=float, default=0.05,
                      help='Ending value for beta schedule')
    parser.add_argument('--min_t', type=int, default=100,
                      help='Minimum number of diffusion steps')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--exp_name', type=str, default='diffusion_default',
                      help='Name of the experiment')
    parser.add_argument('--val_interval', type=int, default=10,
                      help='How often to perform validation (in epochs)')
    
    # Save and load options
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--continue_train', action='store_true',
                      help='Continue training from a checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                      help='Path to the checkpoint file for continue training')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    return args