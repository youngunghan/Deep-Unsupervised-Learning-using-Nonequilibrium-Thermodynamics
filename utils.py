import torch
import logging
import os
import datetime

def setup_logger(exp_name):
    """Set up logger to write to file and console
    
    Args:
        exp_name (str): Name of the experiment for log file naming
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logger('diffusion_exp1')
        >>> logger.info("Training started")
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('diffusion_training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    fh = logging.FileHandler(f'logs/{exp_name}_{current_time}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def generate_samples(model, n_samples=16):
    """Helper function to generate samples during training"""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # Start from random noise
        x = torch.randn(n_samples, model.n_colors, model.spatial_width, model.spatial_width).to(device)
        
        # Reverse diffusion process
        for t in reversed(range(model.trajectory_length)):
            t_tensor = torch.tensor([t], device=device)  # Convert integer t to tensor
            noisy_x = x.clone()
            mu, sigma = model.get_mu_sigma(noisy_x, t_tensor)
            
            if t > 0:  # Don't add noise at t=0
                noise = torch.randn_like(x)
                x = mu + sigma * noise
            else:
                x = mu
    
    model.train()
    return x