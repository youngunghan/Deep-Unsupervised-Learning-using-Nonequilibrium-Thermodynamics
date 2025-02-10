# Import libraries
import torch
import numpy as np


# Main script
import matplotlib.pyplot as plt

def diffusion_step(Xmid, t, get_mu_sigma, denoise_sigma, mask, XT, device):
    """
    Perform single reverse diffusion step with optional denoising and inpainting
    
    Args:
        Xmid (torch.Tensor): Current state
        t (int): Current timestep
        get_mu_sigma (function): Function to get mean and variance
        denoise_sigma (float): Optional denoising strength
        mask (torch.Tensor): Optional inpainting mask
        XT (torch.Tensor): Target image for inpainting
        device (torch.device): Device to use
        
    Returns:
        Xmid (torch.Tensor): Updated state after reverse step
    """
    mu, sigma = get_mu_sigma(Xmid, t)
    if denoise_sigma is not None:
        sigma_new = (sigma**-2 + denoise_sigma**-2)**-0.5
        mu_new = mu * sigma_new**2 * sigma**-2 + XT * sigma_new**2 * denoise_sigma**-2
        sigma = sigma_new
        mu = mu_new
    if mask is not None:
        mu.flat[mask] = XT.flat[mask]
        sigma.flat[mask] = 0.
    Xmid = mu + sigma*(torch.normal(0,1,size=Xmid.shape).to(device))

    return Xmid

def generate_inpaint_mask(n_samples, n_colors, spatial_width):
    """
    The mask will be True where we keep the true image, and False where we're
    inpainting.
    """
    mask = np.zeros((n_samples, n_colors, spatial_width, spatial_width), dtype=bool)
    # simple mask -- just mask out half the image
    mask[:,:,:,spatial_width/2:] = True
    return mask.ravel()

def plot_images(images, title=None, ncols=6):
    # Helper function to plot images with a title and save them if needed.
    images = images.detach().cpu().numpy()
    n_images = images.shape[0]
    nrows = (n_images + ncols - 1) // ncols  # Calculate required rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < n_images:
            img = images[i]
            if img.shape[0] == 1:  # Grayscale image
                img = img[0]
            elif img.shape[0] == 3:  # RGB image
                img = img.transpose(1, 2, 0)
            axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].axis('off')

    # Hide any remaining axes
    for ax in axes[len(images):]:
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()
    plt.close(fig)