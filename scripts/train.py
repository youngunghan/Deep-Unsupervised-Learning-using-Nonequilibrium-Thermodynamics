import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from networks.dpm import DiffusionModel
from options.train_options import train_options
from dataset.dataloader import CustomDataset
from scripts.trainer import train_epoch

def main():
    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cudnn.benchmark = True
    
    args = train_options()
    
    # Reduce batch size
    args.batch_size = min(args.batch_size, 8)  # Further reduced batch size
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    
    # Prepare datasets with CustomDataset wrapper
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = CustomDataset(datasets.MNIST(root="./data", train=True, download=True, transform=transform))
    val_dataset = CustomDataset(datasets.MNIST(root="./data", train=False, download=True, transform=transform))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model and optimizer
    model = DiffusionModel(
        spatial_width=args.spatial_width,
        n_colors=args.n_colors,
        n_temporal_basis=args.n_temporal_basis,
        trajectory_length=args.trajectory_length,
        device=device
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint if continuing training
    start_epoch = 0
    start_batch = 0
    if args.continue_train and args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            
            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch'] + 1  # Start from next batch
            
            print(f"Resuming from epoch {start_epoch+1}, batch {start_batch}")
        else:
            print(f"No checkpoint found at {args.checkpoint_path}")
            return
    
    # Start training
    train_epoch(model, train_loader, val_loader, optimizer, args, 
                start_epoch=start_epoch, start_batch=start_batch)

    # Save the model
    model_path = f"checkpoints/{args.exp_name}_best.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()