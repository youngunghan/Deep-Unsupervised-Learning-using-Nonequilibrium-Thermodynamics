import torch
import datetime
import os
from scripts.validate import validate
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from utils import setup_logger

def train_epoch(model, train_loader, val_loader, optimizer, args, start_epoch=0, start_batch=0):
    # Set up logger
    logger = setup_logger(args.exp_name)
    
    # Initialize tensorboard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'runs/{args.exp_name}_{current_time}'
    writer = SummaryWriter(log_dir)
    
    # Log initial information
    logger.info("="*50)
    logger.info(f"Starting training experiment: {args.exp_name}")
    logger.info(f"Tensorboard logs will be saved to {log_dir}")
    logger.info(f"Training device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Validation interval: {args.val_interval} batches")
    if start_epoch > 0 or start_batch > 0:
        logger.info(f"Resuming from epoch {start_epoch+1}, batch {start_batch}")
    logger.info("="*50)
    print("\n")  # Add empty line for readability

    device = args.device
    best_val_loss = float('inf')
    total_batches = len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        epoch_start_time = datetime.datetime.now()
        
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-"*20)
        
        for batch_idx, x in enumerate(train_loader):
            # Skip batches that were already processed in the last saved epoch
            if epoch == start_epoch and batch_idx < start_batch:
                continue
                
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            
            optimizer.zero_grad()
            loss = model.cost_single_t(x)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:  # More frequent progress updates
                progress = (batch_idx + 1) / total_batches * 100
                print(f"\rProgress: {progress:.1f}% ({batch_idx + 1}/{total_batches}) "
                      f"- Loss: {loss.item():.4f}", end="")
            
            # Validate and save every N batches
            if (batch_idx + 1) % args.val_interval == 0:
                model.eval()
                val_loss = validate(model, val_loader, device)
                print("\n")  # New line after progress
                logger.info(f"Validation at epoch {epoch+1}, batch {batch_idx+1}")
                logger.info(f"Current train loss: {loss.item():.4f}")
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Log validation loss
                writer.add_scalar('eval/loss', val_loss, epoch * len(train_loader) + batch_idx)
                
                # Generate and log sample images
                with torch.no_grad():
                    # Generate samples
                    logger.info("Generating samples for visualization...")
                    samples = model.sample(2)
                    grid_fake = make_grid(samples, nrow=2, normalize=True)
                    writer.add_image('samples/generated', grid_fake, epoch * len(train_loader) + batch_idx)
                    
                    # Log original and noisy images
                    grid_real = make_grid(x[:2], nrow=2, normalize=True)
                    writer.add_image('samples/real', grid_real, epoch * len(train_loader) + batch_idx)
                    
                    noisy_images, _, t, _ = model.forward_process(x[:2])
                    grid_noisy = make_grid(noisy_images, nrow=2, normalize=True)
                    writer.add_image('samples/noisy', grid_noisy, epoch * len(train_loader) + batch_idx)
                    logger.info("Sample generation completed")
                
                model.train()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': args
                }
                
                # Save latest checkpoint
                checkpoint_path = os.path.join(args.save_dir, f'{args.exp_name}_epoch{epoch+1}_batch{batch_idx+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(args.save_dir, f'{args.exp_name}_best.pth')
                    torch.save(checkpoint, best_model_path)
                    logger.info(f"New best model saved! Previous best: {best_val_loss:.4f} -> New best: {val_loss:.4f}")
                
                print("-"*50)  # Separator line
        
        # Epoch completion summary
        epoch_end_time = datetime.datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_train_loss = total_train_loss / len(train_loader)
        
        logger.info(f"Completed epoch {epoch+1}/{args.epochs}")
        logger.info(f"Epoch duration: {epoch_duration}")
        logger.info(f"Average train loss: {avg_train_loss:.4f}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
        logger.info("="*50)
        
        # Log epoch-level metrics
        writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Duration: {epoch_duration}")
        print(f"Average Loss: {avg_train_loss:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print("="*50 + "\n")

    # Training completion
    logger.info("Training completed!")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
    writer.close()