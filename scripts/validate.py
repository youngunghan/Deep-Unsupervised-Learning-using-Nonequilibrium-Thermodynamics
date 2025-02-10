import torch

def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            loss = model.cost_single_t(x)
            total_val_loss += loss.item()
            num_batches += 1
            
            # Free memory
            del loss
            if num_batches % 10 == 0:
                torch.cuda.empty_cache()
    
    return total_val_loss / num_batches