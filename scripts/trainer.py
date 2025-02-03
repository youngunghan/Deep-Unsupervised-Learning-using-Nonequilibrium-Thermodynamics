import torch
from validate import validate  # eval 대신 validate import
import os
import json

def train_epoch(model, train_loader, test_loader, criterion, optimizer, device, epoch, num_epochs, cfg, writer, save_paths):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, labels, indices) in enumerate(train_loader):
        # transform은 이미 데이터셋에서 적용됨
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        x_orig, z1, b, z2, x_recon = model(data)
        
        # Calculate loss
        loss, recon_loss, feat_loss, quant_loss = criterion(x_orig, z1, b, z2, x_recon, data)
        
        # Training stage 설정
        if epoch < num_epochs // 2:
            criterion.training_stage = 1
            for param in model.bottleneck1.parameters():
                param.requires_grad = False
        else:
            criterion.training_stage = 2
            for param in model.bottleneck1.parameters():
                param.requires_grad = True
            
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        
        # 배치 단위 로깅
        if batch_idx % cfg.PRINT_FREQ == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss:.4f}')
            
            # Training metrics
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss, global_step)
            
            # 배치 단위 validation
            if batch_idx % cfg.VAL_BATCH_FREQ == 0:
                print(f"Performing fast validation at epoch {epoch+1}, batch {batch_idx}")
                model.eval()
                with torch.no_grad():
                    eval_results = validate(model, test_loader, device, cfg, fast_eval=False)
                    
                    # Validation metrics logging
                    writer.add_scalar('val/loss', eval_results.get('loss', 0), global_step)
                    writer.add_scalar('val/mAP', eval_results['mAP'], global_step)
                    
                    # Precision metrics
                    for k in eval_results['precision']:
                        writer.add_scalar(f'val/precision@{k}', 
                                        eval_results['precision'][k], 
                                        global_step)
                    
                    # 배치 단위 저장
                    save_path = os.path.join(save_paths['model'], f'model-epoch{epoch+1}-batch{batch_idx}.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'eval_results': eval_results,
                    }, save_path)
                    print(f"Checkpoint saved to {save_path}")
                model.train()
    
    return total_loss / len(train_loader) 