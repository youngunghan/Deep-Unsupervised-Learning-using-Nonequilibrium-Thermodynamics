import torch
from tqdm import tqdm
from utils.metrics import (
    calculate_mAP, 
    calculate_precision_at_k, 
    calculate_precision_recall_curve
)

def validate(model, test_loader, device, cfg, fast_eval=True):
    model.eval()
    database_codes = []
    database_labels = []
    
    # Fast evaluation mode
    max_samples = 1000 if fast_eval else None
    sample_count = 0
    
    with torch.no_grad():
        for data, labels, indices in tqdm(test_loader, desc='Validating'):
            if fast_eval and sample_count >= max_samples:
                break
                
            data = data.to(device)
            _, _, b, _ = model.encode(data)
            # Convert to binary codes
            b = (b.sign() + 1) / 2
            database_codes.append(b.cpu())
            database_labels.append(labels)
            
            sample_count += data.size(0)
    
    database_codes = torch.cat(database_codes, dim=0)
    database_labels = torch.cat(database_labels, dim=0)
    
    results = {
        'mAP': calculate_mAP(database_codes, database_codes, database_labels, database_labels),
        'precision': {}
    }
    
    # Fast evaluation mode에서는 PR curve 계산 생략
    if not fast_eval:
        results['pr_curve'] = calculate_precision_recall_curve(
            database_codes, database_codes, database_labels, database_labels
        )
    
    # Calculate precision@K for selected K values only
    for k in [100]:  # Fast evaluation에서는 K=100만 계산
        results['precision'][k] = calculate_precision_at_k(
            database_codes, database_codes, database_labels, database_labels, k=k
        )
    
    return results 