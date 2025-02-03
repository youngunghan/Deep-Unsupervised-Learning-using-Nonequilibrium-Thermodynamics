import torch
from tqdm import tqdm
import argparse
from models.tbh import TBH
from dataset.dataloader import create_dataloaders
from config.config import Config
from utils.metrics import (
    calculate_mAP, 
    calculate_precision_at_k, 
    calculate_precision_recall_curve
)

def evaluate_retrieval(model, query_loader, db_loader, device, cfg):
    """
    Perform retrieval evaluation as in TBH paper
    """
    model.eval()
    
    # Database codes cache
    database_codes_cache = {}
    database_labels_cache = {}
    
    # Extract database codes and labels with caching
    database_codes = []
    database_labels = []
    print("Extracting database codes...")
    with torch.no_grad():
        for batch_idx, (data, labels, indices) in enumerate(tqdm(db_loader)):
            # 캐시된 코드가 있으면 사용
            uncached_indices = []
            uncached_data = []
            uncached_batch_indices = []
            
            for i, idx in enumerate(indices):
                idx = idx.item()
                if idx in database_codes_cache:
                    database_codes.append(database_codes_cache[idx].unsqueeze(0))
                    database_labels.append(database_labels_cache[idx].unsqueeze(0))
                else:
                    uncached_indices.append(idx)
                    uncached_data.append(data[i])
                    uncached_batch_indices.append(i)
            
            # 캐시되지 않은 데이터만 처리
            if uncached_data:
                uncached_data = torch.stack(uncached_data).to(device)
                _, _, b, _ = model.encode(uncached_data)
                b = (b.sign() + 1) / 2
                b = b.cpu()
                
                # 캐시 업데이트
                for i, idx in enumerate(uncached_indices):
                    database_codes_cache[idx] = b[i]
                    database_labels_cache[idx] = labels[uncached_batch_indices[i]]
                    database_codes.append(b[i].unsqueeze(0))
                    database_labels.append(labels[uncached_batch_indices[i]].unsqueeze(0))
    
    database_codes = torch.cat(database_codes, dim=0)
    database_labels = torch.cat(database_labels, dim=0)
    
    # Query codes cache
    query_codes_cache = {}
    query_labels_cache = {}
    
    # Extract query codes and labels with caching
    query_codes = []
    query_labels = []
    print("Extracting query codes...")
    with torch.no_grad():
        for batch_idx, (data, labels, indices) in enumerate(tqdm(query_loader)):
            # 캐시된 코드가 있으면 사용
            uncached_indices = []
            uncached_data = []
            uncached_batch_indices = []
            
            for i, idx in enumerate(indices):
                idx = idx.item()
                if idx in query_codes_cache:
                    query_codes.append(query_codes_cache[idx].unsqueeze(0))
                    query_labels.append(query_labels_cache[idx].unsqueeze(0))
                else:
                    uncached_indices.append(idx)
                    uncached_data.append(data[i])
                    uncached_batch_indices.append(i)
            
            # 캐시되지 않은 데이터만 처리
            if uncached_data:
                uncached_data = torch.stack(uncached_data).to(device)
                _, _, b, _ = model.encode(uncached_data)
                b = (b.sign() + 1) / 2
                b = b.cpu()
                
                # 캐시 업데이트
                for i, idx in enumerate(uncached_indices):
                    query_codes_cache[idx] = b[i]
                    query_labels_cache[idx] = labels[uncached_batch_indices[i]]
                    query_codes.append(b[i].unsqueeze(0))
                    query_labels.append(labels[uncached_batch_indices[i]].unsqueeze(0))
    
    query_codes = torch.cat(query_codes, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    
    # Calculate metrics
    print("Calculating metrics...")
    results = {
        'mAP': calculate_mAP(query_codes, database_codes, query_labels, database_labels),
        'precision': {},
        'pr_curve': calculate_precision_recall_curve(query_codes, database_codes, 
                                                   query_labels, database_labels)
    }
    
    # Calculate precision@K for all K values
    for k in cfg.TOP_K:
        results['precision'][k] = calculate_precision_at_k(
            query_codes, database_codes, query_labels, database_labels, k=k
        )
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name (default: cifar10)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    args = parser.parse_args()
    
    # 설정
    cfg = Config()
    # 커맨드라인 인자로 받은 값들로 cfg 업데이트
    cfg.DATASET_NAME = args.dataset
    cfg.BATCH_SIZE = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로더 생성 (query set과 database set 분리)
    train_loader, test_loader = create_dataloaders(cfg)  # test set을 query로 사용
    
    # 모델 로드
    model = TBH(
        hash_dim=cfg.HASH_DIM,
        feature_dim=cfg.FEATURE_DIM,
        bottleneck_dim=cfg.BOTTLENECK_DIM
    ).to(device)
    
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # Evaluation 수행
    results = evaluate_retrieval(model, test_loader, train_loader, device, cfg)
    
    # 결과 출력
    print("\nRetrieval Results:")
    print(f"mAP: {results['mAP']:.4f}")
    for k in sorted(results['precision'].keys()):
        print(f"Precision@{k}: {results['precision'][k]:.4f}")

if __name__ == '__main__':
    main() 