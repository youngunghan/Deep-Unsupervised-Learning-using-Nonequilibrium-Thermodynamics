import torch
import torch.optim as optim
from models.tbh import TBH
from utils.losses import TBHLoss
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import argparse
from config.config import Config
from dataset.dataloader import create_dataloaders
from trainer import train_epoch
from validate import validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint for resume training')
    args = parser.parse_args()

    # 설정
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 초기화
    model = TBH(
        hash_dim=cfg.HASH_DIM,
        feature_dim=cfg.FEATURE_DIM,
        bottleneck_dim=cfg.BOTTLENECK_DIM
    ).to(device)

    # Loss function 초기화
    criterion = TBHLoss(
        alpha=1.0,
        beta=0.1,
        gamma=0.1
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # 시작 epoch 설정
    start_epoch = 0

    # 체크포인트에서 학습 재개
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # 데이터 로더 설정
    train_loader, test_loader = create_dataloaders(cfg)

    # 학습 결과 저장 경로 설정
    date_today = datetime.datetime.now().strftime('%Y%m%d')
    save_paths = {
        'model': f'./result/{cfg.DATASET_NAME}/model/{date_today}',
        'log': f'./result/{cfg.DATASET_NAME}/log/{date_today}',
        'code': f'./result/{cfg.DATASET_NAME}/code/{date_today}'
    }
    writer = SummaryWriter(save_paths['log'])

    # 학습 재개
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        # Training with batch-level validation
        avg_loss = train_epoch(model, train_loader, test_loader, criterion, 
                             optimizer, device, epoch, cfg.NUM_EPOCHS, cfg, writer, save_paths)
        
        # 모델 저장 (epoch 단위)
        if (epoch + 1) % cfg.SAVE_FREQ == 0:
            # 모델 체크포인트 저장
            save_path = os.path.join(save_paths['model'], f'model-epoch{epoch+1}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Epoch checkpoint saved to {save_path}")
            
            # Epoch 단위 전체 validation 수행
            print(f"Performing full validation at epoch {epoch+1}")
            eval_results = validate(model, test_loader, device, cfg, fast_eval=False)
            
            # Validation 결과 저장 (json 형식)
            eval_json_path = os.path.join(save_paths['model'], f'eval-epoch{epoch+1}.json')
            import json
            with open(eval_json_path, 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'mAP': float(eval_results['mAP']),
                    'precision': {k: float(v) for k, v in eval_results['precision'].items()},
                    'loss': float(avg_loss),
                }, f, indent=4)
            print(f"Validation results saved to {eval_json_path}")
            
            # Validation 결과와 함께 모델 저장
            eval_save_path = os.path.join(save_paths['model'], f'model-epoch{epoch+1}-eval.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'eval_results': eval_results,
            }, eval_save_path)
            print(f"Validation checkpoint saved to {eval_save_path}")

if __name__ == '__main__':
    main()
