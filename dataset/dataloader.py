from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

class TBHDataset(Dataset):
    def __init__(self, dataset, transform_base, transform_strong=None):
        self.dataset = dataset
        self.transform_base = transform_base
        self.transform_strong = transform_strong
        
    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform_base:
            img = self.transform_base(img)  # transform 적용
        return img, label, index
        
    def __len__(self):
        return len(self.dataset)

def get_transforms(cfg):
    transform_base = transforms.Compose([
        transforms.Resize(cfg.INPUT_SIZE),
        transforms.CenterCrop(cfg.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(cfg.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform_base, transform_strong

def create_dataloaders(cfg):
    transform_base, transform_strong = get_transforms(cfg)
    
    # 기본 CIFAR10 데이터셋 (transform 없이)
    train_dataset_raw = datasets.CIFAR10(root=cfg.DATA_DIR, train=True,
                                       download=True, transform=None)
    test_dataset_raw = datasets.CIFAR10(root=cfg.DATA_DIR, train=False,
                                      download=True, transform=None)
    
    # Custom Dataset으로 감싸기
    train_dataset = TBHDataset(train_dataset_raw, transform_base, transform_strong)
    test_dataset = TBHDataset(test_dataset_raw, transform_base)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=cfg.BATCH_SIZE,
                            shuffle=True, 
                            num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, 
                           batch_size=cfg.BATCH_SIZE,
                           shuffle=False, 
                           num_workers=cfg.NUM_WORKERS)
    
    return train_loader, test_loader 