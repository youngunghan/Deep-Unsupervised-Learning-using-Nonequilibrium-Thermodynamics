from torch.utils.data import Dataset

# Define training dataset loader
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx][0]  # Only return the image, not the label