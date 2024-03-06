from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

class ImageDataset(Dataset):
    def __init__(self, data_dir, train = True, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.ds = CIFAR10(root=data_dir, 
                              train=True, 
                              download=True,
                              transform=transform)
        else: 
            self.ds = CIFAR10(root=data_dir, 
                              train=False, 
                              download=True,
                              transform=transform)
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image= self.ds[index][0]
        label = self.ds[index][1]
        
        return image, label