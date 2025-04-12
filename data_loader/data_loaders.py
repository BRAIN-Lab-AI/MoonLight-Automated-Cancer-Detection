from base.base_data_loader import BaseDataLoader
from torchvision import datasets, transforms

class BCDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.3,
                 num_workers=1, training=None, augment='basic', pin_memory=False):

        if augment == 'basic':
            trsfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        elif augment == 'advanced':
            trsfm = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        else:
            raise ValueError(f"Unknown augmentation type: {augment}")

        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory)
