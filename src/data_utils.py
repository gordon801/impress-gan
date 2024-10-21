# Imports
import os
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image

# Define constants
IMAGE_SIZE = 64

def get_transform():
    """Get transformation pipeline for training images."""
    transform = transforms.Compose([
            transforms.Resize(size=IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Common practice to min-max normalise pixel values to [-1, 1] for GANs
        ])
    return transform
    
class ImpressDataset(Dataset):
    """
    Custom dataset for the Impressionism Artwork dataset.
    """
    def __init__(self, data_dir, transform=get_transform()):
        """
        Initialise the Impressionism Artwork dataset.

        Args:
            data_dir (str): The directory containing the dataset.
            transform (transforms.Compose, optional): Optional transform to be applied to the images. Defaults to the 
                                                      result of `get_transform()` if not provided.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)
        
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)
        
    def __getitem__(self, index):
        """Get the image and label for a specific index."""
        img_path = os.path.join(self.data_dir, self.image_files[index])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image

def prepare_dataloader(data_dir, batch_size, num_workers, dataset_type='full'):
    """Prepare DataLoader for dataset."""   
    # Initialise datasetz
    dataset = ImpressDataset(
        data_dir=data_dir, 
        transform=get_transform()
    )

    if dataset_type == 'mini':
        dataset = Subset(dataset, list(range(300)))

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return dataset, dataloader