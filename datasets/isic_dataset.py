"""
ISIC 2019 Dataset for skin lesion classification
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


ISIC_CLASSES = [
    'MEL',   # Melanoma
    'NV',    # Melanocytic nevus
    'BCC',   # Basal cell carcinoma
    'AK',    # Actinic keratosis
    'BKL',   # Benign keratosis
    'DF',    # Dermatofibroma
    'VASC',  # Vascular lesion
    'SCC'    # Squamous cell carcinoma
]


class ISICDataset(Dataset):
    """
    ISIC 2019 Dataset
    Supports both classification and MAE pretraining
    """
    def __init__(self, csv_path, img_dir, transform=None, return_path=False):
        """
        Args:
            csv_path: path to CSV with image names and labels
            img_dir: directory containing images
            transform: image transformations
            return_path: return image path along with data
        """
        self.img_dir = img_dir
        self.transform = transform
        self.return_path = return_path
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Extract image names and labels
        if 'image' in self.df.columns:
            self.image_names = self.df['image'].values
        else:
            self.image_names = self.df.iloc[:, 0].values
        
        # Get labels (either 'label' column or one-hot encoded)
        if 'label' in self.df.columns:
            self.labels = self.df['label'].values
        else:
            # Assume one-hot encoding in columns
            label_cols = [col for col in ISIC_CLASSES if col in self.df.columns]
            if label_cols:
                label_matrix = self.df[label_cols].values
                self.labels = label_matrix.argmax(axis=1)
            else:
                raise ValueError("No label information found in CSV")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_names[idx]
        if not img_name.endswith('.jpg'):
            img_name = img_name + '.jpg'
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        if self.return_path:
            return image, label, img_path
        else:
            return image, label


def get_transforms(train=True, img_size=224):
    """
    Get data transforms for ISIC dataset
    Args:
        train: bool - training or validation transforms
        img_size: int - target image size
    Returns:
        transform: torchvision transforms
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform