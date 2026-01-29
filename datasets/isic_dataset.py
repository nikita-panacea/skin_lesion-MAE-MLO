# datasets/isic_dataset.py
"""
ISIC 2019 Dataset Loader

Expected CSV format:
- 'image' column: image filename (without extension)
- One-hot encoded columns for each class OR 'label' column with integer labels
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# ISIC 2019 class names
ISIC_CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']


class ISICDataset(Dataset):
    """
    ISIC 2019 Dataset
    
    Args:
        csv_path: Path to CSV file with image names and labels
        img_dir: Directory containing images
        transform: Torchvision transforms to apply
    """
    
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Extract image paths and labels
        self.image_names = self.df['image'].values
        
        # Check if labels are one-hot encoded or already as integers
        if 'label' in self.df.columns:
            # Labels already as integers
            self.labels = self.df['label'].astype(int).values
        else:
            # One-hot encoded - convert to integers
            label_cols = [col for col in ISIC_CLASSES if col in self.df.columns]
            if not label_cols:
                raise ValueError(f"CSV must contain 'label' column or one-hot columns {ISIC_CLASSES}")
            
            # Get one-hot matrix
            onehot = self.df[label_cols].fillna(0).astype(int).values
            
            # Convert to integer labels
            self.labels = np.argmax(onehot, axis=1)
            
            # Map from CSV column order to ISIC_CLASSES order
            label_mapping = {i: ISIC_CLASSES.index(col) for i, col in enumerate(label_cols)}
            self.labels = np.array([label_mapping[int(lbl)] for lbl in self.labels])
        
        print(f"Loaded {len(self)} samples from {csv_path}")
        print(f"Label distribution: {np.bincount(self.labels, minlength=len(ISIC_CLASSES))}")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_names[idx]
        
        # Try different extensions
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = os.path.join(self.img_dir, img_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        if img_path is None:
            # Try without adding extension (maybe it's already in the name)
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_name}")
        
        # Load and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


def test_dataset():
    """Test the dataset loader"""
    import torchvision.transforms as transforms
    
    # Test transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test paths (update these)
    csv_path = '/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv'
    img_dir = '/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/images'
    
    print("Testing ISIC Dataset Loader...")
    print("="*60)
    
    try:
        dataset = ISICDataset(csv_path, img_dir, transform)
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test loading first sample
        img, label = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Image shape: {img.shape}")
        print(f"  Label: {label.item()} ({ISIC_CLASSES[label.item()]})")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Test a few more samples
        for i in range(min(5, len(dataset))):
            img, label = dataset[i]
            print(f"Sample {i}: shape={img.shape}, label={label.item()}")
        
        print("\n✓ Dataset test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_dataset()