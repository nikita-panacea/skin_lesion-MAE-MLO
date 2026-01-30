# datasets/isic2019_dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# canonical 8 ISIC-2019 classes (paper)
ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

class ISIC2019Dataset(Dataset):
    """
    CSV-based ISIC-2019 loader.

    CSV can be:
      - Has column 'label' with integer 0..7
      - Or contains one-hot columns named with ISIC_CLASSES (any subset allowed).
      - Must contain an 'image' or 'image_id' column with image id (without .jpg or with extension).
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # image id column detection
        if "image" in self.df.columns:
            self.images = self.df["image"].astype(str).values
        elif "image_id" in self.df.columns:
            self.images = self.df["image_id"].astype(str).values
        elif "filename" in self.df.columns:
            self.images = self.df["filename"].astype(str).values
        else:
            raise ValueError("CSV must contain 'image' or 'image_id' or 'filename' column")

        # labels: either integer 'label' OR one-hot columns for ISIC_CLASSES
        if "label" in self.df.columns:
            self.labels = self.df["label"].astype(int).values
        else:
            # If the CSV contains some or all of the ISIC_CLASSES columns, infer class index.
            available = [c for c in ISIC_CLASSES if c in self.df.columns]
            if not available:
                raise ValueError("CSV missing 'label' column and no one-hot class columns found")
            onehots = self.df[available].fillna(0).astype(int).values
            # If available columns are subset, map argmax over those columns to indices in the full list.
            # Map to indices in the full ISIC_CLASSES list:
            # We compute argmax over available columns, then map to the full-class index.
            argmax_idx = onehots.argmax(axis=1)
            # convert to full index
            mapping = {i: ISIC_CLASSES.index(col) for i, col in enumerate(available)}
            self.labels = [mapping[int(a)] for a in argmax_idx]

        # final conversions
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.images = [str(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        if not img_name.lower().endswith(".jpg") and not img_name.lower().endswith(".png"):
            img_name = img_name + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, img_name
