import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        
        # 고유한 클래스 레이블을 자동으로 생성
        self.class_names = sorted({img.split('_')[0] for img in self.image_files})
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)

        # Extract label from file name, e.g., classname_number.jpg -> classname
        label_name = img_name.split('_')[0]
        label = self.class_to_idx[label_name]  # Convert class name to index
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)  # Return the label as a tensor


# Transformations to apply (Resizing, Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main(args):
    # For sanity check
    train_dir = os.path.join(args.data_dir, 'train')

    # Create datasets
    train_dataset = ImageDataset(train_dir, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Check the first batch
    for images, labels in train_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch: {labels}")
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and dataloaders")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of the dataset (e.g., /images)")
    
    args = parser.parse_args()
    main(args)