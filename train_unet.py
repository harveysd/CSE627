# Create and run a UNet to automatically create 1+px-wide skeletons of road
# maps in a 256 x 256 rasterized images based off of thick, raw road data. This
# data will be imported from OpenStreetMap.

# Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = 'data/thinning'
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 5e-3
MODEL_SAVE_PATH = 'misaligned_unet_model.pth'
IMAGE_SIZE = (256, 256)

# --- U-NET MODEL DEFINITION ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            CBR(1, 32),
            CBR(32, 64),
            nn.MaxPool2d(2),
            CBR(64, 64),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            CBR(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CBR(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- CUSTOM DATASET ---
class RoadSkeletonDataset(Dataset):
    def __init__(self, image_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.startswith('image') and f.endswith('.png')])
        self.targets = sorted([f for f in os.listdir(image_dir) if f.startswith('target') and f.endswith('.png')])
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        tgt_path = os.path.join(self.image_dir, self.targets[idx])
        image = Image.open(img_path).convert('L')
        target = Image.open(tgt_path).convert('L')

        # --- AUGMENTATION ---
        if self.augment:
            if np.random.rand() > 0.5:
                image = transforms.functional.hflip(image)
                target = transforms.functional.hflip(target)
            if np.random.rand() > 0.5:
                image = transforms.functional.vflip(image)
                target = transforms.functional.vflip(target)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                image = transforms.functional.rotate(image, angle)
                target = transforms.functional.rotate(target, angle)

        # --- MISALIGNMENT ---
        if self.augment:
            max_shift = 3  # pixels
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)
            target = transforms.functional.affine(target, angle=0, translate=(dx, dy), scale=1.0, shear=0)

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

# --- Dice LOSS FUNCTION
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# --- MAIN TRAINING LOOP ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    # Set augment to False to turn off augmentation
    dataset = RoadSkeletonDataset(DATA_DIR, transform=transform, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(device)
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded existing model from {MODEL_SAVE_PATH}. Continuing training.")
    else:
        print("No existing model found. Starting fresh training.")
    
    #criterion = nn.BCEWithLogitsLoss()  # for binary segmentation
    criterion = DiceLoss() # for Dice evaluation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()