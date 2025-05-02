import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from train_unet import UNet, RoadSkeletonDataset, IMAGE_SIZE, DATA_DIR
from evaluate_unet import compute_dice, compute_iou, compute_mse

# Config
# Refined learning rates for second-stage ablation
LEARNING_RATES = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3]
EPOCHS = 3
BATCH_SIZE = 8
CSV_FILENAME = 'ablation_eta_results.csv'
PLOT_PREFIX = 'ablation_eta_plot_'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
existing_etas = set()

# Load previously tested learning rates if CSV exists
if os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_etas.add(float(row['LearningRate']))

# Dice loss
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

# Data
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])
train_dataset = RoadSkeletonDataset(DATA_DIR, transform=transform, augment=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = RoadSkeletonDataset(DATA_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Store results
results = []

for eta in LEARNING_RATES:
    
    if eta in existing_etas:
        print(f"Skipping eta={eta:.0e} (already in results)")
        continue
    
    print(f"\n--- Training with learning rate {eta} ---")
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=eta)
    criterion = DiceLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    dice_scores, iou_scores, mse_scores = [], [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            pred = torch.sigmoid(outputs).cpu().numpy()[0, 0] > 0.5
            target_np = targets.numpy()[0, 0] > 0.5
            dice = compute_dice(pred, target_np)
            iou = compute_iou(pred, target_np)
            mse = compute_mse(outputs, targets)
            dice_scores.append(dice)
            iou_scores.append(iou)
            mse_scores.append(mse)

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_mse = np.mean(mse_scores)
    results.append((eta, avg_dice, avg_iou, avg_mse))
    print(f"Avg Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, MSE: {avg_mse:.4f}")

# Save to CSV
with open(CSV_FILENAME, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['LearningRate', 'AvgDice', 'AvgIoU', 'AvgMSE'])
    for row in results:
        writer.writerow(row)

print(f"\nResults saved to {CSV_FILENAME}")

# Plotting
etas = [r[0] for r in results]
dice_vals = [r[1] for r in results]
iou_vals = [r[2] for r in results]
mse_vals = [r[3] for r in results]

def plot_metric(etas, values, metric_name):
    plt.figure()
    plt.plot(etas, values, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Learning Rate')
    plt.grid(True)
    plt.savefig(f'{PLOT_PREFIX}{metric_name.lower()}.png')
    plt.close()

plot_metric(etas, dice_vals, 'Dice')
plot_metric(etas, iou_vals, 'IoU')
plot_metric(etas, mse_vals, 'MSE')

print("Plots saved as PNG files.")