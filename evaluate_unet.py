import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np

from train_unet import UNet, RoadSkeletonDataset, DATA_DIR, IMAGE_SIZE, MODEL_SAVE_PATH  # reuse config + model

# --- METRICS ---
def compute_iou(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union if union != 0 else 0.0

def compute_dice(pred, target):
    intersection = (pred & target).sum()
    return (2 * intersection) / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) != 0 else 0.0

def compute_mse(pred_logits, target):
    pred_probs = torch.sigmoid(pred_logits)
    return F.mse_loss(pred_probs, target).item()  # average per-pixel MSE

# --- DISPLAY IMAGES ---
def show_image_triplet(input_img, pred_img, target_img, idx, iou, dice, mse):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Input (Noisy Road)',
              f'Predicted Skeleton\nIoU={iou:.3f},  Dice={dice:.3f},  MSE={mse:.4f}', 
              'Ground Truth']
    images = [input_img, pred_img, target_img]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"Sample #{idx}")
    plt.tight_layout()
    plt.show()

# --- MAIN EVALUATION ---
def evaluate(model_path=MODEL_SAVE_PATH, num_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    dataset = RoadSkeletonDataset(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    iou_scores = []
    dice_scores = []
    mse_scores = []

    with torch.no_grad():
        for idx, (image, target) in enumerate(loader):
            if idx >= num_samples:
                break

            image = image.to(device)
            output = model(image)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5
            target_np = target.numpy()[0, 0] > 0.5

            iou = compute_iou(pred, target_np)
            dice = compute_dice(pred, target_np)
            mse = compute_mse(output, target)
            iou_scores.append(iou)
            dice_scores.append(dice)
            mse_scores.append(mse)                  
            
            show_image_triplet(image.cpu().numpy()[0, 0], pred, target_np, idx, iou, dice, mse)

    print(f"\nAverage IoU over {num_samples} samples: {np.mean(iou_scores):.4f}")
    print(f"Average Dice Coefficient over {num_samples} samples: {np.mean(dice_scores):.4f}")
    print(f"Average MSE over {num_samples} samples: {np.mean(mse_scores):.4f}")

if __name__ == '__main__':
    evaluate()