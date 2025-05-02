import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import convolve
from sklearn.neighbors import KDTree
import numpy as np

from train_unet import UNet, RoadSkeletonDataset, DATA_DIR, IMAGE_SIZE, MODEL_SAVE_PATH  # reuse config + model

# --- METRICS ---
def compute_valence_map(binary_skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors = convolve(binary_skeleton.astype(np.uint8), kernel, mode='constant')
    return neighbors * binary_skeleton

def match_valence_nodes(pred, gt, valence, radius=3):
    pred_coords = np.argwhere(pred == 1)
    gt_coords = np.argwhere(gt == 1)

    pred_vals = compute_valence_map(pred)
    gt_vals = compute_valence_map(gt)

    pred_k = np.argwhere(pred_vals == valence)
    gt_k = np.argwhere(gt_vals == valence)

    if len(gt_k) == 0:
        return 0, 0, 0  # no gt points of this valence

    tree = KDTree(gt_k)
    matched_gt = set()
    matched_pred = 0

    for p in pred_k:
        dist, idx = tree.query([p], k=1)
        if dist[0][0] <= radius:
            gt_match = tuple(gt_k[idx[0][0]])
            if gt_match not in matched_gt:
                matched_gt.add(gt_match)
                matched_pred += 1

    precision = matched_pred / len(pred_k) if len(pred_k) > 0 else 0
    recall = matched_pred / len(gt_k)
    return precision, recall, matched_pred

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
    valence_stats = {v: {'precision': [], 'recall': []} for v in [1, 2, 3, 4]}

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

            # --- Valence metrics ---
            for valence in [1, 2, 3, 4]:
                precision, recall, _ = match_valence_nodes(pred, target_np, valence)
                valence_stats[valence]['precision'].append(precision)
                valence_stats[valence]['recall'].append(recall)

            show_image_triplet(image.cpu().numpy()[0, 0], pred, target_np, idx, iou, dice, mse)

    print(f"\nAverage IoU over {num_samples} samples: {np.mean(iou_scores):.4f}")
    print(f"Average Dice Coefficient over {num_samples} samples: {np.mean(dice_scores):.4f}")
    print(f"Average MSE over {num_samples} samples: {np.mean(mse_scores):.4f}")
    print("\n=== Per-Valence Average Precision & Recall ===")
    for valence in [1, 2, 3, 4]:
        prec_list = valence_stats[valence]['precision']
        rec_list = valence_stats[valence]['recall']
        if prec_list:  # avoid division by zero
            avg_prec = np.mean(prec_list)
            avg_rec = np.mean(rec_list)
            print(f"Valence {valence}: Precision = {avg_prec:.3f}, Recall = {avg_rec:.3f}")
        else:
            print(f"Valence {valence}: No matches found.")

if __name__ == '__main__':
    evaluate()