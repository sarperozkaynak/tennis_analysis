#!/usr/bin/env python3
"""
Tennis court keypoint detector — training script.

Architecture : EfficientNet-B4 backbone + regression head
               outputs 14 (x,y) keypoints normalised to [0, 1]
Loss         : Wing Loss  (outperforms MSE/L1 for keypoint regression)
Metric       : PCK@5%  (% of keypoints within 5 % of image diagonal)
Augmentation : horizontal flip (with correct keypoint swap), colour jitter,
               blur, perspective distortion

Usage
-----
python3 training/train.py \\
    --train-ann training/annotations/train.json \\
    --val-ann   training/annotations/val.json   \\
    --img-dir   training/raw_frames             \\
    --epochs 100 --batch 32
"""

import argparse
import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_KP   = 14
IMG_SIZE = 224   # resize all frames to this before feeding the network

# When the frame is flipped left↔right these keypoint indices swap:
#   0↔1  (far-left ↔ far-right doubles corners)
#   2↔3  (near-left ↔ near-right doubles corners)
#   4↔6  (far-left singles ↔ far-right singles)
#   5↔7  (near-left singles ↔ near-right singles)
#   8↔9  (far-left service ↔ far-right service)
#   10↔11(near-left service ↔ near-right service)
#   12, 13 are centre T-marks — x coord mirrors but index stays
FLIP_PAIRS = [(0, 1), (2, 3), (4, 6), (5, 7), (8, 9), (10, 11)]


# ── Wing Loss ─────────────────────────────────────────────────────────────────

class WingLoss(nn.Module):
    """Wing Loss — designed for facial/body keypoint regression.
    Strongly penalises small errors while being robust to large ones."""

    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1.0 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.C,
        )
        if mask is not None:
            loss = loss * mask
        return loss.mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class CourtKeypointDataset(Dataset):
    """Reads COCO-style keypoint annotations exported from Roboflow / LabelStudio.

    Expected JSON structure
    -----------------------
    {
      "images": [
        {"id": 1, "file_name": "tennis_sarp_f00024.jpg", "width": 1024, "height": 576},
        ...
      ],
      "annotations": [
        {
          "image_id": 1,
          "keypoints": [x0,y0,v0, x1,y1,v1, ... x13,y13,v13],
          "num_keypoints": 14
        },
        ...
      ]
    }

    Visibility flag v:
      0 = not labelled (excluded from loss via mask)
      1 = labelled but occluded
      2 = labelled and visible
    """

    def __init__(self, ann_file: str, img_dir: str, augment: bool = True):
        with open(ann_file) as f:
            data = json.load(f)

        id_to_img = {img["id"]: img for img in data["images"]}
        self.samples = []
        for ann in data["annotations"]:
            info = id_to_img[ann["image_id"]]
            raw = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            if len(raw) != NUM_KP:
                continue
            self.samples.append({
                "path":   os.path.join(img_dir, info["file_name"]),
                "kps":    raw,           # (14, 3): x_px, y_px, visibility
                "orig_w": info["width"],
                "orig_h": info["height"],
            })

        self.augment = augment
        print(f"  {ann_file}: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    # ── augmentation helpers ──────────────────────────────────────────────────

    @staticmethod
    def _flip(img, kps: np.ndarray) -> tuple:
        img = TF.hflip(img)
        kps = kps.copy()
        kps[:, 0] = 1.0 - kps[:, 0]       # mirror x
        for a, b in FLIP_PAIRS:
            kps[[a, b]] = kps[[b, a]]      # swap paired indices
        return img, kps

    @staticmethod
    def _colour_jitter(img):
        img = TF.adjust_brightness(img, 1 + random.uniform(-0.35, 0.35))
        img = TF.adjust_contrast(img,   1 + random.uniform(-0.35, 0.35))
        img = TF.adjust_saturation(img, 1 + random.uniform(-0.35, 0.35))
        img = TF.adjust_hue(img, random.uniform(-0.08, 0.08))
        return img

    @staticmethod
    def _perspective(img, kps: np.ndarray) -> tuple:
        """Random perspective warp — simulates different camera tilts."""
        import PIL.Image as PILImage
        w, h = img.size
        margin = int(min(w, h) * 0.08)
        # Perturb corners
        def jitter(v, lo, hi):
            return max(lo, min(hi, v + random.randint(-margin, margin)))
        tl = [jitter(0, 0, margin),   jitter(0, 0, margin)]
        tr = [jitter(w, w-margin, w), jitter(0, 0, margin)]
        bl = [jitter(0, 0, margin),   jitter(h, h-margin, h)]
        br = [jitter(w, w-margin, w), jitter(h, h-margin, h)]
        src = np.float32([[0,0],[w,0],[0,h],[w,h]])
        dst = np.float32([tl, tr, bl, br])
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp image
        img_np = np.array(img)
        img_np = cv2.warpPerspective(img_np, M, (w, h))
        # Warp keypoints
        kps_px = kps.copy()
        kps_px[:, 0] *= w
        kps_px[:, 1] *= h
        pts = kps_px[:, :2].reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        kps_new = kps.copy()
        kps_new[:, 0] = warped[:, 0] / w
        kps_new[:, 1] = warped[:, 1] / h
        # Mark out-of-bounds as invisible
        oob = (kps_new[:, 0] < 0) | (kps_new[:, 0] > 1) | \
              (kps_new[:, 1] < 0) | (kps_new[:, 1] > 1)
        kps_new[oob, 2] = 0
        return PILImage.fromarray(img_np), kps_new

    def _augment(self, img, kps: np.ndarray) -> tuple:
        if random.random() < 0.5:
            img, kps = self._flip(img, kps)
        img = self._colour_jitter(img)
        if random.random() < 0.3:
            ks = random.choice([3, 5])
            img = TF.gaussian_blur(img, ks)
        if random.random() < 0.4:
            img, kps = self._perspective(img, kps)
        return img, kps

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        import PIL.Image as PILImage

        s = self.samples[idx]
        img = PILImage.open(s["path"]).convert("RGB")
        kps = s["kps"].copy()               # (14, 3): x_px, y_px, vis

        # Normalise coordinates to [0, 1]
        kps[:, 0] /= s["orig_w"]
        kps[:, 1] /= s["orig_h"]

        if self.augment:
            img, kps = self._augment(img, kps)

        img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # (28,) flat vectors
        kps_flat = torch.tensor(kps[:, :2].flatten(), dtype=torch.float32)
        vis_flat = torch.tensor(
            np.repeat((kps[:, 2] > 0).astype(np.float32), 2),
            dtype=torch.float32,
        )
        return img, kps_flat, vis_flat


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B4 backbone with a keypoint regression head.

    Using ImageNet pretrained weights speeds up convergence significantly
    even when 'training from scratch' on court keypoints — the backbone
    already understands edges, textures and shapes.
    """
    from torchvision.models import EfficientNet_B4_Weights
    weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b4(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_KP * 2),
        nn.Sigmoid(),   # clamp outputs to [0, 1]
    )
    return net


# ── Metric: PCK ───────────────────────────────────────────────────────────────

def compute_pck(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.05) -> float:
    """Percentage of Correct Keypoints within `threshold` × image diagonal."""
    dist = (pred - target).view(-1, NUM_KP, 2).pow(2).sum(-1).sqrt()
    diag = math.sqrt(2.0)   # diagonal of a 1×1 normalised image
    return (dist < threshold * diag).float().mean().item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: training on CPU will be very slow. "
              "Use Google Colab (free GPU) or a local GPU machine.")

    print("Loading datasets…")
    train_ds = CourtKeypointDataset(args.train_ann, args.img_dir, augment=True)
    val_ds   = CourtKeypointDataset(args.val_ann,   args.img_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=device.type == "cuda",
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=device.type == "cuda")

    print("Building model…")
    model = build_model(pretrained=not args.no_pretrain).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")

    criterion = WingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.out_dir, exist_ok=True)
    best_pck  = 0.0
    best_path = os.path.join(args.out_dir, "best_court_kp_model.pth")

    print(f"\nTraining for {args.epochs} epochs…")
    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, kps, vis in train_loader:
            imgs, kps, vis = imgs.to(device), kps.to(device), vis.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, kps, mask=vis)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_pck = 0.0
        with torch.no_grad():
            for imgs, kps, vis in val_loader:
                imgs, kps, vis = imgs.to(device), kps.to(device), vis.to(device)
                pred = model(imgs)
                val_loss += criterion(pred, kps, mask=vis).item()
                val_pck  += compute_pck(pred, kps)
        val_loss /= len(val_loader)
        val_pck  /= len(val_loader)

        marker = ""
        if val_pck > best_pck:
            best_pck = val_pck
            torch.save(model.state_dict(), best_path)
            marker = "  ← best"

        print(f"[{epoch:03d}/{args.epochs}]  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"PCK@5%={val_pck:.3f}{marker}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}")

    print(f"\nBest PCK@5%: {best_pck:.3f}")
    print(f"Model saved → {best_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train tennis court keypoint detector")
    p.add_argument("--train-ann",   required=True,
                   help="COCO JSON annotations for training split")
    p.add_argument("--val-ann",     required=True,
                   help="COCO JSON annotations for validation split")
    p.add_argument("--img-dir",     default="training/raw_frames",
                   help="Root directory for image paths listed in the JSON")
    p.add_argument("--out-dir",     default="training/checkpoints",
                   help="Where to save model checkpoints")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch",       type=int,   default=32,
                   help="Reduce to 16 if you get CUDA out-of-memory")
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--no-pretrain", action="store_true",
                   help="Do NOT load ImageNet weights (slower convergence, needs more data)")
    args = p.parse_args()
    train(args)
