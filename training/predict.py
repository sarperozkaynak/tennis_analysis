#!/usr/bin/env python3
"""
Run the trained court keypoint model on a single image or video.
Also used by main.py as a drop-in replacement for the classical CV detector.

Usage (standalone):
    python3 training/predict.py --model training/checkpoints/best_court_kp_model.pth
                                --image input_videos/tennis_sarp.mp4
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF

NUM_KP   = 14
IMG_SIZE = 224


def build_model() -> nn.Module:
    """Must match the architecture in train.py exactly."""
    net = models.efficientnet_b4(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_KP * 2),
        nn.Sigmoid(),
    )
    return net


class CourtKeypointPredictor:
    """Loads the trained model once and predicts keypoints for any frame."""

    def __init__(self, model_path: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model  = build_model()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval().to(self.device)
        print(f"Loaded court keypoint model from {model_path} on {self.device}")

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        frame : H×W×3 BGR numpy array (OpenCV format)

        Returns
        -------
        keypoints : (28,) float array — [x0, y0, x1, y1, …, x13, y13]
                    in *pixel* coordinates for the original frame size
        """
        import PIL.Image as PILImage
        h, w = frame.shape[:2]
        img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        t   = TF.normalize(TF.to_tensor(img),
                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.no_grad():
            out = self.model(t.unsqueeze(0).to(self.device))  # (1, 28)
        kp_norm = out.squeeze().cpu().numpy()   # (28,) in [0,1]
        kp_px   = kp_norm.copy()
        kp_px[0::2] *= w   # x coords
        kp_px[1::2] *= h   # y coords
        return kp_px


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path to best_court_kp_model.pth")
    p.add_argument("--image", required=True,
                   help="Image file or video file to run on")
    p.add_argument("--out",   default=None,
                   help="Output image/video path (optional)")
    args = p.parse_args()

    predictor = CourtKeypointPredictor(args.model)

    def draw_kp(frame, kp):
        for i in range(0, len(kp), 2):
            x, y = int(kp[i]), int(kp[i+1])
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(frame, str(i//2), (x+4, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame

    ext = os.path.splitext(args.image)[1].lower()
    if ext in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(args.image)
        kp    = predictor.predict(frame)
        vis   = draw_kp(frame.copy(), kp)
        out   = args.out or "predicted_keypoints.jpg"
        cv2.imwrite(out, vis)
        print(f"Saved → {out}")
    else:
        cap  = cv2.VideoCapture(args.image)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 24
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out  = args.out or "predicted_keypoints.avi"
        vw   = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"XVID"),
                               fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kp  = predictor.predict(frame)
            vw.write(draw_kp(frame, kp))
        cap.release()
        vw.release()
        print(f"Saved → {out}")
