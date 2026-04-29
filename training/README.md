# Training a Tennis Court Keypoint Detector

## Pipeline overview

```
1. Label frames  →  2. Train  →  3. Drop model into main.py
```

---

## Step 1 — Label the training data  ← START HERE

### 1a. Upload frames to Roboflow

1. Go to **https://roboflow.com** and create a free account
2. Click **"Create New Project"**
   - Project type: **Keypoint Detection**
   - Project name: `tennis-court-keypoints`
3. Upload the extracted frames:
   - All images are in `training/raw_frames/` (148 frames already extracted)
   - **Also collect more frames** — search YouTube for "indoor tennis match POV"
     and extract frames with `ffmpeg -i video.mp4 -vf fps=1 extra_frames/frame%05d.jpg`
   - Aim for **at least 500 labelled frames**, ideally 1000+
4. In Project Settings → define **14 keypoints** with these exact names (order matters):
   ```
   0_far_left_doubles
   1_far_right_doubles
   2_near_left_doubles
   3_near_right_doubles
   4_far_left_singles
   5_near_left_singles
   6_far_right_singles
   7_near_right_singles
   8_far_left_service
   9_far_right_service
   10_near_left_service
   11_near_right_service
   12_far_center_T
   13_near_center_T
   ```

### 1b. Label each frame

Open `training/keypoint_reference.png` side-by-side while labelling so you know
which number corresponds to which intersection.

Key rules:
- If a keypoint is **off-screen or hidden by a player**, mark it as **occluded**
  (not invisible) — the model still learns from occluded points
- If a keypoint is **completely outside the frame** and you cannot estimate where
  it would be, skip it (leave as not labelled)
- Be consistent: always click the **centre of the line intersection**, not the
  edge of the white paint
- The **right side of the court extends off-screen** in the sarp video —
  for those frames, mark points 1, 3, 6, 7, 9, 11 as outside frame

### 1c. Export

1. In Roboflow: Generate → Export Dataset
2. Format: **COCO Keypoints** (JSON)
3. Split: 80% train / 20% val (Roboflow does this automatically)
4. Download the zip and extract:
   - `train/_annotations.coco.json`  → `training/annotations/train.json`
   - `valid/_annotations.coco.json`  → `training/annotations/val.json`
   - All images → `training/raw_frames/`  (or update `--img-dir`)

---

## Step 2 — Train the model

### Requirements

```bash
pip install torch torchvision efficientnet-pytorch
```

GPU is strongly recommended. If you don't have one locally, use **Google Colab**:
1. Upload the `training/` folder to Google Drive
2. Open Colab, mount Drive, then:
   ```python
   !python training/train.py \
       --train-ann training/annotations/train.json \
       --val-ann   training/annotations/val.json \
       --img-dir   training/raw_frames \
       --epochs 100 --batch 32
   ```
3. With a T4 GPU (free Colab): ~3 min/epoch → 100 epochs in ~5 hours

### Local GPU

```bash
python3 training/train.py \
    --train-ann training/annotations/train.json \
    --val-ann   training/annotations/val.json \
    --img-dir   training/raw_frames \
    --epochs 100 --batch 32
```

Checkpoints are saved to `training/checkpoints/best_court_kp_model.pth`.

### What good training looks like

| Epoch | PCK@5% | Notes |
|-------|--------|-------|
| 10    | ~0.40  | Model starting to learn court shape |
| 30    | ~0.65  | Baselines and corners mostly correct |
| 60    | ~0.80  | Service lines improving |
| 100   | ~0.88+ | Good generalisation |

PCK@5% above **0.85** is production-ready for this use case.

---

## Step 3 — Wire the trained model into main.py

Once training is done, open `main.py` and replace the `estimate_court_keypoints`
call in `analyze_video()` with the trained model:

```python
# Add near the top of main.py:
from training.predict import CourtKeypointPredictor
_court_predictor = CourtKeypointPredictor('training/checkpoints/best_court_kp_model.pth')

# In analyze_video(), replace the keypoints computation block with:
kp_stub = os.path.join(stub_dir, f"{video_name}_court_keypoints.pkl")
if os.path.exists(kp_stub):
    with open(kp_stub, 'rb') as f:
        court_keypoints_list = pickle.load(f)
    if isinstance(court_keypoints_list, np.ndarray):
        court_keypoints_list = [court_keypoints_list] * len(video_frames)
else:
    print(f"  Predicting court keypoints for {len(video_frames)} frames …")
    court_keypoints_list = [_court_predictor.predict(f) for f in video_frames]
    with open(kp_stub, 'wb') as f:
        pickle.dump(court_keypoints_list, f)
```

No other changes needed — the model outputs the same (28,) pixel-coordinate array
that the rest of the pipeline already expects.

---

## Troubleshooting

**CUDA out of memory** → reduce `--batch 16` or `--batch 8`

**PCK not improving past 0.5** → you likely need more diverse training data;
  add frames from broadcast matches and different camera angles

**Keypoints jumping on output video** → add temporal smoothing (already done
  in main.py with the ±5-frame rolling mean)
