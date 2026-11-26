# ============================================================
# SignBridge AI — Person 2: MediaPipe Holistic + BiLSTM
# 70 classes | 16 frames/video | 82 features/frame (164 w/ vel)
# CPU-friendly | Versioned cached extraction | 3 random seeds
# ============================================================

# %% 1) Imports
from pathlib import Path
import hashlib
import random

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# %% 2) Configuration
SPLIT_CSV = Path("split_date.csv")
VIDEO_ROOT = Path(".")
MODEL_PATH = "holistic_landmarker.task"

FRAME_WIDTH = 960
T = 16
NUM_CLASSES = 70

# 20 non-wrist landmarks * 2 (x, y) per hand + 2 detection flags = 82
HAND_LANDMARKS = 20
COORDS_PER_HAND = HAND_LANDMARKS * 2  # 40
COORDS = COORDS_PER_HAND * 2          # 80
F = COORDS + 2                        # 82

FEATURE_VERSION = "v2_hand_iso_wc_palm_xy"
OUT_DIR = Path("keypoints")
CACHE_DIR = OUT_DIR / "cache_v2"

SEEDS = [42, 123, 2026]
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %% 3) Video processing
def read_video_frames(path):
    """Read original video and uniformly sample T RGB frames, returning (frames, aspect_ratio)."""
    video_path = VIDEO_ROOT / str(path).replace("\\", "/")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    aspect_ratio = 16.0 / 9.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        aspect_ratio = w / float(h)

        if w > FRAME_WIDTH:
            new_h = int(h * FRAME_WIDTH / w)
            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, new_h),
                interpolation=cv2.INTER_AREA,
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise IOError(f"Empty video: {video_path}")

    indices = np.linspace(0, len(frames) - 1, T).round().astype(int)
    return [frames[i] for i in indices], aspect_ratio


def lm_to_xy(landmarks, aspect_ratio):
    """Convert MediaPipe landmarks to isotropic 2D coordinates (x scaled by aspect_ratio)."""
    xy = np.array([[lm.x * aspect_ratio, lm.y] for lm in landmarks], dtype=np.float32)
    return xy


# %% 4) Feature extraction
def normalize_hand_xy(landmarks, aspect_ratio):
    """
    Wrist-centered, palm-scaled 2D hand shape (20 non-wrist landmarks -> 40 features).
    Avoids mixing pose.z (hip-centered) with hand.z (wrist-centered) and avoids
    session-specific shoulder/pose offsets.
    """
    if not landmarks:
        return np.zeros(COORDS_PER_HAND, dtype=np.float32)

    xy = lm_to_xy(landmarks, aspect_ratio)
    centered = xy - xy[0:1]
    palm_size = float(np.linalg.norm(centered[9])) + 1e-4
    normalized = (centered[1:] / palm_size).astype(np.float32)
    return normalized.ravel()


def frame_features(res, aspect_ratio):
    """Extract 82 isotropic hand-shape features + detection flags for one frame."""
    left_hand = normalize_hand_xy(res.left_hand_landmarks, aspect_ratio)
    right_hand = normalize_hand_xy(res.right_hand_landmarks, aspect_ratio)

    flags = np.array(
        [
            float(bool(res.left_hand_landmarks)),
            float(bool(res.right_hand_landmarks)),
        ],
        dtype=np.float32,
    )

    features = np.concatenate([left_hand, right_hand, flags])
    assert features.shape == (F,)
    return features


def video_features(frames, aspect_ratio, landmarker):
    """Run HolisticLandmarker on T sampled frames and stack (T, F) features."""
    output = []
    for frame in frames:
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame),
        )
        result = landmarker.detect(image)
        output.append(frame_features(result, aspect_ratio))

    result = np.stack(output).astype(np.float32)
    assert result.shape == (T, F)
    return result


# %% 5) Cached feature extraction
def cache_filename(video_path):
    """Versioned cache key so feature changes never silently load stale .npy files."""
    key = hashlib.sha1(f"{FEATURE_VERSION}:{video_path}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.npy"


def load_and_validate_split():
    """Load split_date.csv and validate required columns and label bounds."""
    split_df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
    required_columns = {"path", "y", "split"}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {missing_columns}")

    split_df["y"] = pd.to_numeric(split_df["y"], errors="raise").astype(int)
    if split_df["y"].min() < 0 or split_df["y"].max() >= NUM_CLASSES:
        raise ValueError("Labels must be between 0 and 69.")

    for s in ["train", "val", "test"]:
        if s not in split_df["split"].values:
            raise ValueError(f"Missing split: {s}")

    return split_df


def extract_split(split_df, split, landmarker):
    """Extract and cache features for one split, saving keypoints/<split>_v2.npz."""
    subset = split_df[split_df["split"] == split].reset_index(drop=True)
    features = []
    no_hand_rates = []

    for video_path in tqdm(subset["path"].astype(str), desc=f"Extracting {split}"):
        cache_file = cache_filename(video_path)

        if cache_file.exists():
            video_feature = np.load(cache_file)
            if video_feature.shape != (T, F) or not np.isfinite(video_feature).all():
                cache_file.unlink()
                video_feature = None
        else:
            video_feature = None

        if video_feature is None:
            frames, aspect_ratio = read_video_frames(video_path)
            video_feature = video_features(frames, aspect_ratio, landmarker)
            np.save(cache_file, video_feature)

        features.append(video_feature)
        hand_flags = video_feature[:, -2:]
        no_hand = 1.0 - hand_flags.max(axis=1).mean()
        no_hand_rates.append(no_hand)

    X = np.stack(features).astype(np.float32)
    y = subset["y"].to_numpy(dtype=np.int64)

    np.savez_compressed(OUT_DIR / f"{split}_v2.npz", X=X, y=y)

    no_hand_pct = float(np.mean(no_hand_rates))
    print(f"\n{split}: X={X.shape}, y={y.shape}")
    print(f"Frames without any hand: {no_hand_pct:.2%}")
    return no_hand_pct


def run_feature_extraction(split_df):
    """Initialize MediaPipe HolisticLandmarker and extract all splits."""
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_hand_landmarks_confidence=0.3,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)

    no_hand_stats = {}
    print("\nStarting feature extraction...")
    try:
        for split in ["train", "val", "test"]:
            no_hand_stats[split] = extract_split(split_df, split, landmarker)
    finally:
        landmarker.close()
    print("\nFeature extraction completed.")
    return no_hand_stats


# %% 6) Standardization & Dataset
def compute_standardization(train_X):
    """Compute per-feature mean/std strictly over detected training hand frames."""
    mean = np.zeros(F, dtype=np.float32)
    std = np.ones(F, dtype=np.float32)

    lf_mask = train_X[:, :, -2] > 0.5
    rf_mask = train_X[:, :, -1] > 0.5

    if lf_mask.any():
        lh_vals = train_X[:, :, :COORDS_PER_HAND][lf_mask]
        mean[:COORDS_PER_HAND] = lh_vals.mean(axis=0)
        std[:COORDS_PER_HAND] = np.maximum(lh_vals.std(axis=0), 1e-3)

    if rf_mask.any():
        rh_vals = train_X[:, :, COORDS_PER_HAND:COORDS][rf_mask]
        mean[COORDS_PER_HAND:COORDS] = rh_vals.mean(axis=0)
        std[COORDS_PER_HAND:COORDS] = np.maximum(rh_vals.std(axis=0), 1e-3)

    return mean, std


def rotate_hand_2d(pts, angle_rad):
    """Rotate (T, 20, 2) wrist-centered hand landmarks in-plane by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x = pts[..., 0] * c - pts[..., 1] * s
    y = pts[..., 0] * s + pts[..., 1] * c
    return np.stack([x, y], axis=-1)


class KPDataset(Dataset):
    def __init__(self, X, y, mean, std, train=False, rot_deg=15.0):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std
        self.train = train
        self.rot_deg = rot_deg

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index].copy()
        lf = (x[:, -2:-1] > 0.5).astype(np.float32)
        rf = (x[:, -1:] > 0.5).astype(np.float32)

        if self.train:
            if self.rot_deg > 0:
                ang_l = np.deg2rad(np.random.uniform(-self.rot_deg, self.rot_deg))
                ang_r = np.deg2rad(np.random.uniform(-self.rot_deg, self.rot_deg))
                x[:, :COORDS_PER_HAND] = rotate_hand_2d(
                    x[:, :COORDS_PER_HAND].reshape(T, HAND_LANDMARKS, 2), ang_l
                ).reshape(T, COORDS_PER_HAND)
                x[:, COORDS_PER_HAND:COORDS] = rotate_hand_2d(
                    x[:, COORDS_PER_HAND:COORDS].reshape(T, HAND_LANDMARKS, 2), ang_r
                ).reshape(T, COORDS_PER_HAND)

            scale = np.random.uniform(0.9, 1.1)
            x[:, :COORDS] *= scale

            noise = np.random.normal(0.0, 0.03, (T, COORDS)).astype(np.float32)
            x[:, :COORDS_PER_HAND] += noise[:, :COORDS_PER_HAND] * lf
            x[:, COORDS_PER_HAND:COORDS] += noise[:, COORDS_PER_HAND:COORDS] * rf

            shift = np.random.randint(-2, 3)
            indices = np.clip(np.arange(T) - shift, 0, T - 1)
            x = x[indices]
            lf = lf[indices]
            rf = rf[indices]

        x = (x - self.mean) / self.std
        # Keep undetected hand coordinates strictly zero after standardization
        x[:, :COORDS_PER_HAND] *= lf
        x[:, COORDS_PER_HAND:COORDS] *= rf

        velocity = np.diff(x, axis=0, prepend=x[:1])
        x = np.concatenate([x, velocity], axis=1)

        return torch.tensor(x, dtype=torch.float32), int(self.y[index])


# %% 7) BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, in_dim=F * 2, hidden=128, num_layers=1, dropout=0.5, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        output, _ = self.lstm(x)
        output = output.mean(dim=1)
        return self.classifier(output)


# %% 8) Evaluation helpers
def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            predictions = model(x).argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += len(y)
    return correct / total


def get_predictions(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            predictions = model(x).argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_predictions)


# %% 9) Training and evaluation pipeline
def train_and_evaluate():
    print("Device:", DEVICE)
    print("Features per frame:", F, f"(with velocity: {F * 2})")

    split_df = load_and_validate_split()
    print("\nDataset split counts:")
    print(split_df["split"].value_counts())

    run_feature_extraction(split_df)

    data = {
        split: np.load(OUT_DIR / f"{split}_v2.npz")
        for split in ["train", "val", "test"]
    }
    for split in data:
        print(f"{split}: X={data[split]['X'].shape}, y={data[split]['y'].shape}")

    mean, std = compute_standardization(data["train"]["X"])
    print("\nStandardization prepared using detected training hand frames only.")

    results = []

    for seed in SEEDS:
        print("\n" + "=" * 55)
        print(f"Training with seed {seed}")
        print("=" * 55)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_loader = DataLoader(
            KPDataset(data["train"]["X"], data["train"]["y"], mean, std, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            KPDataset(data["val"]["X"], data["val"]["y"], mean, std, train=False),
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            KPDataset(data["test"]["X"], data["test"]["y"], mean, std, train=False),
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        model = BiLSTM().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val = 0.0
        patience_counter = 0
        checkpoint = f"best_lstm_v2_seed_{seed}.pt"

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for x, y in train_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = Fn.cross_entropy(logits, y, label_smoothing=0.1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y)

            scheduler.step()
            val_accuracy = evaluate_accuracy(model, val_loader)
            average_loss = total_loss / len(train_loader.dataset)

            if (epoch + 1) % 10 == 0 or val_accuracy > best_val:
                print(
                    f"Seed {seed} | Epoch {epoch + 1:3d} | "
                    f"Loss {average_loss:.4f} | Val Acc {val_accuracy:.4f}"
                )

            if val_accuracy > best_val:
                best_val = val_accuracy
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        y_true, y_pred = get_predictions(model, test_loader)
        test_accuracy = float(np.mean(y_true == y_pred))

        print(f"\nSeed {seed} | Best Val Acc: {best_val:.4f} | Test Acc: {test_accuracy:.4f}")

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (y_true, y_pred), 1)
        support = confusion.sum(axis=1)
        class_correct = np.diag(confusion)
        class_accuracy = np.divide(
            class_correct,
            support,
            out=np.zeros(NUM_CLASSES, dtype=np.float64),
            where=support != 0,
        )

        pd.DataFrame(confusion).to_csv(f"confusion_matrix_v2_seed_{seed}.csv", index=False)
        pd.DataFrame({
            "class_id": np.arange(NUM_CLASSES),
            "correct": class_correct,
            "support": support,
            "accuracy": class_accuracy,
        }).to_csv(f"per_class_accuracy_v2_seed_{seed}.csv", index=False)

        results.append({
            "seed": seed,
            "best_val_accuracy": best_val,
            "test_accuracy": test_accuracy,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("lstm_v2_results.csv", index=False)

    val_accs = results_df["best_val_accuracy"].to_numpy()
    test_accs = results_df["test_accuracy"].to_numpy()

    print("\n" + "=" * 55)
    print("FINAL RESULTS (v2)")
    print("=" * 55)
    print(results_df.to_string(index=False))
    print(f"\nVal Accuracy : {np.mean(val_accs):.4f} ± {np.std(val_accs, ddof=1):.4f}")
    print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs, ddof=1):.4f}")
    print("\nAll training and evaluation completed.")


if __name__ == "__main__":
    train_and_evaluate()
