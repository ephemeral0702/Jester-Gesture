"""20BN-Jester 時序資料集。

每段影片 = 一個資料夾、內含 37 張 JPEG 影格。
JesterDataset 負責：讀 CSV → 找資料夾 → 取樣 T 張影格 → transform → 回傳 clip tensor。

訓練 transform：resize 128 → 隨機裁切 112 + 色彩抖動（同一 clip 用同一組隨機參數）
驗證 transform：resize 128 → 中心裁切 112（固定可重現）

關鍵：同一段 clip 的所有影格**必須套用同一組隨機參數**，否則會在影片裡製造人造
的「相機抖動 + 閃光燈」效果，破壞時序連續性。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

import config


# 訓練時的色彩抖動強度（±20%）
JITTER = 0.2


class ClipTrainTransform:
    """訓練：resize 128 → 隨機裁切 112 → 色彩抖動 → tensor + normalize。"""

    def __call__(self, frames):
        # 一次抽好整段 clip 共用的隨機參數
        max_off = config.RESIZE_SIZE - config.IMG_SIZE
        top = int(np.random.randint(0, max_off + 1))
        left = int(np.random.randint(0, max_off + 1))
        b = 1.0 + (np.random.random() * 2 - 1) * JITTER
        c = 1.0 + (np.random.random() * 2 - 1) * JITTER
        s = 1.0 + (np.random.random() * 2 - 1) * JITTER

        out = []
        for img in frames:
            img = F.resize(img, [config.RESIZE_SIZE, config.RESIZE_SIZE])
            img = F.crop(img, top, left, config.IMG_SIZE, config.IMG_SIZE)
            img = F.adjust_brightness(img, b)
            img = F.adjust_contrast(img, c)
            img = F.adjust_saturation(img, s)
            img = F.to_tensor(img)
            img = F.normalize(img, list(config.NORM_MEAN), list(config.NORM_STD))
            out.append(img)
        return torch.stack(out)


class ClipValTransform:
    """驗證 / 測試：resize 128 → 中心裁切 112 → tensor + normalize。固定。"""

    def __call__(self, frames):
        out = []
        for img in frames:
            img = F.resize(img, [config.RESIZE_SIZE, config.RESIZE_SIZE])
            img = F.center_crop(img, [config.IMG_SIZE, config.IMG_SIZE])
            img = F.to_tensor(img)
            img = F.normalize(img, list(config.NORM_MEAN), list(config.NORM_STD))
            out.append(img)
        return torch.stack(out)


def sample_indices(n_frames, T, train):
    """從 n_frames 張影格挑出 T 個索引（TSN 式分段取樣）。

    - train=True ：每段內隨機取一張（時序資料增強）
    - train=False：每段取中間那張（固定可重現）
    """
    if n_frames <= 0:
        return [0] * T
    seg = n_frames / T
    idx = []
    for i in range(T):
        start = i * seg
        offset = np.random.random() * seg if train else seg / 2
        idx.append(min(int(start + offset), n_frames - 1))
    return idx


class JesterDataset(Dataset):
    """回傳 (clip, label)：

    - clip ：tensor [T, C, H, W]，T 張取樣影格疊起來。
    - label：int，0–26 的 label_id；Test 無標籤時回傳 -1。
    """

    def __init__(self, csv_path, frames_dir, split="train", T=config.T, limit=None):
        self.frames_dir = Path(frames_dir)
        self.T = T
        self.train = (split == "train")
        self.transform = ClipTrainTransform() if self.train else ClipValTransform()

        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.iloc[:limit].reset_index(drop=True)
        self.df = df

        self.id_col = "id" if "id" in df.columns else "video_id"
        self.has_label = "label_id" in df.columns and df["label_id"].notna().any()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        folder = self.frames_dir / str(row[self.id_col])
        frames = sorted(folder.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"找不到影格: {folder}")

        idx = sample_indices(len(frames), self.T, self.train)
        pil_frames = [Image.open(frames[j]).convert("RGB") for j in idx]
        clip = self.transform(pil_frames)  # [T, C, H, W]

        label = int(row["label_id"]) if self.has_label else -1
        return clip, label
