"""20BN-Jester 時序資料集。

每段影片 = 一個資料夾、內含 37 張 JPEG 影格。
JesterDataset 負責：讀 CSV → 找資料夾 → 取樣 T 張影格 → transform → 回傳 clip tensor。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


def build_transform():
    """每張影格套用的 transform：resize → 轉 tensor → 正規化。

    注意：**不做水平翻轉**。Jester 有 Swiping Left / Swiping Right 這種
    左右相反的手勢，水平翻轉會把標籤變錯，屬於不能用的增強。
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),                                  # [0,1], [C,H,W]
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])


def sample_indices(n_frames, T, train):
    """從 n_frames 張影格挑出 T 個索引（TSN 式分段取樣）。

    把影片平均切成 T 段，每段取一張：
    - train=True ：每段內隨機取一張 → 當作時序上的資料增強。
    - train=False：每段取中間那張 → 固定、可重現，驗證用。
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
        self.transform = build_transform()

        df = pd.read_csv(csv_path)
        if limit is not None:                    # 快速測試用：只取前 limit 筆
            df = df.iloc[:limit].reset_index(drop=True)
        self.df = df

        # Train/Validation 的影片欄叫 video_id；Test 叫 id。
        self.id_col = "id" if "id" in df.columns else "video_id"
        # Test 的 label_id 是空的 → 視為無標籤。
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
        clip = torch.stack([
            self.transform(Image.open(frames[j]).convert("RGB"))
            for j in idx
        ])  # [T, C, H, W]

        label = int(row["label_id"]) if self.has_label else -1
        return clip, label
