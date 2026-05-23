"""快速驗證 JesterDataset 能正常運作。

跑法（先 conda activate jester）：
    python src/test_dataset.py
"""
from torch.utils.data import DataLoader

import config
from dataset import JesterDataset


def main():
    print("=== 建立 Dataset（各取前 64 筆快速測試）===")
    train_ds = JesterDataset(config.TRAIN_CSV, config.TRAIN_DIR, split="train", limit=64)
    val_ds   = JesterDataset(config.VAL_CSV,   config.VAL_DIR,   split="val",   limit=64)
    test_ds  = JesterDataset(config.TEST_CSV,  config.TEST_DIR,  split="val",   limit=64)
    print(f"train: {len(train_ds)} 筆 / val: {len(val_ds)} 筆 / test: {len(test_ds)} 筆")
    print(f"test 有標籤嗎: {test_ds.has_label}（預期 False）")

    print("\n=== 單筆樣本 ===")
    clip, label = train_ds[0]
    print(f"clip shape : {tuple(clip.shape)}  dtype: {clip.dtype}")
    print(f"label      : {label}  ({config.CLASSES[label]})")
    print(f"數值範圍   : [{clip.min():.3f}, {clip.max():.3f}]（已正規化，含負值正常）")
    expected = (config.T, 3, config.IMG_SIZE, config.IMG_SIZE)
    assert clip.shape == expected, f"shape 應為 {expected}"

    print("\n=== DataLoader 取一個 batch（num_workers=0）===")
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    clips, labels = next(iter(loader))
    print(f"batch clips : {tuple(clips.shape)}")
    print(f"batch labels: {labels.tolist()}")

    print("\n=== DataLoader 多進程載入（num_workers=2，驗證 Windows 沒問題）===")
    loader_mp = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    clips_mp, _ = next(iter(loader_mp))
    print(f"batch clips : {tuple(clips_mp.shape)}")

    print("\n✅ Dataset 驗證通過")


if __name__ == "__main__":   # Windows 多進程載入必須包這層
    main()
