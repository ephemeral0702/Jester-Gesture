"""訓練 TSM-ResNet18 手勢辨識模型。

針對 4GB VRAM 做了三項記憶體優化：
- 小 physical batch（預設 4）。
- AMP 混合精度（fp16）：活化值記憶體約砍半，速度也較快。
- 梯度累積（accum-steps）：累積多個小 batch 的梯度再更新一次，
  「等效 batch size」= batch-size × accum-steps，兼顧訓練穩定與省記憶體。

跑法（先 conda activate jester、設 $env:PYTHONIOENCODING="utf-8"）：

    # 快速試跑：過擬合小子集，驗證 pipeline（train acc 應一路爬高）
    python src/train.py --limit 512 --epochs 15

    # 正式訓練（全量）
    python src/train.py --epochs 15

    # crash 後續訓：從 checkpoints/last.pth 接著上次的 epoch 跑
    # （--epochs 要和原本一致，cosine 學習率排程才接得準）
    python src/train.py --epochs 15 --resume

若仍 OOM：把 --batch-size 降到 2、--accum-steps 加到 8（等效 batch 不變）。
建議先設環境變數減少記憶體碎片：
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import JesterDataset
from model import TSMResNet18
from plot import plot_history


def compute_class_weights(csv_path, num_classes):
    """用訓練集的類別次數算反頻率權重 → 給 CrossEntropyLoss 處理不平衡。

    樣本少的類別權重大，模型分錯它時受罰更重。
    """
    df = pd.read_csv(csv_path)
    counts = (df["label_id"].value_counts()
              .reindex(range(num_classes), fill_value=0)
              .sort_index()
              .to_numpy())
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None,
              scaler=None, accum_steps=1):
    """跑一個 epoch。傳 optimizer 代表訓練模式，否則為評估模式。

    每個 batch 都在同一行原地刷新進度（含 ETA），知道還在跑、大概多久。
    """
    train = optimizer is not None
    model.train() if train else model.eval()
    use_amp = (device == "cuda")
    tag = "train" if train else " val "
    n_batches = len(loader)

    total, correct, loss_sum = 0, 0, 0.0
    if train:
        optimizer.zero_grad()
    t0 = time.time()

    with torch.set_grad_enabled(train):
        for i, (clips, labels) in enumerate(loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)

            if train:
                # 梯度累積：loss 先除以累積步數，累積 accum_steps 次才真正更新一次
                scaler.scale(loss / accum_steps).backward()
                if (i + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            # 進度回報：每個 batch 用 \r 在同一行原地刷新（不洗版）
            done = i + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (n_batches - done)
            print(f"\r  [{tag}] {done}/{n_batches} ({100 * done / n_batches:.0f}%) "
                  f"| 已用 {elapsed:.0f}s | 預估剩 {eta:.0f}s "
                  f"| running loss {loss_sum / total:.3f}   ",
                  end="", flush=True)

    print()  # 收尾換行，讓後面的 epoch 摘要不會蓋到進度列
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="physical batch（4GB VRAM 建議 4，OOM 再降到 2）")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="梯度累積步數；等效 batch = batch-size × accum-steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="train/val 各只取前 N 筆（快速試跑流程用）")
    parser.add_argument("--resume", action="store_true",
                        help="從 checkpoints/last.pth 接續上次的訓練")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eff_batch = args.batch_size * args.accum_steps
    print(f"裝置: {device} | physical batch {args.batch_size} "
          f"× accum {args.accum_steps} = 等效 batch {eff_batch}")

    # ── 資料 ──────────────────────────────────────────────
    train_ds = JesterDataset(config.TRAIN_CSV, config.TRAIN_DIR, "train", limit=args.limit)
    val_ds   = JesterDataset(config.VAL_CSV,   config.VAL_DIR,   "val",   limit=args.limit)
    print(f"train: {len(train_ds)} 筆 / val: {len(val_ds)} 筆")

    # 讀小 JPEG 是瓶頸 → 多 worker 平行讀；persistent_workers 避免每 epoch 重開
    # 進程；prefetch_factor 讓每個 worker 多預抓幾個 batch，餵飽 GPU。
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── 模型 / 損失 / 優化器 ───────────────────────────────
    model = TSMResNet18(pretrained=True).to(device)
    weights = compute_class_weights(config.TRAIN_CSV, config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── 斷點續訓 ───────────────────────────────────────────
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_acc = 0.0
    start_epoch = 1
    last_ckpt = config.CKPT_DIR / "last.pth"
    if args.resume:
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            history = ckpt["history"]
            best_acc = ckpt["best_acc"]
            start_epoch = ckpt["epoch"] + 1
            print(f"續訓：已完成 {ckpt['epoch']} epochs，從第 {start_epoch} epoch 接續")
        else:
            print(f"找不到 {last_ckpt}，從頭開始訓練。")

    # ── 訓練迴圈 ───────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scaler, args.accum_steps)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | "
              f"{time.time() - t0:.0f}s")

        # 記錄每 epoch 數據並即時存檔（中途中斷也保得住已跑的紀錄）
        for key, val in zip(history, (epoch, tr_loss, tr_acc, va_loss, va_acc)):
            history[key].append(val)
        (config.CKPT_DIR / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        # 每個 epoch 都重畫曲線 → 可邊訓練邊開著 training_curves.png 看進度
        plot_history(history, config.CKPT_DIR / "training_curves.png")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": va_acc},
                       config.CKPT_DIR / "best.pth")
            print(f"  -> 存檔 best.pth（val acc {va_acc:.3f}）")

        # 存完整斷點（含 optimizer / scheduler / scaler）→ crash 後 --resume 接續
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                    "history": history},
                   last_ckpt)

    print(f"\n完成。最佳 val acc: {best_acc:.3f}")


if __name__ == "__main__":   # Windows 多進程 DataLoader 必須包這層
    main()
