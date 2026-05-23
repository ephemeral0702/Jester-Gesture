"""對 Validation 集評估訓練好的 TSM-ResNet18。

產出：
- Top-1 / Top-5 準確率
- 每類準確率（recall），由低到高排序
- 混淆矩陣 PNG（`checkpoints/confusion_matrix.png`）
- 最容易搞混的前 10 組配對
- 文字報告（`checkpoints/eval_report.txt`）

跑法（先 conda activate jester、設 $env:PYTHONIOENCODING="utf-8"）：

    python src/eval.py
    python src/eval.py --limit 256       # 快速 sanity check
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import JesterDataset
from model import TSMResNet18


def collect_predictions(model, loader, device):
    """跑遍 loader，回傳 (preds, labels, top5_indices)，全部為 numpy。"""
    model.eval()
    use_amp = (device == "cuda")
    preds_all, labels_all, top5_all = [], [], []
    n_batches = len(loader)

    with torch.no_grad():
        for i, (clips, labels) in enumerate(loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(clips)
            top5 = logits.topk(5, dim=1).indices
            preds_all.append(top5[:, 0].cpu())
            labels_all.append(labels.cpu())
            top5_all.append(top5.cpu())

            done = i + 1
            print(f"\r  推論 {done}/{n_batches}", end="", flush=True)
    print()
    return (torch.cat(preds_all).numpy(),
            torch.cat(labels_all).numpy(),
            torch.cat(top5_all).numpy())


def make_confusion_matrix(labels, preds, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, classes, save_path):
    """畫行歸一化的混淆矩陣（每列總和=1，即 recall 視角）。"""
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized = recall)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def top_confused_pairs(cm, classes, k=10):
    """從非對角線找出最容易搞混的前 k 組 (true_cls, pred_cls, count, ratio)。"""
    totals = cm.sum(axis=1)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat_idx = np.argsort(cm_off.flatten())[::-1][:k]
    rows, cols = np.unravel_index(flat_idx, cm.shape)
    results = []
    for r, c in zip(rows, cols):
        if cm_off[r, c] == 0:
            break
        ratio = cm_off[r, c] / max(totals[r], 1)
        results.append((classes[r], classes[c], int(cm_off[r, c]), ratio))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(config.CKPT_DIR / "best.pth"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 筆（快速 sanity check）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"裝置: {device}")

    # ── 載入模型權重 ──────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"載入 {args.ckpt}  (epoch {ckpt['epoch']}, 訓練時 val acc {ckpt['val_acc']:.4f})")

    model = TSMResNet18(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    # ── 跑 Validation ────────────────────────────────────
    val_ds = JesterDataset(config.VAL_CSV, config.VAL_DIR, "val", limit=args.limit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Validation: {len(val_ds)} 筆")

    preds, labels, top5 = collect_predictions(model, val_loader, device)

    # ── 總體準確率 ────────────────────────────────────────
    top1 = (preds == labels).mean()
    top5_acc = (top5 == labels[:, None]).any(axis=1).mean()

    # ── 混淆矩陣 + 每類準確率 + 最常搞混配對 ──────────────
    cm = make_confusion_matrix(labels, preds, config.NUM_CLASSES)
    cm_path = config.CKPT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, config.CLASSES, cm_path)

    per_class_acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    sort_idx = np.argsort(per_class_acc)   # 由低到高

    pairs = top_confused_pairs(cm, config.CLASSES, k=10)

    # ── 輸出：螢幕 + 文字檔 ───────────────────────────────
    lines = []
    lines.append(f"Top-1 accuracy: {top1:.4f}")
    lines.append(f"Top-5 accuracy: {top5_acc:.4f}")
    lines.append("")
    lines.append("=== 每類準確率（recall），由低到高 ===")
    for i in sort_idx:
        lines.append(f"  {per_class_acc[i]:.3f}  {config.CLASSES[i]:<35s} ({int(cm[i].sum())} 筆)")
    lines.append("")
    lines.append("=== 最容易搞混的前 10 組（true → predicted）===")
    for true_cls, pred_cls, cnt, ratio in pairs:
        lines.append(f"  {ratio * 100:5.1f}% ({cnt:3d})  {true_cls}  ->  {pred_cls}")

    report = "\n".join(lines)
    print("\n" + report)

    report_path = config.CKPT_DIR / "eval_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n混淆矩陣已存: {cm_path}")
    print(f"文字報告已存: {report_path}")


if __name__ == "__main__":
    main()
