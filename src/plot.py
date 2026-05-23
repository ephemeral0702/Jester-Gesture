"""畫訓練曲線：train vs val 的 loss 與 accuracy。

用來觀察模型狀況：
- 兩條 loss 都降、val 跟得上 train → 正常收斂。
- train 持續變好但 val 卡住甚至變差（兩條岔開）→ overfitting。
- train loss 也降不下去 → underfitting / 學習率或架構問題。

train.py 訓練完會自動呼叫；也可單獨重畫：
    python src/plot.py
"""
import json

import matplotlib.pyplot as plt

import config


def plot_history(history, save_path):
    """history：dict，含 epoch / train_loss / train_acc / val_loss / val_acc 五個 list。"""
    epochs = history["epoch"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左：loss
    ax_loss.plot(epochs, history["train_loss"], "o-", label="train")
    ax_loss.plot(epochs, history["val_loss"], "s-", label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # 右：accuracy
    ax_acc.plot(epochs, history["train_acc"], "o-", label="train")
    ax_acc.plot(epochs, history["val_acc"], "s-", label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("Training curves (train vs val)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"訓練曲線已存: {save_path}")


def main():
    hist_path = config.CKPT_DIR / "history.json"
    if not hist_path.exists():
        print(f"找不到 {hist_path}，請先執行 train.py。")
        return
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    plot_history(history, config.CKPT_DIR / "training_curves.png")


if __name__ == "__main__":
    main()
