# Jester-Gesture

用 **20BN-Jester** 手勢資料集練習時序影像分類模型，最終目標部署到 PYNQ-Z2 FPGA。

## 環境

Python 3.11，conda 環境 `jester`。套件清單見 [`requirements.txt`](requirements.txt)
（PyTorch 因需特殊 index URL，依檔內註解單獨安裝）。

## 用法

```powershell
python src/train.py --epochs 15            # 訓練
python src/train.py --epochs 15 --resume   # 續訓
python src/eval.py                         # 評估 best.pth
```

## 結構

```
src/           config / dataset / model (TSM-ResNet18) / train / eval / plot
notebooks/     01_data_analysis.ipynb
data/          ← 需自行下載 20BN-Jester 並解壓到這（見下方「資料集」段）
checkpoints/   ← 跑 train.py / eval.py 自動產生（權重、history、曲線、報告），未入版控
```

## 資料集

[20BN-Jester (toxicmender, Kaggle)](https://www.kaggle.com/datasets/toxicmender/20bn-jester) ——
27 類手勢，50,420 train / 7,047 val，每段影片 37 張 100px 高 JPEG。約 22 GB，需自行下載到 `data/`。

## 成果

| Top-1 | Top-5 | 架構 | 參數 |
|-------|-------|------|------|
| **0.930** | 0.994 | TSM-ResNet18（ImageNet 預訓練）| 11.19M |

主要失分在 Turning Hand Clockwise ↔ Counterclockwise（旋轉手勢的時序解析度不足）。
