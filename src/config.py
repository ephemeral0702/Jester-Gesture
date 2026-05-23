"""專案集中設定 —— 路徑、超參數、類別表都放這裡，不要散落各檔。"""
from pathlib import Path

# ── 資料路徑 ──────────────────────────────────────────────
DATA_ROOT = Path(r"D:/JESTER/data")
TRAIN_DIR = DATA_ROOT / "Train"
VAL_DIR   = DATA_ROOT / "Validation"
TEST_DIR  = DATA_ROOT / "Test"
TRAIN_CSV = DATA_ROOT / "Train.csv"
VAL_CSV   = DATA_ROOT / "Validation.csv"
TEST_CSV  = DATA_ROOT / "Test.csv"

# 訓練產出（權重檔）放這裡，已在 .gitignore 排除
CKPT_DIR = Path(r"D:/JESTER/checkpoints")

# ── 影格取樣 / 影像尺寸（階段 1 資料分析的決策）─────────────
T = 16            # 每段影片取樣的影格數（影片固定 37 幀，均勻取 16 張）
IMG_SIZE = 112    # resize 後的邊長，影格統一成 IMG_SIZE x IMG_SIZE

# 影像正規化（採 ImageNet 統計值，通用）
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

# ── 類別表 ────────────────────────────────────────────────
# list 的 index 即 label_id（依字母序，與 CSV 的 label_id 一致）。
CLASSES = [
    "Doing other things",              # 0
    "Drumming Fingers",                # 1
    "No gesture",                      # 2
    "Pulling Hand In",                 # 3
    "Pulling Two Fingers In",          # 4
    "Pushing Hand Away",               # 5
    "Pushing Two Fingers Away",        # 6
    "Rolling Hand Backward",           # 7
    "Rolling Hand Forward",            # 8
    "Shaking Hand",                    # 9
    "Sliding Two Fingers Down",        # 10
    "Sliding Two Fingers Left",        # 11
    "Sliding Two Fingers Right",       # 12
    "Sliding Two Fingers Up",          # 13
    "Stop Sign",                       # 14
    "Swiping Down",                    # 15
    "Swiping Left",                    # 16
    "Swiping Right",                   # 17
    "Swiping Up",                      # 18
    "Thumb Down",                      # 19
    "Thumb Up",                        # 20
    "Turning Hand Clockwise",          # 21
    "Turning Hand Counterclockwise",   # 22
    "Zooming In With Full Hand",       # 23
    "Zooming In With Two Fingers",     # 24
    "Zooming Out With Full Hand",      # 25
    "Zooming Out With Two Fingers",    # 26
]
NUM_CLASSES = len(CLASSES)  # 27
