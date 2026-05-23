"""TSM + ResNet18 手勢辨識模型。

TSN 的純平均池化對「順序」不敏感（clip 正放 / 倒放輸出相同），
Jester 偏偏滿是成對反向手勢（Swiping Left/Right、Zooming In/Out…）。
TSM（Temporal Shift Module）在不加參數的前提下讓 2D 卷積「看得到」鄰幀，補上順序資訊。
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

import config


class TemporalShift(nn.Module):
    """時序位移：把一部分 channel 沿時間軸前後移一格。

    輸入 [B*T, C, H, W]（T 幀已攤進 batch 維）。
    - 前 1/fold_div 的 channel：往前移一幀（看「下一幀」）
    - 次 1/fold_div 的 channel：往後移一幀（看「上一幀」）
    - 其餘 channel：不動（保留當幀資訊）
    位移騰出的邊界補 0。幾乎零成本，卻給了 2D 卷積時序感受野。
    """

    def __init__(self, n_segment, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.n_segment
        b = nt // t
        x = x.view(b, t, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                  # 往前（下一幀）
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # 往後（上一幀）
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]             # 不動
        return out.view(nt, c, h, w)


def _insert_tsm(resnet, n_segment, fold_div=8):
    """在 ResNet 每個 BasicBlock 的 residual 分支前插入 TSM（blockres 式）。"""
    for layer in (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4):
        for block in layer:
            block.conv1 = nn.Sequential(
                TemporalShift(n_segment, fold_div),
                block.conv1,
            )


class TSMResNet18(nn.Module):
    """clip [B, T, C, H, W] → logits [B, num_classes]。"""

    def __init__(self, num_classes=config.NUM_CLASSES, n_segment=config.T,
                 fold_div=8, pretrained=True, dropout=0.5):
        super().__init__()
        self.n_segment = n_segment

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        _insert_tsm(backbone, n_segment, fold_div)

        self.feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()              # 拔掉原本的 1000 類分類頭
        self.backbone = backbone

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)        # 把 T 攤進 batch，每幀獨立過 CNN
        feat = self.backbone(x)           # [B*T, 512]（TSM 已在內部交換時序資訊）
        feat = feat.view(b, t, -1)        # [B, T, 512]
        feat = feat.mean(dim=1)           # 時序平均 → [B, 512]
        feat = self.dropout(feat)
        return self.fc(feat)              # [B, num_classes]


if __name__ == "__main__":   # 快速自我測試
    model = TSMResNet18(pretrained=False)   # 測試不下載權重
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {n_params / 1e6:.2f} M")

    dummy = torch.randn(2, config.T, 3, config.IMG_SIZE, config.IMG_SIZE)
    out = model(dummy)
    print(f"輸入 : {tuple(dummy.shape)}")
    print(f"輸出 : {tuple(out.shape)}（預期 (2, {config.NUM_CLASSES})）")
    assert out.shape == (2, config.NUM_CLASSES)
    print("\n模型前向通過")
