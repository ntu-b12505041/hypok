"""ECGFounder-compatible backbone and dyskalemia multitask adapter.

The backbone structure follows the MIT-licensed official ECGFounder Net1D
implementation: https://github.com/PKUDigitalHealth/ECGFounder
"""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # keep configuration and non-training tests importable
    torch = None
    nn = None
    F = None


if nn is not None:

    class SamePadConv1d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            groups: int = 1,
        ) -> None:
            super().__init__()
            self.kernel_size = int(kernel_size)
            self.stride = int(stride)
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                groups=groups,
            )

        def forward(self, x):
            in_dim = x.shape[-1]
            out_dim = (in_dim + self.stride - 1) // self.stride
            padding = max(
                0, (out_dim - 1) * self.stride + self.kernel_size - in_dim
            )
            return self.conv(F.pad(x, (padding // 2, padding - padding // 2)))


    class SamePadMaxPool1d(nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            self.kernel_size = int(kernel_size)
            self.pool = nn.MaxPool1d(kernel_size)

        def forward(self, x):
            padding = max(0, self.kernel_size - 1)
            return self.pool(F.pad(x, (padding // 2, padding - padding // 2)))


    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


    class FounderBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ratio: float,
            kernel_size: int,
            stride: int,
            groups: int,
            downsample: bool,
            first_block: bool,
            use_bn: bool,
            use_dropout: bool,
        ) -> None:
            super().__init__()
            middle = int(out_channels * ratio)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.downsample = downsample
            self.first_block = first_block
            self.use_bn = use_bn
            self.use_dropout = use_dropout
            actual_stride = stride if downsample else 1

            self.bn1 = nn.BatchNorm1d(in_channels)
            self.activation1 = Swish()
            self.do1 = nn.Dropout(0.5)
            self.conv1 = SamePadConv1d(in_channels, middle, 1, 1)
            self.bn2 = nn.BatchNorm1d(middle)
            self.activation2 = Swish()
            self.do2 = nn.Dropout(0.5)
            self.conv2 = SamePadConv1d(
                middle, middle, kernel_size, actual_stride, groups
            )
            self.bn3 = nn.BatchNorm1d(middle)
            self.activation3 = Swish()
            self.do3 = nn.Dropout(0.5)
            self.conv3 = SamePadConv1d(middle, out_channels, 1, 1)
            self.se_fc1 = nn.Linear(out_channels, out_channels // 2)
            self.se_fc2 = nn.Linear(out_channels // 2, out_channels)
            self.se_activation = Swish()
            if downsample:
                self.max_pool = SamePadMaxPool1d(stride)

        def forward(self, x):
            identity = x
            out = x
            if not self.first_block:
                if self.use_bn:
                    out = self.bn1(out)
                out = self.activation1(out)
                if self.use_dropout:
                    out = self.do1(out)
            out = self.conv1(out)
            if self.use_bn:
                out = self.bn2(out)
            out = self.activation2(out)
            if self.use_dropout:
                out = self.do2(out)
            out = self.conv2(out)
            if self.use_bn:
                out = self.bn3(out)
            out = self.activation3(out)
            if self.use_dropout:
                out = self.do3(out)
            out = self.conv3(out)

            squeeze = self.se_activation(self.se_fc1(out.mean(-1)))
            squeeze = torch.sigmoid(self.se_fc2(squeeze))
            out = torch.einsum("bct,bc->bct", out, squeeze)

            if self.downsample:
                identity = self.max_pool(identity)
            if self.out_channels != self.in_channels:
                identity = identity.transpose(-1, -2)
                left = (self.out_channels - self.in_channels) // 2
                right = self.out_channels - self.in_channels - left
                identity = F.pad(identity, (left, right))
                identity = identity.transpose(-1, -2)
            return out + identity


    class FounderStage(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ratio: float,
            kernel_size: int,
            stride: int,
            groups: int,
            stage_index: int,
            blocks: int,
            use_bn: bool,
            use_dropout: bool,
        ) -> None:
            super().__init__()
            items = []
            for block_index in range(blocks):
                items.append(
                    FounderBlock(
                        in_channels if block_index == 0 else out_channels,
                        out_channels,
                        ratio,
                        kernel_size,
                        stride,
                        groups,
                        downsample=block_index == 0,
                        first_block=stage_index == 0 and block_index == 0,
                        use_bn=use_bn,
                        use_dropout=use_dropout,
                    )
                )
            # Match the official checkpoint key hierarchy: stage_list.*.block_list.*
            self.block_list = nn.ModuleList(items)

        def forward(self, x):
            for block in self.block_list:
                x = block(x)
            return x


    class ECGFounderBackbone(nn.Module):
        """Checkpoint-compatible 12-lead ECGFounder Net1D encoder."""

        def __init__(self) -> None:
            super().__init__()
            filters = [64, 160, 160, 400, 400, 1024, 1024]
            blocks = [2, 2, 2, 3, 3, 4, 4]
            self.first_conv = SamePadConv1d(12, 64, 16, 2)
            self.first_bn = nn.BatchNorm1d(64)
            self.first_activation = Swish()
            stages = []
            in_channels = 64
            for index, (out_channels, count) in enumerate(zip(filters, blocks)):
                stages.append(
                    FounderStage(
                        in_channels,
                        out_channels,
                        ratio=1,
                        kernel_size=16,
                        stride=2,
                        groups=out_channels // 16,
                        stage_index=index,
                        blocks=count,
                        use_bn=False,
                        use_dropout=False,
                    )
                )
                in_channels = out_channels
            self.stage_list = nn.ModuleList(stages)
            # Kept only for checkpoint key compatibility; removed after loading.
            self.dense = nn.Linear(1024, 150)
            self.feature_dim = 1024

        def forward_features(self, x):
            x = self.first_activation(self.first_conv(x))
            for stage in self.stage_list:
                x = stage(x)
            return x.mean(-1)

        def forward(self, x):
            return self.dense(self.forward_features(x))


    class ECGFounderDyskalemia(nn.Module):
        def __init__(
            self,
            checkpoint_path: str,
            checkpoint_sha256: str | None = None,
            dropout: float = 0.2,
            num_classes: int = 3,
            potassium_center: float = 4.3,
            potassium_scale: float = 1.0,
        ) -> None:
            super().__init__()
            self.potassium_center = float(potassium_center)
            self.potassium_scale = float(potassium_scale)
            self.backbone = ECGFounderBackbone()
            load_ecgfounder_checkpoint(
                self.backbone, checkpoint_path, checkpoint_sha256
            )
            self.backbone.dense = nn.Identity()
            feature_dim = self.backbone.feature_dim
            self.feature_dropout = nn.Dropout(dropout)
            self.classification_head = nn.Linear(feature_dim, num_classes)
            # Import here to avoid a circular module dependency.
            from .model import OrdinalHead

            self.ordinal_head = OrdinalHead(feature_dim, num_classes - 1)
            self.regression_head = nn.Linear(feature_dim, 1)

        def freeze_backbone(self) -> None:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        def unfreeze_backbone(self) -> None:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True

        def forward(self, x):
            features = self.feature_dropout(self.backbone.forward_features(x))
            potassium_z = self.regression_head(features).squeeze(-1)
            return {
                "features": features,
                "logits": self.classification_head(features),
                "ordinal_logits": self.ordinal_head(features),
                "potassium_z": potassium_z,
                "potassium": (
                    potassium_z * self.potassium_scale + self.potassium_center
                ),
            }

else:

    class ECGFounderDyskalemia:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to construct ECGFounder")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ecgfounder_checkpoint(
    backbone,
    checkpoint_path: str,
    expected_sha256: str | None = None,
) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to load ECGFounder")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ECGFounder checkpoint: {path}. "
            "Run scripts/download_ecgfounder.py first."
        )
    if expected_sha256:
        actual = _sha256(path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"ECGFounder checkpoint SHA-256 mismatch: expected "
                f"{expected_sha256}, got {actual}"
            )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if not key.removeprefix("module.").startswith("dense.")
    }
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    allowed_missing = {"dense.weight", "dense.bias"}
    real_missing = set(missing) - allowed_missing
    if real_missing or unexpected:
        raise ValueError(
            "Checkpoint is incompatible with the ECGFounder backbone. "
            f"Missing={sorted(real_missing)}, unexpected={sorted(unexpected)}"
        )
