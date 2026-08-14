from __future__ import annotations

from typing import Sequence


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to construct the ECG model") from exc
    return torch, nn, functional


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # permit non-training utilities/tests without PyTorch installed
    torch = None
    nn = None
    F = None


if nn is not None:

    class SqueezeExcitation1D(nn.Module):
        def __init__(self, channels: int, reduction: int = 16) -> None:
            super().__init__()
            hidden = max(8, channels // reduction)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.net = nn.Sequential(
                nn.Conv1d(channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.net(self.pool(x))


    class SEResidualBlock1D(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            kernel_size: int,
            se_reduction: int,
            dropout: float,
        ) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            )
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.se = SqueezeExcitation1D(out_channels, se_reduction)
            self.dropout = nn.Dropout(dropout)
            self.projection = (
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                if stride != 1 or in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x):
            residual = self.projection(x)
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
            x = self.dropout(x)
            x = self.bn2(self.conv2(x))
            x = self.se(x)
            return F.relu(x + residual, inplace=True)


    class OrdinalHead(nn.Module):
        """Monotonic cumulative logits P(y >= 1), P(y >= 2)."""

        def __init__(self, feature_dim: int, num_thresholds: int = 2) -> None:
            super().__init__()
            self.score = nn.Linear(feature_dim, 1)
            self.offset = nn.Parameter(torch.tensor(-0.5))
            self.raw_deltas = nn.Parameter(torch.zeros(num_thresholds))

        def forward(self, features):
            deltas = F.softplus(self.raw_deltas)
            cuts = self.offset + torch.cumsum(deltas, dim=0)
            return self.score(features) - cuts.unsqueeze(0)


    class SEResNet1DMultitask(nn.Module):
        def __init__(
            self,
            input_leads: int = 12,
            base_channels: int = 64,
            stage_blocks: Sequence[int] = (2, 2, 2, 2),
            kernel_size: int = 7,
            dropout: float = 0.2,
            se_reduction: int = 16,
            num_classes: int = 3,
            potassium_center: float = 4.3,
            potassium_scale: float = 1.0,
        ) -> None:
            super().__init__()
            self.potassium_center = float(potassium_center)
            self.potassium_scale = float(potassium_scale)
            stem_kernel = 15
            self.stem = nn.Sequential(
                nn.Conv1d(
                    input_leads,
                    base_channels,
                    stem_kernel,
                    stride=2,
                    padding=stem_kernel // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
            channels = [base_channels * (2**idx) for idx in range(len(stage_blocks))]
            stages = []
            current = base_channels
            for stage_idx, (out_channels, blocks) in enumerate(zip(channels, stage_blocks)):
                for block_idx in range(blocks):
                    stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                    stages.append(
                        SEResidualBlock1D(
                            current,
                            out_channels,
                            stride,
                            kernel_size,
                            se_reduction,
                            dropout,
                        )
                    )
                    current = out_channels
            self.backbone = nn.Sequential(*stages)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.feature_dropout = nn.Dropout(dropout)
            self.classification_head = nn.Linear(current, num_classes)
            self.ordinal_head = OrdinalHead(current, num_classes - 1)
            self.regression_head = nn.Linear(current, 1)
            self._initialize()

        def _initialize(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x):
            x = self.stem(x)
            x = self.backbone(x)
            features = self.feature_dropout(self.pool(x).squeeze(-1))
            potassium_z = self.regression_head(features).squeeze(-1)
            potassium = potassium_z * self.potassium_scale + self.potassium_center
            return {
                "features": features,
                "logits": self.classification_head(features),
                "ordinal_logits": self.ordinal_head(features),
                "potassium_z": potassium_z,
                "potassium": potassium,
            }


    def _three_class_logits(binary_logits):
        """Map independent HypoK/HyperK evidence to ordered three-class logits."""
        hypok = binary_logits[:, 0]
        hyperk = binary_logits[:, 1]
        normok = -0.5 * (hypok + hyperk)
        return torch.stack((hypok, normok, hyperk), dim=1)


    class SEResNet1DDualBinary(SEResNet1DMultitask):
        """Ablation model: the baseline encoder with separate low/high K heads."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            feature_dim = self.classification_head.in_features
            self.classification_head = nn.Identity()
            self.binary_head = nn.Linear(feature_dim, 2)
            nn.init.xavier_uniform_(self.binary_head.weight)
            nn.init.zeros_(self.binary_head.bias)

        def forward(self, x):
            x = self.stem(x)
            x = self.backbone(x)
            features = self.feature_dropout(self.pool(x).squeeze(-1))
            binary_logits = self.binary_head(features)
            potassium_z = self.regression_head(features).squeeze(-1)
            return {
                "features": features,
                "logits": _three_class_logits(binary_logits),
                "binary_logits": binary_logits,
                "ordinal_logits": self.ordinal_head(features),
                "potassium_z": potassium_z,
                "potassium": (
                    potassium_z * self.potassium_scale + self.potassium_center
                ),
            }


    class MultiScalePerLeadStem(nn.Module):
        """Apply shared short/medium/long temporal filters to every ECG lead."""

        def __init__(
            self,
            output_channels: int,
            branch_channels: int,
            kernel_sizes: Sequence[int],
        ) -> None:
            super().__init__()
            if not kernel_sizes or any(int(kernel) % 2 == 0 for kernel in kernel_sizes):
                raise ValueError("stem_kernel_sizes must contain positive odd integers")
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            1,
                            branch_channels,
                            int(kernel),
                            stride=2,
                            padding=int(kernel) // 2,
                            bias=False,
                        ),
                        nn.BatchNorm1d(branch_channels),
                        nn.GELU(),
                    )
                    for kernel in kernel_sizes
                ]
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(
                    branch_channels * len(kernel_sizes),
                    output_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )

        def forward(self, x):
            return self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


    class TemporalAttentionPool(nn.Module):
        def __init__(self, channels: int, hidden: int) -> None:
            super().__init__()
            self.score = nn.Sequential(
                nn.Conv1d(channels, hidden, kernel_size=1),
                nn.Tanh(),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )

        def forward(self, x):
            attention = torch.softmax(self.score(x), dim=-1)
            return torch.sum(x * attention, dim=-1), attention.squeeze(1)


    class KMorphNetV2(nn.Module):
        """Per-lead multi-scale morphology encoder with two dyskalemia experts."""

        def __init__(
            self,
            input_leads: int = 12,
            base_channels: int = 32,
            stage_blocks: Sequence[int] = (3, 4, 6, 3),
            kernel_size: int = 7,
            stem_kernel_sizes: Sequence[int] = (7, 15, 31),
            stem_branch_channels: int = 16,
            embedding_dim: int = 256,
            transformer_layers: int = 2,
            attention_heads: int = 4,
            transformer_ff_dim: int = 512,
            temporal_attention_hidden: int = 64,
            dropout: float = 0.1,
            se_reduction: int = 16,
            num_classes: int = 3,
            potassium_center: float = 4.3,
            potassium_scale: float = 1.0,
        ) -> None:
            super().__init__()
            if num_classes != 3:
                raise ValueError("KMorphNetV2 currently requires exactly three classes")
            if embedding_dim % attention_heads != 0:
                raise ValueError("embedding_dim must be divisible by attention_heads")
            self.input_leads = int(input_leads)
            self.potassium_center = float(potassium_center)
            self.potassium_scale = float(potassium_scale)
            self.stem = MultiScalePerLeadStem(
                base_channels,
                stem_branch_channels,
                stem_kernel_sizes,
            )
            channels = [base_channels * (2**idx) for idx in range(len(stage_blocks))]
            stages = []
            current = base_channels
            for stage_idx, (out_channels, blocks) in enumerate(zip(channels, stage_blocks)):
                for block_idx in range(int(blocks)):
                    stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                    stages.append(
                        SEResidualBlock1D(
                            current,
                            out_channels,
                            stride,
                            kernel_size,
                            se_reduction,
                            dropout,
                        )
                    )
                    current = out_channels
            self.backbone = nn.Sequential(*stages)
            self.temporal_pool = TemporalAttentionPool(
                current, temporal_attention_hidden
            )
            self.lead_projection = nn.Linear(current, embedding_dim)
            self.lead_embedding = nn.Parameter(
                torch.zeros(1, self.input_leads, embedding_dim)
            )
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.lead_transformer = nn.TransformerEncoder(
                transformer_layer, num_layers=transformer_layers
            )
            attention_hidden = max(16, embedding_dim // 4)
            self.hypok_lead_score = nn.Sequential(
                nn.Linear(embedding_dim, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, 1),
            )
            self.hyperk_lead_score = nn.Sequential(
                nn.Linear(embedding_dim, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, 1),
            )
            self.hypok_head = nn.Linear(embedding_dim, 1)
            self.hyperk_head = nn.Linear(embedding_dim, 1)
            self.per_lead_head = nn.Linear(embedding_dim, 2)
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.ordinal_head = OrdinalHead(embedding_dim, num_classes - 1)
            self.regression_head = nn.Linear(embedding_dim, 1)
            self._initialize()

        def _initialize(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.normal_(self.lead_embedding, std=0.02)

        def forward(self, x):
            if x.ndim != 3 or x.shape[1] != self.input_leads:
                raise ValueError(
                    f"Expected [batch, {self.input_leads}, time], got {tuple(x.shape)}"
                )
            batch_size, leads, samples = x.shape
            x = x.reshape(batch_size * leads, 1, samples)
            x = self.backbone(self.stem(x))
            lead_features, temporal_attention = self.temporal_pool(x)
            lead_features = self.lead_projection(lead_features).reshape(
                batch_size, leads, -1
            )
            lead_features = self.lead_transformer(
                lead_features + self.lead_embedding
            )
            hypok_attention = torch.softmax(
                self.hypok_lead_score(lead_features).squeeze(-1), dim=1
            )
            hyperk_attention = torch.softmax(
                self.hyperk_lead_score(lead_features).squeeze(-1), dim=1
            )
            hypok_features = torch.sum(
                lead_features * hypok_attention.unsqueeze(-1), dim=1
            )
            hyperk_features = torch.sum(
                lead_features * hyperk_attention.unsqueeze(-1), dim=1
            )
            binary_logits = torch.cat(
                (
                    self.hypok_head(hypok_features),
                    self.hyperk_head(hyperk_features),
                ),
                dim=1,
            )
            features = self.fusion(
                torch.cat((hypok_features, hyperk_features), dim=1)
            )
            potassium_z = self.regression_head(features).squeeze(-1)
            return {
                "features": features,
                "logits": _three_class_logits(binary_logits),
                "binary_logits": binary_logits,
                "per_lead_binary_logits": self.per_lead_head(lead_features),
                "ordinal_logits": self.ordinal_head(features),
                "potassium_z": potassium_z,
                "potassium": (
                    potassium_z * self.potassium_scale + self.potassium_center
                ),
                "temporal_attention": temporal_attention.reshape(
                    batch_size, leads, -1
                ),
                "hypok_lead_attention": hypok_attention,
                "hyperk_lead_attention": hyperk_attention,
            }

else:

    class SEResNet1DMultitask:  # pragma: no cover - informative fallback
        def __init__(self, *args, **kwargs) -> None:
            _torch_modules()

    class SEResNet1DDualBinary(SEResNet1DMultitask):
        pass

    class KMorphNetV2(SEResNet1DMultitask):
        pass


def build_model(config: dict):
    section = dict(config["model"])
    name = section.pop("name")
    pretrained = section.pop("pretrained_checkpoint", None)
    section.pop("freeze_backbone_epochs", None)
    section.pop("backbone_learning_rate", None)
    section.pop("head_learning_rate", None)
    if name == "se_resnet1d_multitask":
        section.pop("checkpoint_path", None)
        section.pop("checkpoint_sha256", None)
        model = SEResNet1DMultitask(**section)
    elif name == "se_resnet1d_dual_binary":
        section.pop("checkpoint_path", None)
        section.pop("checkpoint_sha256", None)
        model = SEResNet1DDualBinary(**section)
    elif name == "k_morphnet_v2":
        section.pop("checkpoint_path", None)
        section.pop("checkpoint_sha256", None)
        model = KMorphNetV2(**section)
    elif name == "ecgfounder_multitask":
        from .ecgfounder import ECGFounderDyskalemia

        # Architecture parameters are fixed to remain checkpoint-compatible.
        allowed = {
            "checkpoint_path",
            "checkpoint_sha256",
            "dropout",
            "num_classes",
            "potassium_center",
            "potassium_scale",
        }
        unexpected_config = set(section) - allowed
        if unexpected_config:
            raise ValueError(
                f"Unsupported ECGFounder model options: {sorted(unexpected_config)}"
            )
        model = ECGFounderDyskalemia(**section)
    else:
        raise ValueError(f"Unknown model: {name}")
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected pretrained keys: {unexpected}")
        model.pretrained_missing_keys = missing
    return model
