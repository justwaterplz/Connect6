from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        h = self.bn2(h)
        return F.relu(x + h, inplace=True)

class ResEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: int = 96, blocks: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return x

class PolicyHeadSingle(nn.Module):
    def __init__(self, channels: int, board_size: int = 11):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, 1)
        self.bn   = nn.BatchNorm2d(2)
        self.board_size = board_size
        self.fc  = nn.Linear(2 * board_size * board_size, board_size * board_size)

    def forward(self, feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(feat)), inplace=True)
        x = torch.flatten(x, 1)
        logits = self.fc(x)  
        if mask is not None:
            logits = apply_action_mask(logits, mask)
        return logits

class PolicyHeadDual(nn.Module):
    def __init__(self, channels: int, board_size: int = 13):
        super().__init__()
        self.board_size = board_size
        self.pi1 = nn.Sequential(
            nn.Conv2d(channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU(inplace=True)
        )
        self.pi1_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        self.pi2_cond = nn.Conv2d(channels + 1, 2, 1) 
        self.pi2_bn   = nn.BatchNorm2d(2)
        self.pi2_fc   = nn.Linear(2 * board_size * board_size, board_size * board_size)

    def forward(
        self,
        feat: torch.Tensor,
        cond_first: Optional[torch.Tensor] = None,  
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.pi1(feat)
        h1 = torch.flatten(h1, 1)
        logits1 = self.pi1_fc(h1)
        if mask1 is not None:
            logits1 = apply_action_mask(logits1, mask1)

        if cond_first is None:
            B, _, H, W = feat.shape
            cond_first = feat.new_zeros((B, 1, H, W))
        h2 = torch.cat([feat, cond_first], dim=1)
        h2 = F.relu(self.pi2_bn(self.pi2_cond(h2)), inplace=True)
        h2 = torch.flatten(h2, 1)
        logits2 = self.pi2_fc(h2)
        if mask2 is not None:
            logits2 = apply_action_mask(logits2, mask2)
        return logits1, logits2

class ValueHead(nn.Module):
    def __init__(self, channels: int, board_size: int = 11):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.bn   = nn.BatchNorm2d(1)
        self.board_size = board_size
        input_dim = 1 * board_size * board_size
        hidden_dim = 90  
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(feat)), inplace=True)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        v = torch.tanh(self.fc2(x))  
        return v.squeeze(-1)

class SixMokuNetSingle(nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: int, board_size: int = 11):
        super().__init__()
        self.encoder = ResEncoder(in_channels, channels, blocks)
        self.policy  = PolicyHeadSingle(channels, board_size=board_size)
        self.value   = ValueHead(channels, board_size=board_size)

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat = self.encoder(obs)
        logits = self.policy(feat, mask=mask)
        v = self.value(feat)
        return {"pi": logits, "v": v}

    def loss(self, batch: Dict[str, torch.Tensor], label_smoothing: float = 0.0, value_coef: float = 1.0) -> torch.Tensor:
        out = self.forward(batch["obs"], mask=batch.get("mask"))
        p_loss = policy_ce_loss(out["pi"], batch["pi"], label_smoothing=label_smoothing)
        v_loss = F.mse_loss(out["v"], batch["z"].float())
        return p_loss + value_coef * v_loss

class SixMokuNetDual(nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: int, board_size: int = 11):
        super().__init__()
        self.encoder = ResEncoder(in_channels, channels, blocks)
        self.policy  = PolicyHeadDual(channels, board_size=board_size)
        self.value   = ValueHead(channels, board_size=board_size)

    def forward(
        self,
        obs: torch.Tensor,
        cond_first: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.encoder(obs)
        logits1, logits2 = self.policy(feat, cond_first=cond_first, mask1=mask1, mask2=mask2)
        v = self.value(feat)
        return {"pi1": logits1, "pi2": logits2, "v": v}

    def loss(self, batch: Dict[str, torch.Tensor], label_smoothing: float = 0.0, value_coef: float = 1.0) -> torch.Tensor:
        out = self.forward(
            batch["obs"],
            cond_first=batch.get("cond_first"),
            mask1=batch.get("mask1"),
            mask2=batch.get("mask2"),
        )
        p1 = policy_ce_loss(out["pi1"], batch["pi1"], label_smoothing=label_smoothing)
        p2 = policy_ce_loss(out["pi2"], batch["pi2"], label_smoothing=label_smoothing)
        v  = F.mse_loss(out["v"],  batch["z"].float())
        return p1 + p2 + value_coef * v

def build_model(kind: str = "single", blocks: int = 12, channels: int = 96, in_channels: int = 6, board_size: int = 11) -> nn.Module:
    
    if kind == "single":
        model = SixMokuNetSingle(in_channels=in_channels, channels=channels, blocks=blocks, board_size=board_size)
    elif kind == "dual":
        model = SixMokuNetDual(in_channels=in_channels, channels=channels, blocks=blocks, board_size=board_size)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    _init_weights(model)

    return model

@torch.no_grad()
def _init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def policy_ce_loss(logits: torch.Tensor, target_prob: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    B, A = logits.shape
    if label_smoothing > 0:
        u = torch.full_like(target_prob, 1.0 / A)
        target_prob = (1 - label_smoothing) * target_prob + label_smoothing * u
    logp = F.log_softmax(logits, dim=-1)
    loss = -(target_prob * logp).sum(dim=-1).mean()
    return loss

def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor, illegal_value: float = -1e9) -> torch.Tensor:
    if mask is None:
        return logits
    if mask.dtype != torch.bool:
        mask = mask.bool()
    illegal = ~mask
    logits = logits.masked_fill(illegal, illegal_value)
    return logits

if __name__ == "__main__":
    B, C, H, W = 2, 6, 13, 13
    obs = torch.randn(B, C, H, W)
    mask = torch.ones(B, H*W, dtype=torch.bool)

    ms = build_model("single", blocks=8, channels=64)
    out_s = ms(obs, mask=mask)
    print("single:", out_s["pi"].shape, out_s["v"].shape)

    md = build_model("dual", blocks=8, channels=64)
    first = torch.zeros(B, 1, H, W)
    first[:, :, 3, 7] = 1.0
    out_d = md(obs, cond_first=first, mask1=mask, mask2=mask)
    print("dual:", out_d["pi1"].shape, out_d["pi2"].shape, out_d["v"].shape)
