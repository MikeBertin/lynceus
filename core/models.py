"""ViT classifier + attention-rollout heatmaps."""
from __future__ import annotations

import torch
import torch.nn as nn

from .morphology import CLASSES

DEFAULT_BACKBONE = "vit_small_patch16_224"


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True,
                name: str = DEFAULT_BACKBONE) -> nn.Module:
    """An ImageNet-pretrained ViT with the head swapped for our morphology classes."""
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


class AttentionRollout:
    """Capture per-block attention and combine via Abnar & Zuidema (2020) rollout.

    Used offline to render "where the transformer looked" heatmaps for the demo
    gallery. Hooks each block's ``attn_drop`` (its input is the softmaxed
    attention map) after forcing the non-fused attention path.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attns: list[torch.Tensor] = []
        self._hooks = []
        for blk in model.blocks:
            if hasattr(blk.attn, "fused_attn"):
                blk.attn.fused_attn = False
            self._hooks.append(
                blk.attn.attn_drop.register_forward_hook(self._capture))

    def _capture(self, _module, _inp, out):
        self._attns.append(out.detach())

    def remove(self):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def heatmap(self, x: torch.Tensor):
        """Return (logits, heatmap) where heatmap is (B, g, g) over patch tokens."""
        self._attns = []
        self.model.eval()
        logits = self.model(x)
        result = None
        for a in self._attns:
            a = a.mean(1)                                   # average heads -> (B, N, N)
            a = a + torch.eye(a.size(-1), device=a.device)  # residual connection
            a = a / a.sum(-1, keepdim=True)
            result = a if result is None else a @ result
        mask = result[:, 0, 1:]                             # CLS -> patch tokens
        g = int(round(mask.size(1) ** 0.5))
        return logits, mask.reshape(mask.size(0), g, g)
