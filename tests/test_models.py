import torch

from core.models import build_model, AttentionRollout
from core.morphology import CLASSES


def test_forward_and_attention_shapes():
    model = build_model(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)

    logits = model(x)
    assert logits.shape == (2, len(CLASSES))
    assert torch.isfinite(logits).all()

    rollout = AttentionRollout(model)
    logits2, heat = rollout.heatmap(x)
    rollout.remove()
    assert logits2.shape == (2, len(CLASSES))
    # vit_small_patch16_224 -> 14x14 patch grid
    assert heat.shape == (2, 14, 14)
    assert torch.isfinite(heat).all()
    assert (heat >= 0).all()
