"""Export the trained photo-z MLP to ONNX for in-browser inference.

The exported graph folds in the feature standardisation ((x - mu) / sd) so the
browser only has to replicate ``core.photoz.featurize`` (raw band fluxes -> raw
feature vector), run the net, and softmax the logits into a redshift PDF.

    python -m experiments.export_photoz_onnx
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from core import config
from core.photoz import PhotoZNet, Z_CENTRES, PHOTOZ_BANDS, FEATURE_DIM

CKPT = config.MODELS_DIR / "photoz.pt"
FP32 = config.MODELS_DIR / "photoz.onnx"
WEB_DIR = config.WEB_DIR / "dropout"
WEB_MODEL = WEB_DIR / "photoz.onnx"


class StandardisedPhotoZ(nn.Module):
    """Wrap the net with (x - mu) / sd so ONNX accepts raw featurize() output."""

    def __init__(self, net, mu, sd):
        super().__init__()
        self.net = net
        self.register_buffer("mu", torch.tensor(mu))
        self.register_buffer("sd", torch.tensor(sd))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def main() -> None:
    if not CKPT.exists():
        raise SystemExit("No models/photoz.pt — run experiments.train_photoz first.")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)  # our own ckpt (has np arrays)
    net = PhotoZNet(in_dim=ck["feat_dim"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    model = StandardisedPhotoZ(net, ck["mu"], ck["sd"]).eval()

    dummy = torch.zeros(1, ck["feat_dim"])
    torch.onnx.export(
        model, dummy, FP32.as_posix(),
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(FP32, WEB_MODEL)
    print(f"Exported photo-z ONNX -> {WEB_MODEL} ({WEB_MODEL.stat().st_size/1e3:.0f} KB)")

    # Sidecar JSON the web app needs: band list, z grid, feature dim.
    meta = {
        "bands": list(PHOTOZ_BANDS),
        "z_centres": [round(float(z), 4) for z in Z_CENTRES],
        "feat_dim": int(ck["feat_dim"]),
    }
    (WEB_DIR / "photoz_meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    # Cross-check ONNX vs torch on random valid-ish features.
    import onnxruntime as ort
    x = torch.randn(8, ck["feat_dim"])
    with torch.no_grad():
        ref = torch.softmax(model(x), dim=1).numpy()
    sess = ort.InferenceSession(WEB_MODEL.as_posix(), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"features": x.numpy()})[0]
    got = np.exp(got - got.max(1, keepdims=True)); got /= got.sum(1, keepdims=True)
    err = float(np.abs(ref - got).max())
    print(f"ONNX vs torch max |Δpdf| = {err:.2e}  (feat_dim={ck['feat_dim']})")
    assert err < 1e-4, "ONNX/torch mismatch"
    print(f"Wrote {WEB_DIR/'photoz_meta.json'}")


if __name__ == "__main__":
    main()
