"""Export the trained ViT to ONNX for in-browser inference (onnxruntime-web).

    python -m experiments.export_onnx
"""
from __future__ import annotations

import shutil

import torch

from core import config
from core.models import build_model
from core.morphology import CLASSES

FP32 = config.MODELS_DIR / "vit.onnx"
WEB_MODEL = config.WEB_MORPH_DIR / "model.onnx"


def main() -> None:
    ckpt = config.MODELS_DIR / "vit.pt"
    if not ckpt.exists():
        raise SystemExit("No models/vit.pt; run experiments.train_vit first.")

    model = build_model(pretrained=False).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    dummy = torch.randn(1, 3, config.MODEL_PX, config.MODEL_PX)
    # Legacy TorchScript exporter (dynamo=False) keeps weights in a single
    # self-contained file, essential for onnxruntime-web and quantisation.
    torch.onnx.export(
        model, dummy, FP32.as_posix(),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"Exported fp32 ONNX -> {FP32} ({FP32.stat().st_size/1e6:.1f} MB)")

    config.WEB_MORPH_DIR.mkdir(parents=True, exist_ok=True)
    # Dynamic int8 quantisation shrinks the ~85 MB ViT to ~22 MB for the browser.
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        # Quantise only MatMul (the transformer's attention/MLP, the bulk of the
        # weights). The patch-embed Conv stays fp32: onnxruntime-web's wasm
        # backend has no ConvInteger kernel, and that conv is tiny anyway.
        quantize_dynamic(FP32.as_posix(), WEB_MODEL.as_posix(),
                         weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"])
        print(f"Quantised ONNX -> {WEB_MODEL} ({WEB_MODEL.stat().st_size/1e6:.1f} MB)")
    except Exception as e:  # pragma: no cover
        print(f"Quantisation unavailable ({e!r}); shipping fp32 to web.")
        shutil.copyfile(FP32, WEB_MODEL)

    # Sanity-check the web model runs and matches class count.
    import onnxruntime as ort
    sess = ort.InferenceSession(WEB_MODEL.as_posix(),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    assert out.shape[-1] == len(CLASSES), out.shape
    print(f"Web model OK: output shape {out.shape}")


if __name__ == "__main__":
    main()
