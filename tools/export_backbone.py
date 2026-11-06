"""Export the image feature extractor used by the mouth model to models/backbone.onnx.

  .venv\\Scripts\\python.exe tools/export_backbone.py

MobileNetV3-Large pretrained on ImageNet (torchvision, BSD license), cut before its classifier: an image goes in,
960 numbers describing what's in it come out. Small personal classifiers (mouth_model.py) are trained
on those numbers in seconds, without PyTorch, so training also works in the packaged app.
Input: N x 3 x 160 x 160 RGB in 0..1 (ImageNet normalisation is inside the model).
"""
import torch
import torchvision

OUT = "models/backbone.onnx"
SIZE = 160


class Features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.features, self.pool = m.features, m.avgpool
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.pool(self.features((x - self.mean) / self.std)).flatten(1)


def main():
    model = Features().eval()
    x = torch.rand(2, 3, SIZE, SIZE)
    torch.onnx.export(model, x, OUT, input_names=["image"], output_names=["features"], opset_version=17,
                      dynamic_axes={"image": {0: "n"}, "features": {0: "n"}}, dynamo=False)
    import numpy as np
    import onnxruntime as ort
    ref = model(x).detach().numpy()
    got = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    print(f"exported {OUT}: {got.shape[1]} features, max difference vs PyTorch {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main()
