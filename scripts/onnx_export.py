import sys
import os
import torch
import onnx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import thirdparty.adaface.net as net
from thirdparty.adaface.net import Backbone
from src.model import GenderClassifier


backbone = Backbone(input_size=(224, 224), num_layers=50, mode='ir')
model = GenderClassifier(backbone)#.to(device)
ckpt = torch.load('models/gender_best.pth', map_location='cpu')
# print(ckpt['state_dict'].keys())
model.load_state_dict(ckpt['state_dict'])
model.eval()

example_inputs = (torch.randn(1, 3, 224, 224),)
torch.onnx.export(model, example_inputs, "models/adaface_ir50_ms1mv2_gender.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})

# Verify
new_onnx = onnx.load("models/adaface_ir50_ms1mv2_gender.onnx")
onnx.checker.check_model(new_onnx)