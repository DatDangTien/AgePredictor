---
name: model-architectures
description: Verified layer-by-layer architecture of each ONNX model in models/
metadata:
  type: project
---

## age.onnx — VGGFace (VGG16), DeepFace

- **Backbone:** VGG16 / VGGFace (Parkhi et al. 2015, Oxford VGG group)
- **Origin:** Keras/TensorFlow → tf2onnx
- **Input:** `[1, 224, 224, 3]` (NHWC, then transposed)
- **Output:** `[1, 101]` — softmax over ages 0–100; predicted age = Σ(i × p_i)
- **Conv blocks:** 5 blocks, pattern 2-2-3-3-3, channels 64→128→256→512→512
- **FC head (as convolutions):**
  - FC6: Conv `[4096, 512, 7, 7]` — 7×7 over the 7×7 feature map → 1×1×4096
  - FC7: Conv `[4096, 4096, 1, 1]` → 1×1×4096 (face embedding layer)
  - Out: Conv `[101, 4096, 1, 1]` → softmax → reshape to [101]
- **Total nodes:** 41

## genderage.onnx — MobileNet, InsightFace/MXNet

- **Backbone:** MobileNet (depthwise-separable convolutions: DW 3×3 + PW 1×1)
- **Origin:** MXNet (`mxnet_converted_model`)
- **Input:** `[N, 3, 96, 96]` (NCHW, dynamic batch)
- **Output:** `fc1` shape `[1, 3]` = [gender_logit_0, gender_logit_1, age_value] (non-learnable Concat)
- **Shared stem:** conv_1 (16ch, stride 2) + 11 depthwise blocks → 128ch
- **Task heads fork after conv_12 (128ch):**
  - `t0` (gender): DW block ×2 → 256ch → GlobalAvgPool → Flatten → Gemm(256→2)
  - `t1` (age): DW block ×2 → 256ch → GlobalAvgPool → Flatten → Gemm(256→1)
- **Final Concat:** [2] + [1] → [3] as `fc1` (no weights, just concatenation)
- **Total nodes:** 102

## gender.onnx

Not yet analyzed. Likely same VGGFace backbone as age.onnx but 2-class output.
