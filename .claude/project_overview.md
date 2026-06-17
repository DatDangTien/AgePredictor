---
name: project-overview
description: AgePredictor project goals, current models, and research direction
metadata:
  type: project
---

AgePredictor is a research/engineering project for age and gender estimation from face images. The project is currently studying backbone architectures and resolution effects, guided by papers in `paper/`.

**Current models in `models/`:**
- `age.onnx` — DeepFace age model, VGGFace (VGG16) backbone, 224×224 input, 101-class softmax output (ages 0–100)
- `gender.onnx` — DeepFace gender model (likely same VGGFace backbone)
- `genderage.onnx` — InsightFace/MXNet joint model, MobileNet backbone, 96×96 input, dual-head (gender=2, age=1), outputs concatenated as [3]
- `face_det_lite-onnx-w8a8` — lightweight face detector (quantized w8a8)

**Research direction:**
Exploring swapping the VGGFace backbone with more modern architectures (ResNet, WideResNet, ArcFace-pretrained models) and studying the effect of input resolution on accuracy.

**Published papers collected as reference (`paper/`):**
- `Age and Gender Prediction using Deep CNNs and Transfer Learning.pdf`
- `Impact of Image Resolution on Age Estimation with DeepFace and InsightFace.pdf`
- `Joint Age Estimation and Gender Classification of Asian Faces Using WideResNet.pdf`

These are third-party published academic papers used as study material and guidelines — not authored by the user.

**Why:** Academic/research project studying state-of-art age estimation pipelines.
**How to apply:** Suggest backbone alternatives grounded in the papers already studied. Respect that VGGFace is the baseline, MobileNet variant is the lightweight baseline.
