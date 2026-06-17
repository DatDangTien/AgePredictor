---
name: research-papers
description: PDFs in paper/ and their relevance to backbone and resolution decisions
metadata:
  type: reference
---

All papers are in `/home/ubuntu/tiendat/AgePredictor/paper/`. These are **published academic papers** collected as reference/study material, not authored by the user. They guide architectural and methodology decisions in the project.

1. **Age and Gender Prediction using Deep CNNs and Transfer Learning.pdf**
   — Covers transfer learning from pretrained CNN backbones (likely VGG/ResNet variants) for age+gender. Baseline methodology reference.

2. **Impact of Image Resolution on Age Estimation with DeepFace and InsightFace.pdf**
   — Directly benchmarks resolution effects (likely 224→lower) on DeepFace (VGGFace) and InsightFace (MobileNet/ResNet). Key reference for understanding the 224 vs 96/112 tradeoffs in this project's models.

3. **Joint Age Estimation and Gender Classification of Asian Faces Using WideResNet.pdf**
   — Tests WideResNet as backbone for joint age+gender (multi-task). Supports the multi-head design seen in genderage.onnx. WideResNet is a strong candidate backbone to experiment with.
