---
name: architecture-design
description: Design rationale for FC layers, conv-as-FC trick, input size flexibility, backbone swap strategies
metadata:
  type: project
---

## FC-as-Conv in VGG16 (age.onnx)

VGG16 was originally described with Dense/FC layers, but is almost always saved/deployed as convolutions:
- After 5 MaxPool(2×2) on a 224×224 input: 224/2^5 = 7×7 spatial feature maps
- FC6 = Conv(7×7, 512→4096): a 7×7 kernel over a 7×7 map produces exactly 1×1 output — mathematically identical to a Dense layer
- FC7 = Conv(1×1, 4096→4096): same as Dense(4096→4096) when spatial is already 1×1
- Head = Conv(1×1, 4096→101): same as Dense(4096→101)

**Why not 512→101 directly?** The 4096→4096→101 layers are inherited wholesale from the pretrained VGGFace model. DeepFace's age model is VGGFace (trained for face ID with 4096-dim embeddings) with only the final `predictions` layer replaced. FC6+FC7 are the **face embedding layers**, frozen from VGGFace pretraining. Only `predictions_1` (4096→101) was fine-tuned for age. The capacity of 4096→4096 is overkill for age regression, but it preserves the full pretrained representation.

## genderage.onnx head design

Branching multi-task design after the shared MobileNet stem at 128ch:
- Each task gets its own 2 extra DW blocks → 256ch (task-specific feature refinement)
- Then GAP → Flatten → single Gemm (the "1 FC after flatten" the user noted)
- The final `fc1` Concat is purely structural — [gender(2), age(1)] concatenated into [3], no weights

## Input size flexibility

**VGG-style (FC in head) — FIXED size:**
FC weights are shaped for a specific spatial resolution. Changing 224→112 makes the feature maps 3×3 (not 7×7), breaking the FC6 weight shape. You cannot directly use VGGFace pretrained weights at a different input size. Options:
1. Always resize input to 224×224 (what DeepFace does)
2. Replace FC with GAP + retrain the head (loses pretrained FC weights)

**GAP-based (ResNet, MobileNet, EfficientNet) — FLEXIBLE:**
GAP always collapses spatial dims to `[channels]` regardless of input H×W. ResNet-50 with 112×112 or 224×224 or 96×96 — all produce the same 2048-d embedding. This is the main architectural advantage of GAP-based nets for face tasks:
- ArcFace / InsightFace use ResNet-50/100 at **112×112** → tight-crop alignment standard
- No stem change needed; just resize the face crop

**Common face model input sizes and their context:**
| Size    | Typical use |
|---------|-------------|
| 64×64   | Tiny/fast models |
| 96×96   | MobileNet-based (e.g., genderage.onnx) |
| 112×112 | ArcFace, InsightFace Buffalo, most modern face rec |
| 128×128 | Some lightweight backbones |
| 192×192 | Some higher-res tasks |
| 224×224 | VGG16, ResNet (ImageNet standard), DeepFace |

The size difference is **not a new layer** — it's simply what resolution the face crop is resized to before feeding the network. Margin around the face (e.g., 1.3× padding) is baked into the alignment preprocessing, not the architecture.

**Adding a stem to change size** (less common): A strided conv or pooling block at the front can downsample a large input to match a pretrained backbone's expected feature map size. Useful when you want to accept, say, 448×448 input with a 224×224 pretrained backbone.

## Backbone swap strategies for this project

Confirmed by papers studied:
- **WideResNet** (Zagoruyko & Komodakis): wider ResNet, tested for joint age+gender in `Joint Age Estimation...WideResNet.pdf`
- **ResNet-50/100 pretrained on face (ArcFace/InsightFace)**: strong face priors at 112×112, not natively age-pretrained but transfer well
- **EfficientNet**: good accuracy/efficiency tradeoff, GAP-based, flexible input

All GAP-based backbones can replace VGGFace without the FC size constraint. The head becomes: `backbone → GAP → [optional extra layers] → FC(embedding_dim → 101)`.
