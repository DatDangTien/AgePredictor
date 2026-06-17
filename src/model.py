import torch
import torch.nn as nn

GENDER_LABELS = ['Female', 'Male']

class GenderClassifier(nn.Module):
    """AdaFace IR-50 body (pretrained, frozen) + gender classification head."""

    def __init__(self, pretrained_backbone):
        super().__init__()
        self.input_layer = pretrained_backbone.input_layer
        self.body = pretrained_backbone.body
        bn_head = pretrained_backbone.output_layer[0]   # BatchNorm2d(512), pretrained
        # 224×224 input → 4 stride-2 stages → 14×14 spatial, 512 ch → flatten to 100352
        self.head = nn.Sequential(
            bn_head,
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 2),
            nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        return self.head(x)   # [B, 2] raw logits

    def freeze_backbone(self):
        """Freeze pretrained weights (input_layer, body, headbn) AND lock their
        BatchNorm running stats (running_mean/var still drift in train()
        mode even with requires_grad=False unless the submodule is in
        eval())."""
        for module in (self.input_layer, self.body, self.head[0]):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

    def train(self, mode=True):
        # Keep frozen submodules in eval() even when the rest of the model
        # is switched to train() — overriding nn.Module.train() is the
        # standard way to do this since train()/eval() normally cascades
        # to every submodule.
        super().train(mode)
        if mode and not next(self.input_layer.parameters()).requires_grad:
            self.input_layer.eval()
            self.body.eval()
            self.head[0].eval()
        return self


