"""
Knowledge Distillation Training Pipeline for Compact YOLO26s with Custom P2 Head.
Compresses YOLO26x (Teacher) into YOLO26s (Student) with decoupled regression/classification losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer


class DecoupledDistillationLoss(nn.Module):
    """
    Decouples bounding box coordinate distillation from classification logits:
    - Bounding Box regression (channels 0-3): Smooth L1 Loss.
    - Class Logits (channel 4+): BCEWithLogitsLoss.
    """
    def __init__(self, distill_weight=1.0):
        super().__init__()
        self.distill_weight = distill_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, s_preds, t_preds):
        total_loss = 0.0
        # Map student P3, P4, P5 to teacher P3, P4, P5
        head_map = {1: 0, 2: 1, 3: 2}

        for s_idx, t_idx in head_map.items():
            if s_idx < len(s_preds) and t_idx < len(t_preds):
                s_out = s_preds[s_idx]
                t_out = t_preds[t_idx]

                if s_out.shape == t_out.shape:
                    # Slice bounding box coordinates vs classification probabilities
                    s_box, s_cls = s_out[:, :4, ...], s_out[:, 4:, ...]
                    t_box, t_cls = t_out[:, :4, ...], t_out[:, 4:, ...]

                    box_loss = F.smooth_l1_loss(s_box, t_box)
                    cls_loss = self.bce(s_cls, t_cls.sigmoid())
                    total_loss += (box_loss + cls_loss)

        return total_loss * self.distill_weight


print("Decoupled Distillation Loss module initialized.")
