"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""

import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import generalized_box_iou
from .quad_ops import quad_to_xyxy


class QuadHungarianMatcher(nn.Module):
    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight_cost_class = weight_dict["cost_class"]
        self.weight_cost_quad = weight_dict["cost_quad"]
        self.weight_cost_giou = weight_dict["cost_giou"]
        self.use_focal_loss = use_focal_loss
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = (
            F.sigmoid(outputs["pred_logits"].flatten(0, 1))
            if self.use_focal_loss
            else outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [..., n_class]
        out_quads = outputs["pred_quads"].flatten(0, 1)  # [..., 8]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_quads = torch.cat([v["quads"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_quad = torch.cdist(out_quads, tgt_quads, p=1)
        cost_giou = -generalized_box_iou(
            quad_to_xyxy(out_quads), quad_to_xyxy(tgt_quads)
        )

        C = (
            self.weight_cost_quad * cost_quad
            + self.weight_cost_class * cost_class
            + self.weight_cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
