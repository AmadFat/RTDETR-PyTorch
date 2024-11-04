"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_iou, generalized_box_iou
from .quad_ops import quad_to_xyxy

# from src.misc.dist import get_world_size, is_dist_available_and_initialized


class SetQuadCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        aux_loss_weight_dict,
        alpha=0.2,
        gamma=2.0,
        eos_coef=1e-4,
        num_classes=36,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.aux_loss_weight_dict = aux_loss_weight_dict
        assert all(l in self.supported_aux_losses for l in aux_loss_weight_dict)

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.alpha = alpha
        self.gamma = gamma

    @property
    def supported_aux_losses(self):
        return {
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "quads": self.loss_quads,
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_aux_loss(self, loss_type, outputs, targets, indices, num_quads, **kwargs):
        return self.supported_aux_losses[loss_type](
            outputs, targets, indices, num_quads, **kwargs
        )

    def forward(self, outputs: dict, targets: dict):
        aux_losses = {}
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self.matcher(outputs_without_aux, targets)
        num_quads = max(1, sum(len(t["labels"]) for t in targets))

        for loss_type in self.aux_loss_weight_dict:
            loss_dict = self.get_aux_loss(
                loss_type, outputs, targets, indices, num_quads
            )
            aux_losses.update(
                {k: loss_dict[k] * self.aux_loss_weight_dict[k] for k in loss_dict}
            )

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss_type in self.aux_loss_weight_dict:
                    l_dict = self.get_aux_loss(
                        loss_type, aux_outputs, targets, indices, num_quads
                    )
                    l_dict = {
                        k + f"_aux_{i}": v * self.aux_loss_weight_dict[k]
                        for k, v in l_dict.items()
                    }
                    aux_losses.update(l_dict)

        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs
            indices = get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_quads *= outputs["dn_meta"]["dn_num_group"]
            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                for loss_type in self.aux_loss_weight_dict:
                    l_dict = self.get_aux_loss(
                        loss_type, aux_outputs, targets, indices, num_quads
                    )
                    l_dict = {
                        k + f"_dn_{i}": v * self.aux_loss_weight_dict[k]
                        for k, v in l_dict.items()
                    }
                    aux_losses.update(l_dict)

        return aux_losses

    def loss_labels_focal(self, outputs, targets, indices, num_quads):
        assert "pred_logits" in outputs
        src_logits: torch.Tensor = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_quads

        return {"loss_labels_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_quads):
        assert "pred_quads" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_quads: torch.Tensor = outputs["pred_quads"][idx]

        tgt_quads = torch.cat([t["quads"][j] for t, (_, j) in zip(targets, indices)])
        ious, _ = box_iou(quad_to_xyxy(src_quads), quad_to_xyxy(tgt_quads))
        ious = torch.diag(ious).detach()

        src_logits: torch.Tensor = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_quads
        return {"loss_vfl": loss}

    def loss_quads(self, outputs, targets, indices, num_quads):
        assert "pred_quads" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_quads = outputs["pred_quads"][idx]
        tgt_quads = torch.cat([t["quads"][j] for t, (_, j) in zip(targets, indices)])

        loss_quad = F.l1_loss(src_quads, tgt_quads, reduction="none").sum() / num_quads
        # loss_quad = F.smooth_l1_loss(src_quads, tgt_quads, reduction="none")
        loss_giou = (
            1
            - torch.diag(
                generalized_box_iou(
                    quad_to_xyxy(src_quads),
                    quad_to_xyxy(tgt_quads),
                )
            ).sum()
            / num_quads
        )
        return {
            "loss_quad": loss_quad,
            "loss_giou": loss_giou,
        }


def get_cdn_matched_indices(dn_meta, targets):
    dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    dn_match_indices = []
    for i, num_gt in enumerate(num_gts):
        if num_gt > 0:
            gt_idx = torch.arange(num_gt, dtype=torch.long, device=device).tile(
                dn_num_group
            )
            assert len(dn_positive_idx[i]) == len(gt_idx)
            dn_match_indices.append((dn_positive_idx[i], gt_idx))
        else:
            dn_match_indices.append(
                (
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                )
            )

    return dn_match_indices
