"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from .quad_ops import quad_to_xyxy


__all__ = ['RTDETRPostProcessor']


class RTDETRPostProcessor(nn.Module):
    def __init__(self, num_classes=36, use_focal_loss=True, num_top_queries=300, remap_rm_category=False):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_rm_category = remap_rm_category 
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, quads = outputs['pred_logits'], outputs['pred_quads']
        quads_pred = quad_to_xyxy(quads)
        quads_pred *= orig_target_sizes.tile(1, 4).unsqueeze(1)

        if self.use_focal_loss:
            # FIXME ???
            scores = F.sigmoid(logits)
            scores, index = scores.flatten(1).topk(self.num_top_queries)
            labels = index % self.num_classes
            index = index // self.num_classes
            quads = quads_pred.gather(1, index.unsqueeze(-1).repeat(1, 1, quads_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            quads = quads_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = scores.topk(self.num_top_queries)
                labels = labels.gather(1, index)
                quads = quads.gather(1, index.unsqueeze(-1).tile(1, 1, quads.shape[-1]))

        if self.deploy_mode:
            return labels, quads, scores

        if self.remap_rm_category:
            from data import rm_label2category
            labels = torch.as_tensor([rm_label2category[int(x.item())] for x in labels.flatten()]) \
                .to(quads.device).reshape(labels.shape)

        results = [{
            "labels": lab,
            "quads": qua,
            "scores": sco,
        } for lab, qua, sco in zip(labels, quads, scores)]
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )
