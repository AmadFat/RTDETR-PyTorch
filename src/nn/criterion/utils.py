import torch
import torchvision


__all__ = [
    "format_target",
]


def format_target(tgts):
    outputs = [
        torch.cat(
            [
                torch.ones_like(tgt["labels"]) * i,
                tgt["labels"],
                tgt["quads"],
            ],
            dim=1,
        )
        for i, tgt in enumerate(tgts)
    ]
    return torch.cat(outputs, dim=0)
