import torch


def quad_area(quads: torch.Tensor):
    assert quads.shape[-1] == 8
    area = 0
    for i in range(4):
        j = (i + 1) % 4
        area += quads[..., 2 * i] * quads[..., 2 * j + 1]
        area -= quads[..., 2 * j] * quads[..., 2 * i + 1]
    return area.abs() * 0.5


def quad_to_cxcywh(quads: torch.Tensor):
    assert quads.shape[-1] == 8
    xmin = quads[..., 0::2].min(-1, True).values
    ymin = quads[..., 1::2].min(-1, True).values
    xmax = quads[..., 0::2].max(-1, True).values
    ymax = quads[..., 1::2].max(-1, True).values
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    w, h = (xmax - xmin), (ymax - ymin)
    return torch.cat((cx, cy, w, h), dim=-1)


def quad_to_xywh(quads: torch.Tensor):
    assert quads.shape[-1] == 8
    xmin = quads[..., 0::2].min(-1, True).values
    ymin = quads[..., 1::2].min(-1, True).values
    xmax = quads[..., 0::2].max(-1, True).values
    ymax = quads[..., 1::2].max(-1, True).values
    w, h = (xmax - xmin), (ymax - ymin)
    return torch.cat((xmin, ymin, w, h), dim=-1)


def quad_to_xyxy(quads: torch.Tensor):
    assert quads.shape[-1] == 8
    xmin = quads[..., 0::2].min(-1, True).values
    ymin = quads[..., 1::2].min(-1, True).values
    xmax = quads[..., 0::2].max(-1, True).values
    ymax = quads[..., 1::2].max(-1, True).values
    return torch.cat((xmin, ymin, xmax, ymax), dim=-1)
