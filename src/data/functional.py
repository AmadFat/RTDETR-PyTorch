import torch
import torchvision.transforms.functional as F

from packaging import version
from typing import Optional, List
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def crop(image: torch.Tensor, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "quads" in target:
        cropped_quads: torch.Tensor = target["quads"]
        cropped_quads[..., 0::2].clamp_(min=0, max=w)
        cropped_quads[..., 1::2].clamp_(min=0, max=h)
        area = quad_area(cropped_quads)
        target["quads"] = cropped_quads
        target["area"] = area
        fields.append("quads")

        keep = area > 0
        for f in fields:
            target[f] = target[f][keep]

    return cropped_image, target


def hflip(image: torch.Tensor, target):
    flipped_image: torch.Tensor = F.hflip(image)

    h, w = flipped_image.shape[-2:]

    target = target.copy()
    if "quads" in target:
        quads: torch.Tensor = target["quads"]
        flipped_quads = quads[..., [4, 5, 6, 7, 0, 1, 2, 3]] * torch.as_tensor(
            -1, 1
        ).tile(4) + torch.as_tensor([w, 0]).tile(4)
        target["quads"] = flipped_quads

    return flipped_image, target


def resize(image: torch.Tensor, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    h_new, w_new = rescaled_image.shape[-2:]
    h_ori, w_ori = image.shape[-2:]
    ratio_height, ratio_width = h_new / h_ori, w_new / w_ori

    target = target.copy()
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "quads" in target:
        quads = target["quads"]
        scaled_quads = quads * torch.as_tensor([ratio_width, ratio_height]).tile(4)
        target["quads"] = scaled_quads

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def pad(image: torch.Tensor, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None

    target = target.copy()
    target["size"] = padded_image.shape[-2:]
    return padded_image, target


def quad_to_xywh(quads: torch.Tensor):
    assert quads.shape[-1] == 8
    xmin = quads[..., 0::2].min(-1, True).values
    ymin = quads[..., 1::2].min(-1, True).values
    xmax = quads[..., 0::2].max(-1, True).values
    ymax = quads[..., 1::2].max(-1, True).values
    w, h = (xmax - xmin), (ymax - ymin)
    return torch.cat((xmin, ymin, w, h), dim=-1)


def quad_area(quads: torch.Tensor):
    assert quads.shape[-1] == 8
    area = 0
    for i in range(4):
        j = (i + 1) % 4
        area += quads[..., 2 * i] * quads[..., 2 * j + 1]
        area -= quads[..., 2 * j] * quads[..., 2 * i + 1]
    return area.abs() * 0.5
