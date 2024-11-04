"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
from torchvision.datasets import CocoDetection
from PIL import Image
from pathlib import Path
from typing import Callable, Optional


class ArmorDetection(CocoDetection):
    def __init__(self, root, annFile, transforms: Optional[Callable]):
        assert Path(root).is_dir() and Path(annFile).is_file()
        super().__init__(root, annFile)
        self._transforms = transforms
        self.prepare = DatasetPrepare()

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = Path(self.root) / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        ann = self._load_target(image_id)
        img, ann = self.prepare(img, ann)

        if self.transforms is not None:
            img, ann = self._transforms(img, ann)

        return img, ann


class DatasetPrepare:
    def __call__(self, img: Image.Image, ann: dict):
        w, h = img.size
        ann = [x for x in ann if "iscrowd" not in x or x["iscrowd"] == 0]

        quads = [x["quad"] for x in ann]
        assert all(len(q) == 8 for q in quads)
        quads = torch.as_tensor(quads, dtype=torch.float)
        quads[..., 0::2].clamp_(0, w)
        quads[..., 1::2].clamp_(0, h)
        assert (
            (quads[..., 0] <= quads[..., 4]).all()
            and (quads[..., 2] <= quads[..., 6]).all()
            and (quads[..., 1] >= quads[..., 3]).all()
            and (quads[..., 5] >= quads[..., 7]).all()
        )

        labels = [rm_category2label[x["category_id"]] for x in ann]
        labels = torch.as_tensor(labels, dtype=torch.long)

        iscrowd = torch.as_tensor([0 for _ in ann], dtype=torch.long)
        orig_size = torch.as_tensor([w, h], dtype=torch.long)
        size = torch.as_tensor([w, h], dtype=torch.long)

        ann = {
            "quads": quads,
            "labels": labels,
            "orig_size": orig_size,
            "size": size,
            "iscrowd": iscrowd,
        }

        return img, ann


rm_category2name = {
    0: "B_G",
    1: "B_1",
    2: "B_2",
    3: "B_3",
    4: "B_4",
    5: "B_5",
    6: "B_O",
    7: "B_Bs",
    8: "B_Bb",
    9: "R_G",
    10: "R_1",
    11: "R_2",
    12: "R_3",
    13: "R_4",
    14: "R_5",
    15: "R_O",
    16: "R_Bs",
    17: "R_Bb",
    18: "N_G",
    19: "N_1",
    20: "N_2",
    21: "N_3",
    22: "N_4",
    23: "N_5",
    24: "N_O",
    25: "N_Bs",
    26: "N_Bb",
    27: "P_G",
    28: "P_1",
    29: "P_2",
    30: "P_3",
    31: "P_4",
    32: "P_5",
    33: "P_O",
    34: "P_Bs",
    35: "P_Bb",
}

rm_category2label = {k: i for i, k in enumerate(rm_category2name.keys())}
rm_label2category = {v: k for k, v in rm_category2label.items()}
