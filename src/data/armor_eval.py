# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from .functional import quad_to_xywh


class ArmorEvaluator:
    def __init__(self, armor_gt):
        armor_gt = copy.deepcopy(armor_gt)
        self.armor_gt = armor_gt
        self.armor_eval = COCOeval(armor_gt, iouType="bbox")
        self.image_ids = []
        self.eval_imgs = []

    def update(self, preds: dict):
        image_ids = np.unique(list(preds.keys())).tolist()
        self.image_ids.extend(image_ids)
        results = self.prepare_boxes(preds)

        with open(os.devnull, "w") as f:
            with contextlib.redirect_stdout(f):
                armor_dt = COCO.loadRes(self.armor_gt, results) if results else COCO()
        self.armor_eval.cocoDt = armor_dt
        self.armor_eval.params.imgIds = image_ids
        image_ids, eval_imgs = evaluate(self.armor_eval)

        self.eval_imgs.append(eval_imgs)

    def accumulate(self):
        self.armor_eval.accumulate()

    def summarize(self):
        print(f"IoU metric: {"bbox"}")
        self.armor_eval.summarize()

    def prepare_boxes(self, preds: dict):
        coco_results = []
        for image_id, pred in preds.items():
            if len(pred) == 0:
                continue

            quads = pred["quads"]
            boxes = quad_to_xywh(quads).tolist()
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": labels[idx],
                        "bbox": bbox,
                        "score": scores[idx],
                    }
                    for idx, bbox in enumerate(boxes)
                ]
            )
        return coco_results


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


# import io
# from contextlib import redirect_stdout
# def evaluate(imgs):
#     with redirect_stdout(io.StringIO()):
#         imgs.evaluate()
#     return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(
            "useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType)
        )
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
