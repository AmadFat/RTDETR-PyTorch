from src.nn import SetQuadCriterion, QuadHungarianMatcher


def rtdetr_criterion():
    matcher = QuadHungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )

    criterion = SetQuadCriterion(
        matcher=matcher,
        weight_dict={"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
        losses=["vfl", "quads"],
        alpha=0.75,
        gamma=2.0,
    )
    return criterion
