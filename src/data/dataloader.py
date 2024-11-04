import torch
from torch.utils.data import DataLoader


__all__ = ["DataLoader"]


def default_collate_fn(xs):
    """default collate_fn"""
    imgs = [x[0] for x in xs]
    anns = [x[1] for x in xs]
    return torch.stack(imgs), anns


class ArmorDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers=0,
        drop_last=True,
    ):
        sampler = data.RandomSampler(dataset)
        bsampler = data.BatchSampler(sampler, batch_size, drop_last)
        super().__init__(
            dataset,
            batch_sampler=bsampler,
            num_workers=num_workers,
            collate_fn=default_collate_fn,
        )

    def __repr__(self) -> str:
        fstr = self.__class__.__name__ + "(\n"
        attrs = ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]
        return fstr + "\n".join(f"  {a}={getattr(self, a)}," for a in attrs) + "\n)"
