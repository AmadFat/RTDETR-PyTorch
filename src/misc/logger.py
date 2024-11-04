"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/misc.py
Mostly copy-paste from torchvision references.
"""

import time
import datetime
import numpy as np
from torch import Tensor
from typing import Iterable
from collections import defaultdict, deque


class SmoothedValue:
    def __init__(self, window_size=20, fmt="{value:.4f} ({global_avg:.4f})") -> None:
        self.fmt = fmt
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, cnt=1):
        self.deque.extend(value for _ in range(cnt))
        self.count += cnt
        self.total += value * cnt

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def avg(self):
        return np.mean(self.deque)

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return np.max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(
        self,
        iterable: Iterable,
        print_freq=100,
        header="",
        delimiter=" ",
    ) -> None:
        self.logger = defaultdict(SmoothedValue)
        self.iterable = iterable
        self.print_freq = print_freq
        self.header = header
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            v = v.item() if isinstance(v, Tensor) else v
            assert isinstance(v, (float, int, str))
            self.logger[k].update(v)

    def __getattr__(self, attr):
        if attr in self.logger:
            return self.logger[attr]
        elif attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"Attribute {attr} not found")

    def __str__(self):
        return self.delimiter.join(f"{k, str(v)}" for k, v in self.logger.items())

    def __setattr__(self, name, metric):
        assert isinstance(metric, SmoothedValue)
        self.logger[name] = metric

    def log_every(self):
        i, t = 0, time.time()
        iter_time = SmoothedValue(window_size=self.print_freq, fmt="{avg:.4f}")

        for obj in self.iterable:
            yield obj
            iter_time.update(time.time() - t)
            i, t = i + 1, time.time()

            if i == 1 or i % self.print_freq == 0 or i == len(self.iterable) - 1:
                eta_sec = iter_time.global_avg * (len(self.iterable) - i)
                print(
                    self.delimiter.join(
                        self.header,
                        f"[{i}/{len(self.iterable)}]",
                        f"eta: {datetime.timedelta(seconds=int(eta_sec))}",
                        f"time: {datetime.timedelta(seconds=int(np.sum(iter_time)))}",
                        str(self),
                    )
                )

        print(
            self.header,
            f"total time: {datetime.timedelta(seconds=int(iter_time.total))}",
            f"speed: {iter_time.total / len(self.iterable):.4f} s/it",
        )
