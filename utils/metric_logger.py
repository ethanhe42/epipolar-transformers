from collections import defaultdict
from collections import deque
import os

import torch

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="  ", writer=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.writer=writer

    def update(self, is_train=True, iteration=None, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if is_train:
                self.meters[k].update(v)
            if self.writer is not None:
                if k in ['time', 'data']:
                    tag = 'unclassified'
                    continue
                elif is_train:
                    tag = 'train' 
                else:
                    tag = 'test'
                self.writer.add_scalar('/'.join([tag, k]), v, iteration)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.5f} ({:.5f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)
    
    def get_all_avg(self):
        d = {}
        for name, meter in self.meters.items():
            d[name] = meter.global_avg
        return d
