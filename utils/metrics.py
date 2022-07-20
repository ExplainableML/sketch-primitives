from collections import defaultdict
from typing import Dict
import os
import numpy as np

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter


class TbLogger(object):
    def __init__(self, log_dir: str):
        self.log_dir = os.path.expanduser(log_dir)
        self.tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    def log_metrics(self, metrics: Dict, global_step):
        metrics = metrics.copy()
        # Scalars and images
        for key, value in metrics.items():
            if 'image' in key:
                # 3D shape of [3, H, W] with channels R G B
                self.tb_writer.add_image(**value, global_step=global_step)
            elif isinstance(value, np.ndarray) and len(value) > 1:
                for n, v in enumerate(value):
                    self.tb_writer.add_scalar(f'{key}_{n}', v, global_step)
            else:
                self.tb_writer.add_scalar(key, value, global_step)

    def close(self):
        self.tb_writer.close()


class Accumulator(object):
    def __init__(self, name='accumulator'):
        self.name = name
        self.value = 0.0
        self.count = 0.0

    def add(self, value, count=1):
        if isinstance(value, torch.Tensor):
            assert value.dim() == 0
            value = value.item()
        self.value += value * count
        self.count += count

    def reset(self):
        self.value = 0.0
        self.count = 0.0

    @property
    def avg(self):
        return self.value / self.count


def log_metrics(metrics, summary_logger, global_step):
    aggregated = dict()
    for name, v in metrics.items():
        if isinstance(v, Accumulator):
            v = v.avg
        aggregated[name] = v
    summary_logger.log_metrics(metrics=aggregated, global_step=global_step)
    return aggregated


def log_images(tag, img_tensor, n_row, summary_logger, global_step):
    metrics = defaultdict(Accumulator)
    img_tensor = torchvision.utils.make_grid(img_tensor, nrow=n_row)
    if img_tensor.dtype != torch.uint8:
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        img_tensor = img_tensor.mul(255).add_(0.5).clamp_(0, 255)
        img_tensor = img_tensor.to(dtype=torch.uint8)
    metrics[tag] = {'tag': tag, 'img_tensor': img_tensor}
    log_metrics(metrics, summary_logger, global_step)
