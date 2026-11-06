"""Structure-Preserving Modality-Aware Erasing for unnormalized training images.

This is the fixed SP-MAE V1.1 augmentation, not a model module.  It borrows
the random rectangle idea from Random Erasing; the soft, per-image mean blend
is specific to this experiment.
"""

import math
import random

import torch


class SPMAE:
    def __init__(self, cfg):
        self.prob = cfg.PROB
        self.min_area = cfg.MIN_AREA
        self.max_area = cfg.MAX_AREA
        self.min_aspect = cfg.MIN_ASPECT
        self.max_aspect = cfg.MAX_ASPECT
        self.min_strength = cfg.MIN_STRENGTH
        self.max_strength = cfg.MAX_STRENGTH
        self.edge_fraction = cfg.EDGE_FRACTION
        if not (0 <= self.prob <= 1 and 0 < self.min_area <= self.max_area <= 1):
            raise ValueError("SP_MAE probability or area range is invalid")
        if not (0 < self.min_aspect <= self.max_aspect):
            raise ValueError("SP_MAE aspect range is invalid")
        if not (0 <= self.min_strength <= self.max_strength <= 1):
            raise ValueError("SP_MAE strength range is invalid")
        if not (0 < self.edge_fraction < 0.5):
            raise ValueError("SP_MAE.EDGE_FRACTION must be in (0, 0.5)")
        if cfg.ASPECT_DISTRIBUTION != "log_uniform":
            raise ValueError("SP_MAE supports log_uniform aspect sampling only")
        if cfg.FILL_TARGET != "image_channel_mean":
            raise ValueError("SP_MAE supports image_channel_mean filling only")
        if cfg.MAX_REGIONS != 1:
            raise ValueError("SP_MAE V1.1 supports exactly one region")

    def __call__(self, image):
        if image.ndim != 3 or image.shape[0] != 3 or not torch.is_floating_point(image):
            raise ValueError("SP-MAE expects a floating [3,H,W] tensor from ToTensor")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("SP-MAE expects finite, unnormalized pixels in [0,1]")
        if random.random() >= self.prob:
            return image

        _, height, width = image.shape
        area = height * width
        for _ in range(10):
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect = math.exp(random.uniform(math.log(self.min_aspect),
                                             math.log(self.max_aspect)))  # width / height
            region_height = round(math.sqrt(target_area / aspect))
            region_width = round(math.sqrt(target_area * aspect))
            actual_area = region_height * region_width / area
            if (region_height < 3 or region_width < 3 or
                    region_height > height or region_width > width or
                    not self.min_area <= actual_area <= self.max_area):
                continue
            top = random.randint(0, height - region_height)
            left = random.randint(0, width - region_width)
            break
        else:
            return image

        # Each axis rises linearly from zero at the rectangle boundary.  The
        # product gives a shared 2-D mask with zero outside the rectangle.
        y = torch.arange(region_height, device=image.device, dtype=image.dtype)
        x = torch.arange(region_width, device=image.device, dtype=image.dtype)
        y = torch.minimum(y, region_height - 1 - y)
        x = torch.minimum(x, region_width - 1 - x)
        y = (y / (self.edge_fraction * region_height)).clamp(max=1)
        x = (x / (self.edge_fraction * region_width)).clamp(max=1)
        mask = y[:, None] * x[None, :]

        channel_mean = image.mean(dim=(1, 2), keepdim=True)
        strength = random.uniform(self.min_strength, self.max_strength)
        original_region = image[:, top:top + region_height, left:left + region_width]
        result = image.clone()
        result[:, top:top + region_height, left:left + region_width] = (
            original_region + strength * mask.unsqueeze(0) * (channel_mean - original_region)
        )
        return result
