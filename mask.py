import random

import numpy as np
import cv2
import torch
from torch.nn import functional as F

# Took from https://github.com/jshyunbin/inpainting_cGAN/blob/master/src/mask_generator.py
def pattern_mask(img_size, kernel_size=7, num_points=1, ratio=0.25):
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for num in range(num_points):
        coordinate = np.random.randint(img_size, size=2)
        mask[coordinate[0], coordinate[1]] = 1.0
        mask = cv2.dilate(mask, kernel, iterations=1)

    i = 0

    while np.sum(mask) < ratio * img_size * img_size:
        i += 1
        flag = True
        while flag:
            coordinate = np.random.randint(img_size, size=2)
            if mask[coordinate[0], coordinate[1]] == 1.0:
                mask2 = np.zeros((img_size, img_size), dtype=np.float32)
                mask2[coordinate[0], coordinate[1]] = 1.0
                mask2 = cv2.dilate(mask2, kernel, iterations=1)

                mask[mask + mask2 >= 1.0] = 1.0
                flag = False

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return 1.0 - mask


def make_crop_mask(crop_size, crop_kernel, sizes, device):
    crop_mask = (
        torch.from_numpy(pattern_mask(crop_size, crop_kernel))
        .view(1, 1, crop_size, crop_size)
        .to(device)
    )
    masks = []

    for size in sizes:
        if size[0] != crop_size or size[1] != crop_size:
            crop = F.interpolate(crop_mask, size=size, mode='nearest')

        else:
            crop = crop_mask

        masks.append(crop.squeeze())

    return masks


def make_mask_pyramid(selected, n_mask, sizes, device):
    masks = []

    for i in range(n_mask):
        if i == selected:
            if i < len(sizes):
                m = torch.ones(*sizes[i], device=device)

            else:
                m = torch.ones(1, device=device)

            masks.append(m)

        else:
            if i < len(sizes):
                m = torch.zeros(*sizes[i], device=device)

            else:
                m = torch.zeros(1, device=device)

            masks.append(m)

    return masks


def make_crop_mask_pyramid(selected, n_mask, crop_size, crop_kernel, sizes, device):
    masks = make_crop_mask(crop_size, crop_kernel, sizes[:selected], device)

    masks.append(torch.ones(*sizes[selected], device=device))

    for h, w in sizes[selected + 1 :]:
        masks.append(torch.zeros(h, w, device=device))

    for _ in range(n_mask - len(sizes)):
        masks.append(torch.zeros(1, device=device))

    return masks


def make_mask(
    batch_size,
    device,
    crop_prob=0.3,
    n_mask=7,
    sizes=((112, 112), (56, 56), (28, 28), (14, 14), (7, 7)),
    crop_size=56,
    crop_kernel=31,
):
    selected = torch.randint(0, n_mask, (batch_size,))

    mask_batch = []

    for sel in selected:
        if sel < len(sizes) and random.random() < crop_prob:
            masks = make_crop_mask_pyramid(
                sel, n_mask, crop_size, crop_kernel, sizes, device
            )

        else:
            masks = make_mask_pyramid(sel, n_mask, sizes, device)

        mask_batch.append(masks)

    masks_zip = []

    for masks in zip(*mask_batch):
        masks_zip.append(torch.stack(masks, 0).unsqueeze(1))

    return masks_zip
