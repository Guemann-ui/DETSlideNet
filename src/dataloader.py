"""
Import libraries
"""
import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class CASLDataset(data.Dataset):
    """
    Dataloader for handling two data sources (UAV and Satellite) for segmentation tasks.
    """

    def __init__(self, base_root, regions, input_size=256):
        self.base_root = base_root
        self.regions = regions
        self.input_size = input_size

        self.uav_images = []
        self.uav_masks = []
        self.sat_images = []
        self.sat_masks = []
        # Load paths for all specified regions
        for region in regions:
            uav_path = os.path.join(self.base_root, f"{region}(UAV)")
            sat_path = os.path.join(self.base_root, f"{region}(SAT)")

            uav_imgs = [os.path.join(uav_path, "img", f) for f in os.listdir(os.path.join(uav_path, "img")) if
                        f.endswith('.tif')]
            uav_masks = [os.path.join(uav_path, "label", f) for f in os.listdir(os.path.join(uav_path, "label")) if
                         f.endswith('.tif')]
            sat_imgs = [os.path.join(sat_path, "img", f) for f in os.listdir(os.path.join(sat_path, "img")) if
                        f.endswith('.tif')]
            sat_masks = [os.path.join(sat_path, "label", f) for f in os.listdir(os.path.join(sat_path, "label")) if
                         f.endswith('.tif')]
            uav_imgs.sort()
            uav_masks.sort()
            sat_imgs.sort()
            sat_masks.sort()
            self.uav_images.extend(uav_imgs)
            self.uav_masks.extend(uav_masks)
            self.sat_images.extend(sat_imgs)
            self.sat_masks.extend(sat_masks)

        # Ensure lengths match between images and their masks
        # assert len(self.uav_images) == len(self.uav_masks)
        # assert len(self.sat_images) == len(self.sat_masks)

        # transformations
        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))
        ])

    def __len__(self):
        return len(self.sat_images)

    def __getitem__(self, index):
        # Load UAV data
        uav_image = self.img_loader(self.uav_images[index])
        uav_mask = self.mask_loader(self.uav_masks[index])

        # Load Satellite data
        sat_image = self.img_loader(self.sat_images[index])
        sat_mask = self.mask_loader(self.sat_masks[index])

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        uav_image = self.img_transform(uav_image)
        uav_mask = self.mask_transform(uav_mask)

        sat_image = self.img_transform(sat_image)
        sat_mask = self.mask_transform(sat_mask)

        return {
            "uav_image": uav_image,
            "uav_mask": uav_mask,
            "sat_image": sat_image,
            "sat_mask": sat_mask
        }

    def img_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def mask_loader(self, path):
        with open(path, 'rb') as f:
            mask = Image.open(f)
            mask = mask.convert('L')
            mask = mask.resize((self.input_size, self.input_size))
            mask = np.array(mask) / 255.0
            return torch.tensor(mask, dtype=torch.float32)
