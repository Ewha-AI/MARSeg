import os
import re

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset



class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, fixed_class=0, num_slices=3):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'labels')
        self.img_filenames = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        self.img_filenames.sort()
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]
        self.fixed_class = fixed_class
        self.num_slices = num_slices
        self.patient_slices = {}
        img_filenames = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        img_filenames.sort()
        
        pattern = r'pancreas_(\d+)_(\d+)\.png'   # MSD pancreas dataset
        for filename in img_filenames:
            match = re.match(pattern, filename)
            if match:
                patient_id = int(match.group(1))
                slice_num = int(match.group(2))
                if patient_id not in self.patient_slices:
                    self.patient_slices[patient_id] = []
                self.patient_slices[patient_id].append((slice_num, filename))

        for patient_id in self.patient_slices:
            self.patient_slices[patient_id].sort(key=lambda x: x[0])

        self.slice_indices = []
        for patient_id in self.patient_slices:
            slices = self.patient_slices[patient_id]
            for idx in range(len(slices)):
                self.slice_indices.append((patient_id, idx))

        total_channels = self.num_slices * 3
        self.image_mean = [0.5] * total_channels
        self.image_std = [0.5] * total_channels

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        patient_id, slice_idx = self.slice_indices[idx]
        slices = self.patient_slices[patient_id]
        total_slices = len(slices)

        half_num_slices = self.num_slices // 2
        slice_indices = [slice_idx + i for i in range(-half_num_slices, half_num_slices + 1)]
        slice_indices = [min(max(0, s), total_slices - 1) for s in slice_indices]

        images = []
        for s_idx in slice_indices:
            _, img_name = slices[s_idx]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            images.append(image)

        _, mask_name = slices[slice_idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        mask = mask.convert("L")

        if self.split == 'train':
            crop_params = transforms.RandomCrop.get_params(mask, output_size=(256, 256))
            i, j, h, w = crop_params

            hflip = (torch.rand(1) < 0.5).item()
            vflip = (torch.rand(1) < 0.5).item()
            angle = transforms.RandomRotation.get_params(degrees=(-15, 15))

            transformed_images = []
            for image in images:
                image = transforms.functional.crop(image, i, j, h, w)
                if hflip:
                    image = transforms.functional.hflip(image)
                if vflip:
                    image = transforms.functional.vflip(image)
                image = transforms.functional.rotate(image, angle)
                transformed_images.append(image)

            mask = transforms.functional.crop(mask, i, j, h, w)
            if hflip:
                mask = transforms.functional.hflip(mask)
            if vflip:
                mask = transforms.functional.vflip(mask)
            mask = transforms.functional.rotate(mask, angle)
        else:
            transformed_images = [transforms.functional.center_crop(img, (256, 256)) for img in images]
            mask = transforms.functional.center_crop(mask, (256, 256))
            
        transformed_images = [transforms.functional.to_tensor(img) for img in transformed_images]
        image = torch.cat(transformed_images, dim=0)
        image = transforms.functional.normalize(image, mean=self.image_mean, std=self.image_std)

        mask = np.array(mask).astype(np.int64)
        mask = torch.from_numpy(mask)
        target = mask.long()
        original_mask = mask.long()
        class_label = torch.tensor(self.fixed_class, dtype=torch.long)

        return image, target, class_label, original_mask
