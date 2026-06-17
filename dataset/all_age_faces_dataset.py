import os
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .augmenter import Augmenter


class AllAgeFacesDataset(Dataset):
    """
    All-Age-Faces dataset for gender classification (transfer learning on AdaFace backbone).

    Directory layout expected:
        root/
            original images/  <- uncropped originals (%05dA%02d.jpg)
            image sets/
                train.txt     <- "filename gender\\n", gender 0=female 1=male
                val.txt

    Prefer `AllAgeFacesDataset.split()` over direct construction when you need
    a custom train/val ratio — it shuffles once and hands each dataset its slice,
    guaranteeing no overlap.

    Returns: (image_tensor, gender_label) where gender_label in {0, 1}.
    """

    def __init__(
        self,
        samples,
        img_dir,
        transform=None,
        face_align_fn=None,
        low_res_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
        swap_color_channel=False,
        output_dir='./',
    ):
        self.samples = samples          # list of (img_path, gender_label)
        self.img_dir = img_dir
        self.transform = transform
        self.face_align_fn = face_align_fn
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)

    # ── factory helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_all(root):
        img_dir = os.path.join(root, 'original images')
        samples = []
        for split_name in ('train', 'val'):
            ann_file = os.path.join(root, 'image sets', f'{split_name}.txt')
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        samples.append((os.path.join(img_dir, parts[0]), int(parts[1])))
        return samples, img_dir

    @classmethod
    def from_official_split(cls, root, split='train', **kwargs):
        """Use the dataset's original train/val split files (50/50)."""
        assert split in ('train', 'val')
        img_dir = os.path.join(root, 'original images')
        samples = []
        ann_file = os.path.join(root, 'image sets', f'{split}.txt')
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    samples.append((os.path.join(img_dir, parts[0]), int(parts[1])))
        return cls(samples, img_dir, **kwargs)

    @classmethod
    def split(cls, root, train_ratio=0.8, seed=42, train_kwargs=None, val_kwargs=None):
        """
        Pool all 13,322 samples, shuffle once, split by train_ratio.
        Returns (train_dataset, val_dataset) with guaranteed no overlap.

        train_kwargs / val_kwargs: dicts of constructor kwargs (transform, face_align_fn, etc.)
        """
        all_samples, img_dir = cls._load_all(root)
        random.Random(seed).shuffle(all_samples)
        cut = int(len(all_samples) * train_ratio)
        train_samples, val_samples = all_samples[:cut], all_samples[cut:]

        train_ds = cls(train_samples, img_dir, **(train_kwargs or {}))
        val_ds   = cls(val_samples,   img_dir, **(val_kwargs   or {}))
        return train_ds, val_ds

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img_bgr = cv2.imread(img_path)
        sample = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if self.face_align_fn is not None:
            aligned = self.face_align_fn(sample)
            if aligned is None:
                return self.__getitem__((index + 1) % len(self.samples))
            sample = aligned

        # AdaFace training convention: store BGR channel order in PIL
        if not self.swap_color_channel:
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        sample = self.augmenter.augment(sample)

        sample_save_path = os.path.join(self.output_dir, 'training_samples', 'aaf_sample.jpg')
        if not os.path.isfile(sample_save_path):
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            cv2.imwrite(sample_save_path, np.array(sample))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
