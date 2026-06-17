import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .augmenter import Augmenter


class MegaAgeAsianDataset(Dataset):
    """
    MegaAge Asian dataset for age prediction (transfer learning on AdaFace backbone).

    Directory layout expected:
        root/
            train/          <- pre-cropped face images (already aligned)
            test/           <- pre-cropped face images (already aligned)
            list/
                train_name.txt   <- one filename per line (e.g. "1.jpg")
                train_age.txt    <- one integer age per line
                test_name.txt
                test_age.txt

    Images are pre-cropped face patches (178x218).  No face detection is needed —
    the transform is responsible for resizing to the model's target input size.

    Returns: (image_tensor, age_label) where age_label is an integer.
    """

    def __init__(
        self,
        root,
        split='train',
        transform=None,
        low_res_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
        swap_color_channel=False,
        output_dir='./',
    ):
        assert split in ('train', 'test'), f"split must be 'train' or 'test', got '{split}'"
        self.img_dir = os.path.join(root, split)
        self.transform = transform
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)

        name_file = os.path.join(root, 'list', f'{split}_name.txt')
        age_file = os.path.join(root, 'list', f'{split}_age.txt')

        with open(name_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        with open(age_file, 'r') as f:
            ages = [int(line.strip()) for line in f if line.strip()]

        assert len(names) == len(ages), (
            f"Mismatch: {len(names)} names vs {len(ages)} ages in {split} split"
        )

        self.samples = [
            (os.path.join(self.img_dir, name), age)
            for name, age in zip(names, ages)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        # Load as BGR, convert to PIL RGB
        img_bgr = cv2.imread(img_path)
        sample = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # AdaFace training convention: store BGR channel order in PIL
        if not self.swap_color_channel:
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        sample = self.augmenter.augment(sample)

        sample_save_path = os.path.join(self.output_dir, 'training_samples', 'megaage_sample.jpg')
        if not os.path.isfile(sample_save_path):
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            cv2.imwrite(sample_save_path, np.array(sample))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
