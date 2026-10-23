"""Image-classification datasets used for frozen-feature evaluation.

`EuroSAT` follows the loader of the CACo reference implementation
(third_party/caco_upstream/src/datasets/eurosat_dataset.py,
MIT License, (c) 2023 Utkarsh Mall).
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class _ImageList(Dataset):
    """A list of (path, label) pairs read with PIL and converted to RGB."""

    def __init__(self, samples, classes, transform=None):
        self.samples = samples
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class EuroSAT(_ImageList):
    """EuroSAT RGB with the SeCo/CACo `train.txt` / `val.txt` split files.

    Each line of a split file is an image file name such as `Forest_123.jpg`;
    the class is the prefix before the first underscore.
    """

    def __init__(self, root, split, transform=None):
        root = Path(root)
        classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        names = (root / f'{split}.txt').read_text().split()
        samples = []
        for name in names:
            cls = name.split('_')[0]
            samples.append((root / cls / name, class_to_idx[cls]))
        super().__init__(samples, classes, transform)


class UCMerced(_ImageList):
    """UC Merced Land Use (21 classes x 100 images).

    `split_file` lists one `class/file.tif` path per line, relative to `root`.
    """

    def __init__(self, root, split_file, transform=None):
        root = Path(root)
        classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        rel_paths = Path(split_file).read_text().split()
        samples = [(root / p, class_to_idx[p.split('/')[0]]) for p in rel_paths]
        super().__init__(samples, classes, transform)
