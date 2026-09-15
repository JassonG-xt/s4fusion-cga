"""Manifest-based paired infrared/visible dataset utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from train_gates import rgb_to_y


class DirectoryPairDataset(Dataset):
    """Load heterogeneous paired datasets through one CSV manifest."""

    REQUIRED_FIELDS = {"dataset", "sample_id", "ir_path", "vi_path", "split"}

    def __init__(self, data_root, manifest_path, crop_size=None, use_y=True, augment=False):
        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)
        self.crop_size = crop_size
        self.use_y = use_y
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = self.REQUIRED_FIELDS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"manifest missing fields: {sorted(missing)}")
            self.rows = list(reader)

    def __len__(self):
        return len(self.rows)

    def _crop(self, ir, vi):
        if self.crop_size is None:
            return ir, vi
        if isinstance(self.crop_size, int):
            ch = cw = self.crop_size
        else:
            ch, cw = self.crop_size
        h, w = ir.shape[-2:]
        if h < ch or w < cw:
            pad_h, pad_w = max(0, ch - h), max(0, cw - w)
            ir = torch.nn.functional.pad(ir, (0, pad_w, 0, pad_h), mode="reflect")
            vi = torch.nn.functional.pad(vi, (0, pad_w, 0, pad_h), mode="reflect")
            h, w = ir.shape[-2:]
        if self.augment:
            top = int(torch.randint(0, h - ch + 1, ()).item())
            left = int(torch.randint(0, w - cw + 1, ()).item())
        else:
            top, left = (h - ch) // 2, (w - cw) // 2
        return ir[:, top:top + ch, left:left + cw], vi[:, top:top + ch, left:left + cw]

    def __getitem__(self, index):
        row = self.rows[index]
        ir_path = self.data_root / row["ir_path"]
        vi_path = self.data_root / row["vi_path"]
        with Image.open(ir_path) as image:
            ir = self.to_tensor(image.convert("L"))
        with Image.open(vi_path) as image:
            vi_rgb = self.to_tensor(image.convert("RGB"))
        vi = rgb_to_y(vi_rgb) if self.use_y else vi_rgb
        ir, vi = self._crop(ir, vi)
        if self.augment and torch.rand(()) < 0.5:
            ir, vi = torch.flip(ir, (-1,)), torch.flip(vi, (-1,))
        if self.augment and torch.rand(()) < 0.5:
            ir, vi = torch.flip(ir, (-2,)), torch.flip(vi, (-2,))
        return vi, ir


def dataset_weights(manifest_path):
    """Return inverse-frequency sample weights for balanced joint training."""
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    return torch.as_tensor([1.0 / counts[row["dataset"]] for row in rows], dtype=torch.double)
