from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.orientations as nibo
import nibabel.processing as nibp


class Image:
    def __init__(self, data: np.ndarray, affine: np.ndarray, header):
        self.data = data
        self.affine = affine
        self.header = header

    @staticmethod
    def load(path: str | Path) -> Image:
        path = Path(path)
        img = nib.load(str(path))
        return Image(
            data=np.asanyarray(img.dataobj),
            affine=img.affine.copy(),
            header=img.header,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.data, self.affine, self.header), str(path))

    def copy(self) -> Image:
        return Image(self.data.copy(), self.affine.copy(), self.header.copy())

    def reorient(self, orientation: str = 'RAS') -> Image:
        ornt_orig = nibo.io_orientation(self.affine)
        ornt_targ = nibo.axcodes2ornt(tuple(orientation))
        transform = nibo.ornt_transform(ornt_orig, ornt_targ)
        data = nibo.apply_orientation(self.data, transform)
        affine = self.affine @ nibo.inv_ornt_aff(transform, self.data.shape)
        return Image(data, affine, self.header)

    def resample(self, voxel_size: tuple) -> Image:
        nib_img = nib.Nifti1Image(self.data, self.affine, self.header)
        resampled = nibp.resample_to_output(nib_img, voxel_size)
        return Image(
            np.asanyarray(resampled.dataobj),
            resampled.affine,
            resampled.header,
        )

    def bounding_box(self, label: int = None) -> tuple | None:
        mask = self.data if label is None else (self.data == label)
        nonzero = np.argwhere(mask)
        if len(nonzero) == 0:
            return None
        mins = nonzero.min(axis=0)
        maxs = nonzero.max(axis=0)
        return tuple(slice(int(mn), int(mx) + 1) for mn, mx in zip(mins, maxs))
