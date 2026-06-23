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

    def crop_to_seg(self, seg: 'Image', margin: int = 0) -> 'Image':
        seg_data = np.round(seg.data).astype(np.uint8)
        coords = np.argwhere(seg_data != 0)
        if len(coords) == 0:
            return self.copy()
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        shape = np.array(seg_data.shape)
        lo = np.maximum(mins - margin, 0)
        hi = np.minimum(maxs + margin, shape - 1)
        slices = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
        nib_img = nib.Nifti1Image(self.data, self.affine, self.header)
        cropped = nib_img.slicer[slices]
        return Image(np.asanyarray(cropped.dataobj), cropped.affine, cropped.header)

    def as_canonical(self) -> 'Image':
        nib_img = nib.Nifti1Image(self.data, self.affine, self.header)
        canonical = nib.as_closest_canonical(nib_img)
        return Image(np.asanyarray(canonical.dataobj), canonical.affine, canonical.header)


def save_nifti_image(data: np.ndarray, affine: np.ndarray, header, path) -> None:
    """Save a NIfTI image preserving integer dtype with overflow rescaling.

    If the array dtype is integer and the data values overflow the dtype range
    (can happen after float-space processing), linearly rescale to fit.
    Always sets qform and sform to affine.
    """
    dtype = data.dtype
    if np.issubdtype(dtype, np.integer):
        data = data.astype(np.float64)
        d_min, d_max = data.min(), data.max()
        info = np.iinfo(dtype)
        if d_min < info.min or d_max > info.max:
            rescaled = data * (info.max - info.min) / (d_max - d_min)
            data = rescaled - (rescaled.min() - info.min)
        data = data.astype(dtype)
    header = header.copy() if header is not None else nib.Nifti1Header()
    header.set_data_dtype(dtype)
    nib_img = nib.Nifti1Image(data, affine, header)
    nib_img.set_qform(affine)
    nib_img.set_sform(affine)
    nib.save(nib_img, path)


def save_nifti_seg(data: np.ndarray, affine: np.ndarray, header, path) -> None:
    """Save a segmentation as uint8, rounding first. Always sets qform/sform."""
    data = np.round(data).astype(np.uint8)
    header = header.copy() if header is not None else nib.Nifti1Header()
    header.set_data_dtype(np.uint8)
    nib_img = nib.Nifti1Image(data, affine, header)
    nib_img.set_qform(affine)
    nib_img.set_sform(affine)
    nib.save(nib_img, path)
