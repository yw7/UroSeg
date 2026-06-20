import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
from uroseg.utils.image import Image


def test_load_returns_image(nifti_file):
    img = Image.load(nifti_file)
    assert isinstance(img.data, np.ndarray)
    assert img.data.shape == (20, 20, 20)
    assert img.affine.shape == (4, 4)


def test_load_nii_gz(nifti_file):
    img = Image.load(nifti_file)
    assert img.data[5, 5, 5] == 1


def test_save_roundtrip(nifti_file, tmp_path):
    img = Image.load(nifti_file)
    out = tmp_path / "out.nii.gz"
    img.save(out)
    assert out.exists()
    img2 = Image.load(out)
    np.testing.assert_array_equal(img.data, img2.data)


def test_save_creates_parent_dirs(nifti_file, tmp_path):
    img = Image.load(nifti_file)
    out = tmp_path / "nested" / "dir" / "out.nii.gz"
    img.save(out)
    assert out.exists()


def test_copy_is_independent(nifti_file):
    img = Image.load(nifti_file)
    img2 = img.copy()
    img2.data[0, 0, 0] = 99
    assert img.data[0, 0, 0] != 99


def test_reorient_ras(nifti_file):
    img = Image.load(nifti_file)
    reoriented = img.reorient('RAS')
    assert isinstance(reoriented, Image)
    assert reoriented.data.ndim == 3


def test_resample(nifti_file):
    img = Image.load(nifti_file)
    resampled = img.resample((2.0, 2.0, 2.0))
    assert isinstance(resampled, Image)
    assert resampled.data.ndim == 3


def test_bounding_box_label(nifti_file):
    img = Image.load(nifti_file)
    bb = img.bounding_box(label=1)
    assert bb is not None
    assert len(bb) == 3


def test_bounding_box_empty_returns_none(nifti_file):
    img = Image.load(nifti_file)
    bb = img.bounding_box(label=99)
    assert bb is None
