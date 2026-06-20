import numpy as np
import nibabel as nib
import pytest
from pathlib import Path


@pytest.fixture
def nifti_file(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[5:15, 5:15, 5:15] = 1
    data[7:13, 7:13, 7:13] = 2
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "test.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def nifti_folder(tmp_path, nifti_file):
    folder = tmp_path / "inputs"
    folder.mkdir()
    import shutil
    for i in range(3):
        shutil.copy(nifti_file, folder / f"case{i:03d}.nii.gz")
    return folder
