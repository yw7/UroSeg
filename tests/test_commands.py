import json
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
from uroseg.utils.image import Image


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def seg_file(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[2:8, 2:8, 2:8] = 1
    data[10:16, 10:16, 10:16] = 2
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "seg.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def img_file(tmp_path):
    data = np.random.randint(0, 1000, (20, 20, 20), dtype=np.int16)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "img.nii.gz")
    return tmp_path / "img.nii.gz"


@pytest.fixture
def map_json(tmp_path):
    mapping = {"1": 10, "2": 20}
    path = tmp_path / "map.json"
    path.write_text(json.dumps(mapping))
    return path


# ── map_labels ────────────────────────────────────────────────────────────────

def test_map_labels_remaps_values(seg_file, map_json, tmp_path):
    from uroseg.commands.map_labels import apply_map
    img = Image.load(seg_file)
    with open(map_json) as f:
        mapping = json.load(f)
    result = apply_map(img.data, mapping)
    assert result[3, 3, 3] == 10
    assert result[12, 12, 12] == 20
    assert result[0, 0, 0] == 0


def test_map_labels_unmapped_becomes_zero(seg_file, tmp_path):
    from uroseg.commands.map_labels import apply_map
    img = Image.load(seg_file)
    mapping = {"1": 5}
    result = apply_map(img.data, mapping)
    assert result[3, 3, 3] == 5
    assert result[12, 12, 12] == 0


def test_map_labels_cli(seg_file, map_json, tmp_path):
    import subprocess, sys
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'map',
         '--seg', str(seg_file), '--out', str(out),
         '--map', str(map_json), '--out-suffix', '_mapped'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    out_file = out / 'seg_mapped.nii.gz'
    assert out_file.exists()
    img = Image.load(out_file)
    assert img.data[3, 3, 3] == 10
