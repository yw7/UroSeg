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


# ── resample ──────────────────────────────────────────────────────────────────

def test_resample_changes_spacing(img_file, tmp_path):
    from uroseg.commands.resample import process_one
    import argparse
    out = tmp_path / 'resampled.nii.gz'
    args = argparse.Namespace(spacing=[2.0, 2.0, 2.0], overwrite=True)
    process_one((img_file, out), args)
    assert out.exists()
    img = Image.load(out)
    assert img.data.ndim == 3


def test_resample_cli(img_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'resample',
         '--img', str(img_file), '--out', str(out),
         '--spacing', '2', '2', '2', '--out-suffix', '_resampled'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_resampled.nii.gz').exists()


# ── reorient ──────────────────────────────────────────────────────────────────

def test_reorient_produces_output(img_file, tmp_path):
    from uroseg.commands.reorient_canonical import process_one
    import argparse
    out = tmp_path / 'reoriented.nii.gz'
    args = argparse.Namespace(orientation='RAS', overwrite=True)
    process_one((img_file, out), args)
    assert out.exists()
    img = Image.load(out)
    assert img.data.ndim == 3


def test_reorient_cli(img_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'reorient',
         '--img', str(img_file), '--out', str(out),
         '--orientation', 'RAS', '--out-suffix', '_ras'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_ras.nii.gz').exists()


# ── largest_component ─────────────────────────────────────────────────────────

def test_keep_largest_component_removes_small(tmp_path):
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    data[1:3, 1:3, 1:3] = 1    # small blob
    data[10:20, 10:20, 10:20] = 1  # large blob
    result = keep_largest_component(data, labels=None)
    assert result[1, 1, 1] == 0
    assert result[15, 15, 15] == 1


def test_keep_largest_component_per_label(tmp_path):
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    data[1:3, 1:3, 1:3] = 1
    data[10:20, 10:20, 10:20] = 1
    data[1:3, 20:22, 1:3] = 2
    data[10:20, 1:10, 10:20] = 2
    result = keep_largest_component(data, labels=[1, 2])
    assert result[1, 1, 1] == 0
    assert result[15, 15, 15] == 1
    assert result[1, 21, 1] == 0
    assert result[15, 5, 15] == 2


def test_largest_component_cli(seg_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'largest_component',
         '--seg', str(seg_file), '--out', str(out), '--out-suffix', '_lc'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'seg_lc.nii.gz').exists()


# ── crop_image2seg ────────────────────────────────────────────────────────────

def test_crop_reduces_size(img_file, seg_file, tmp_path):
    from uroseg.commands.crop_image2seg import crop_to_seg
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped_img, cropped_seg = crop_to_seg(img, seg)
    assert cropped_img.data.shape[0] <= img.data.shape[0]
    assert cropped_seg.data.shape == cropped_img.data.shape


def test_crop_preserves_seg_labels(img_file, seg_file, tmp_path):
    from uroseg.commands.crop_image2seg import crop_to_seg
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    _, cropped_seg = crop_to_seg(img, seg)
    assert 1 in np.unique(cropped_seg.data)


def test_crop_cli(img_file, seg_file, tmp_path):
    import subprocess, sys
    out_img = tmp_path / 'out_img'
    out_seg = tmp_path / 'out_seg'
    out_img.mkdir(); out_seg.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'crop',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out-img', str(out_img), '--out-seg', str(out_seg),
         '--img-suffix', '_crop', '--seg-suffix', '_crop'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out_img / 'img_crop.nii.gz').exists()
    assert (out_seg / 'seg_crop.nii.gz').exists()


# ── preview_jpg ───────────────────────────────────────────────────────────────

def test_make_preview_no_seg(img_file):
    from uroseg.commands.preview_jpg import make_preview
    img = Image.load(img_file)
    preview = make_preview(img.data, seg_data=None)
    assert preview.ndim == 3
    assert preview.shape[2] == 3
    assert preview.dtype == np.uint8


def test_make_preview_with_seg(img_file, seg_file):
    from uroseg.commands.preview_jpg import make_preview
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    preview = make_preview(img.data, seg_data=seg.data)
    assert preview.ndim == 3
    assert preview.dtype == np.uint8


def test_preview_cli_no_seg(img_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'preview',
         '--img', str(img_file), '--out', str(out), '--out-suffix', '_preview'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_preview.jpg').exists()


def test_preview_cli_with_seg(img_file, seg_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'preview',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out', str(out), '--out-suffix', '_preview'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_preview.jpg').exists()


# ── transform_seg2image ───────────────────────────────────────────────────────

def test_transform_seg_matches_img_shape(img_file, seg_file, tmp_path):
    from uroseg.commands.transform_seg2image import resample_seg_to_image
    # create a differently-shaped seg
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2:8, 2:8, 2:8] = 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    seg_small = Image(data, affine, None)
    ref = Image.load(img_file)
    result = resample_seg_to_image(seg_small, ref)
    assert result.data.shape == ref.data.shape


def test_transform_seg2image_cli(img_file, seg_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'transform_seg2image',
         '--seg', str(seg_file), '--img', str(img_file),
         '--out-seg', str(out), '--seg-suffix', '_transformed'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'seg_transformed.nii.gz').exists()
