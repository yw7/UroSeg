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


def test_map_labels_keep_unmapped(seg_file, tmp_path):
    from uroseg.commands.map_labels import apply_map
    img = Image.load(seg_file)
    # Only map label 1 -> 5; label 2 should be kept unchanged
    mapping = {1: 5}
    result = apply_map(img.data, mapping, keep_unmapped=True)
    assert result[3, 3, 3] == 5    # label 1 -> 5
    assert result[12, 12, 12] == 2  # label 2 kept


def test_map_labels_direct_pairs_parsing(seg_file, tmp_path):
    """Test that --map 1:2 3:0 style CLI args produce the correct mapping."""
    import subprocess, sys
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'map',
         '--seg', str(seg_file), '--out', str(out),
         '--map', '1:10', '2:20', '--out-suffix', '_mapped'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    out_file = out / 'seg_mapped.nii.gz'
    assert out_file.exists()
    img = Image.load(out_file)
    assert img.data[3, 3, 3] == 10
    assert img.data[12, 12, 12] == 20


def test_map_labels_keep_unmapped_cli(seg_file, tmp_path):
    """--keep-unmapped preserves labels not in the map."""
    import subprocess, sys
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'map',
         '--seg', str(seg_file), '--out', str(out),
         '--map', '1:10', '--keep-unmapped', '--out-suffix', '_mapped'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    out_file = out / 'seg_mapped.nii.gz'
    img = Image.load(out_file)
    assert img.data[3, 3, 3] == 10   # label 1 -> 10
    assert img.data[12, 12, 12] == 2  # label 2 kept


def test_map_labels_update_seg_cli(seg_file, tmp_path):
    """--update-seg fills zeros in output from companion seg."""
    import subprocess, sys
    import nibabel as nib

    # Build a companion seg with label 99 where output will be zero (label 2 region)
    companion_data = np.zeros((20, 20, 20), dtype=np.int16)
    companion_data[10:16, 10:16, 10:16] = 99
    companion_path = tmp_path / "companion.nii.gz"
    nib.save(nib.Nifti1Image(companion_data, np.eye(4)), companion_path)

    out = tmp_path / "out"
    out.mkdir()
    # Map only label 1 -> 10; label 2 becomes 0 (unmapped, no keep-unmapped)
    # --update-seg fills those zeros with companion's 99
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'map',
         '--seg', str(seg_file), '--out', str(out),
         '--map', '1:10', '--update-seg', str(companion_path),
         '--out-suffix', '_mapped'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    img = Image.load(out / 'seg_mapped.nii.gz')
    assert img.data[3, 3, 3] == 10    # label 1 -> 10
    assert img.data[12, 12, 12] == 99  # zero filled from companion


def test_map_labels_update_from_seg_cli(seg_file, tmp_path):
    """--update-from-seg overwrites output where companion seg is non-zero."""
    import subprocess, sys
    import nibabel as nib

    # Build a companion seg that overwrites the label-1 region with 77
    companion_data = np.zeros((20, 20, 20), dtype=np.int16)
    companion_data[2:8, 2:8, 2:8] = 77
    companion_path = tmp_path / "from_seg.nii.gz"
    nib.save(nib.Nifti1Image(companion_data, np.eye(4)), companion_path)

    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'map',
         '--seg', str(seg_file), '--out', str(out),
         '--map', '1:10', '2:20',
         '--update-from-seg', str(companion_path),
         '--out-suffix', '_mapped'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    img = Image.load(out / 'seg_mapped.nii.gz')
    assert img.data[3, 3, 3] == 77    # overwritten by companion
    assert img.data[12, 12, 12] == 20  # label 2 -> 20, companion is zero here


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
    args = argparse.Namespace(mm=[2.0, 2.0, 2.0], overwrite=True)
    process_one((img_file, out), args)
    assert out.exists()
    img = Image.load(out)
    assert img.data.ndim == 3


def test_resample_mm_single_value_isotropic(img_file, tmp_path):
    """--mm 1 (single value) resamples isotropically."""
    from uroseg.commands.resample import process_one
    import argparse
    out = tmp_path / 'resampled_iso.nii.gz'
    args = argparse.Namespace(mm=[1.0], overwrite=True)
    process_one((img_file, out), args)
    assert out.exists()
    result = Image.load(out)
    assert result.data.ndim == 3


def test_resample_mm_three_values_anisotropic(img_file, tmp_path):
    """--mm 1 0.5 0.5 (three values) resamples anisotropically."""
    from uroseg.commands.resample import process_one
    import argparse
    out = tmp_path / 'resampled_aniso.nii.gz'
    args = argparse.Namespace(mm=[1.0, 0.5, 0.5], overwrite=True)
    process_one((img_file, out), args)
    assert out.exists()
    result = Image.load(out)
    assert result.data.ndim == 3


def test_resample_cli_mm(img_file, tmp_path):
    """CLI --mm flag (single value) works end-to-end."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'resample',
         '--img', str(img_file), '--out', str(out),
         '--mm', '2', '--out-suffix', '_resampled'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_resampled.nii.gz').exists()


def test_resample_cli_mm_three_values(img_file, tmp_path):
    """CLI --mm with three values works end-to-end."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'resample',
         '--img', str(img_file), '--out', str(out),
         '--mm', '1', '0.5', '0.5', '--out-suffix', '_resampled'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_resampled.nii.gz').exists()


def test_resample_cli(img_file, tmp_path):
    """--spacing backwards-compat alias still works."""
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


def test_keep_largest_component_binarize(tmp_path):
    """--binarize: keep only the single largest connected region across all labels."""
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    # Large multi-label blob (connected)
    data[10:20, 10:20, 10:20] = 1
    data[10:20, 10:20, 20:25] = 2   # adjacent to label-1 block, forms one big region
    # Small isolated blob
    data[1:3, 1:3, 1:3] = 1
    result = keep_largest_component(data, binarize=True)
    # Large region should be kept with original label values
    assert result[15, 15, 15] == 1
    assert result[15, 15, 22] == 2
    # Small isolated blob should be removed
    assert result[1, 1, 1] == 0


def test_keep_largest_component_dilate(tmp_path):
    """--dilate connects nearby components then masks back to original voxels."""
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    # Two small blobs of label 1 separated by a 2-voxel gap
    data[10:13, 10:13, 10:13] = 1   # left blob  (~27 voxels)
    data[10:13, 10:13, 15:18] = 1   # right blob (~27 voxels), gap of 2 voxels
    # Tiny isolated blob far away
    data[1:2, 1:2, 1:2] = 1         # 1 voxel
    # Without dilation the two main blobs are separate CCs; the larger one wins.
    # With dilate=3 the two main blobs merge and together dominate the tiny one.
    result = keep_largest_component(data, dilate=3)
    # Both main blobs should survive (they merge under dilation)
    assert result[11, 11, 11] == 1
    assert result[11, 11, 16] == 1
    # Tiny isolated blob should be removed
    assert result[1, 1, 1] == 0
    # Background voxels in the gap must remain zero (dilation is undone)
    assert result[10, 10, 13] == 0


def test_keep_largest_component_26connectivity():
    """Diagonal neighbours (face-corner touching) should be connected with 26-connectivity."""
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((10, 10, 10), dtype=np.int16)
    # Two voxels touching only at a corner (diagonal in all three axes)
    data[3, 3, 3] = 1
    data[4, 4, 4] = 1   # 26-connected to (3,3,3) but NOT 6-connected
    result = keep_largest_component(data)
    # Both voxels belong to the same (and only) CC, so both are kept
    assert result[3, 3, 3] == 1
    assert result[4, 4, 4] == 1


def test_largest_component_binarize_cli(seg_file, tmp_path):
    """CLI --binarize flag produces valid output."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'largest_component',
         '--seg', str(seg_file), '--out', str(out),
         '--out-suffix', '_lc', '--binarize'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'seg_lc.nii.gz').exists()


def test_largest_component_dilate_cli(seg_file, tmp_path):
    """CLI --dilate flag produces valid output."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'largest_component',
         '--seg', str(seg_file), '--out', str(out),
         '--out-suffix', '_lc', '--dilate', '2'],
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


def test_crop_margin_expands_bbox(img_file, seg_file, tmp_path):
    """--margin N expands the bounding box by N voxels on each side."""
    from uroseg.commands.crop_image2seg import crop_to_seg
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped_no_margin, _ = crop_to_seg(img, seg, margin=0)
    cropped_with_margin, _ = crop_to_seg(img, seg, margin=2)
    # Margin should make the cropped volume at least as large along every axis
    for i in range(3):
        assert cropped_with_margin.data.shape[i] >= cropped_no_margin.data.shape[i]
    # At least one dimension must be strictly larger (unless clamped by image bounds)
    orig_shape = img.data.shape
    expanded = any(
        cropped_with_margin.data.shape[i] > cropped_no_margin.data.shape[i]
        or cropped_no_margin.data.shape[i] == orig_shape[i]  # already at boundary
        for i in range(3)
    )
    assert expanded


def test_crop_margin_clamped_to_image(img_file, seg_file, tmp_path):
    """Very large margin is clamped to the image boundary, not out-of-bounds."""
    from uroseg.commands.crop_image2seg import crop_to_seg
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped_img, cropped_seg = crop_to_seg(img, seg, margin=100)
    # Result should not exceed original image shape
    for i in range(3):
        assert cropped_img.data.shape[i] <= img.data.shape[i]


def test_crop_margin_cli(img_file, seg_file, tmp_path):
    """CLI --margin flag is accepted and produces valid output."""
    import subprocess, sys
    out_img = tmp_path / 'out_img'
    out_seg = tmp_path / 'out_seg'
    out_img.mkdir(); out_seg.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'crop',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out-img', str(out_img), '--out-seg', str(out_seg),
         '--img-suffix', '_crop', '--seg-suffix', '_crop',
         '--margin', '2'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out_img / 'img_crop.nii.gz').exists()
    assert (out_seg / 'seg_crop.nii.gz').exists()


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


# ── cpdir ─────────────────────────────────────────────────────────────────────

def test_cpdir_copies_files(nifti_folder, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'cpdir',
         '--img', str(nifti_folder), '--out', str(out), '--out-suffix', ''],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob('*.nii.gz'))) == 3


def test_cpdir_skips_existing_without_overwrite(nifti_folder, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    # first copy
    subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'cpdir',
         '--img', str(nifti_folder), '--out', str(out), '--out-suffix', ''],
        capture_output=True
    )
    # second copy without --overwrite should skip
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'cpdir',
         '--img', str(nifti_folder), '--out', str(out), '--out-suffix', ''],
        capture_output=True, text=True
    )
    assert result.returncode == 0


# ── transform_seg2image ───────────────────────────────────────────────────────

def test_transform_seg_matches_img_shape(img_file, seg_file, tmp_path):
    from uroseg.commands.transform_seg2image import resample_seg_to_image
    # create a differently-shaped seg
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2:8, 2:8, 2:8] = 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    seg_small = Image(data, affine, None)
    ref = Image.load(img_file)
    result = resample_seg_to_image(seg_small, ref, interpolation='nearest')
    assert result.data.shape == ref.data.shape


def test_transform_seg2image_nearest_regression(img_file, seg_file, tmp_path):
    """Regression: --interpolation nearest produces integer segmentation output."""
    from uroseg.commands.transform_seg2image import resample_seg_to_image
    seg = Image.load(seg_file)
    ref = Image.load(img_file)
    result = resample_seg_to_image(seg, ref, interpolation='nearest')
    assert result.data.shape == ref.data.shape
    assert np.issubdtype(result.data.dtype, np.integer) or result.data.dtype == np.uint8


def test_transform_seg2image_linear_produces_float(img_file, seg_file, tmp_path):
    """--interpolation linear produces float32 output."""
    from uroseg.commands.transform_seg2image import resample_seg_to_image
    seg = Image.load(seg_file)
    ref = Image.load(img_file)
    result = resample_seg_to_image(seg, ref, interpolation='linear')
    assert result.data.shape == ref.data.shape
    assert result.data.dtype == np.float32


def test_transform_seg2image_label_single_voxel(tmp_path):
    """--interpolation label places single-voxel landmarks at center of mass."""
    from uroseg.commands.transform_seg2image import resample_seg_to_image
    import nibabel as nib
    # Seg: 20^3 at 1mm, two single-voxel labels
    seg_data = np.zeros((20, 20, 20), dtype=np.uint8)
    seg_data[5, 5, 5] = 1
    seg_data[15, 15, 15] = 2
    seg = Image(seg_data, np.eye(4), nib.Nifti1Header())
    # Ref: same space (should round-trip)
    ref_data = np.zeros((20, 20, 20), dtype=np.int16)
    ref = Image(ref_data, np.eye(4), nib.Nifti1Header())
    result = resample_seg_to_image(seg, ref, interpolation='label')
    assert result.data.shape == ref.data.shape
    # Each label should appear exactly once
    assert np.sum(result.data == 1) == 1
    assert np.sum(result.data == 2) == 1


def test_transform_seg2image_cli_nearest(img_file, seg_file, tmp_path):
    """CLI --interpolation nearest (default) produces output."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'transform_seg2image',
         '--seg', str(seg_file), '--img', str(img_file),
         '--out-seg', str(out), '--seg-suffix', '_transformed',
         '--interpolation', 'nearest'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'seg_transformed.nii.gz').exists()


def test_transform_seg2image_cli_linear(img_file, seg_file, tmp_path):
    """CLI --interpolation linear produces float output file."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'transform_seg2image',
         '--seg', str(seg_file), '--img', str(img_file),
         '--out-seg', str(out), '--seg-suffix', '_transformed',
         '--interpolation', 'linear'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    out_file = out / 'seg_transformed.nii.gz'
    assert out_file.exists()
    loaded = Image.load(out_file)
    assert np.issubdtype(loaded.data.dtype, np.floating)


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
