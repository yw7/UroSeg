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
    from uroseg.tools.map_labels import map_labels
    img = Image.load(seg_file)
    with open(map_json) as f:
        mapping = json.load(f)
    result = map_labels(img.data, mapping)
    assert result[3, 3, 3] == 10
    assert result[12, 12, 12] == 20
    assert result[0, 0, 0] == 0


def test_map_labels_unmapped_becomes_zero(seg_file, tmp_path):
    from uroseg.tools.map_labels import map_labels
    img = Image.load(seg_file)
    mapping = {"1": 5}
    result = map_labels(img.data, mapping)
    assert result[3, 3, 3] == 5
    assert result[12, 12, 12] == 0


def test_map_labels_keep_unmapped(seg_file, tmp_path):
    from uroseg.tools.map_labels import map_labels
    img = Image.load(seg_file)
    # Only map label 1 -> 5; label 2 should be kept unchanged
    mapping = {1: 5}
    result = map_labels(img.data, mapping, keep_unmapped=True)
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
    from uroseg.tools.resample import resample_file
    out = tmp_path / 'resampled.nii.gz'
    resample_file(img_file, out, mm=[2.0, 2.0, 2.0], overwrite=True)
    assert out.exists()
    img = Image.load(out)
    assert img.data.ndim == 3


def test_resample_mm_single_value_isotropic(img_file, tmp_path):
    """--mm 1 (single value) resamples isotropically."""
    from uroseg.tools.resample import resample_file
    out = tmp_path / 'resampled_iso.nii.gz'
    resample_file(img_file, out, mm=1.0, overwrite=True)
    assert out.exists()
    result = Image.load(out)
    assert result.data.ndim == 3


def test_resample_mm_three_values_anisotropic(img_file, tmp_path):
    """--mm 1 0.5 0.5 (three values) resamples anisotropically."""
    from uroseg.tools.resample import resample_file
    out = tmp_path / 'resampled_aniso.nii.gz'
    resample_file(img_file, out, mm=[1.0, 0.5, 0.5], overwrite=True)
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
    from uroseg.tools.reorient import reorient_file
    out = tmp_path / 'reoriented.nii.gz'
    reorient_file(img_file, out, overwrite=True)
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
         '--out-suffix', '_ras'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_ras.nii.gz').exists()


# ── largest_component ─────────────────────────────────────────────────────────

def test_keep_largest_component_removes_small(tmp_path):
    from uroseg.tools.largest_component import largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    data[1:3, 1:3, 1:3] = 1    # small blob
    data[10:20, 10:20, 10:20] = 1  # large blob
    result = largest_component(data, labels=None)
    assert result[1, 1, 1] == 0
    assert result[15, 15, 15] == 1


def test_keep_largest_component_per_label(tmp_path):
    from uroseg.tools.largest_component import largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    data[1:3, 1:3, 1:3] = 1
    data[10:20, 10:20, 10:20] = 1
    data[1:3, 20:22, 1:3] = 2
    data[10:20, 1:10, 10:20] = 2
    result = largest_component(data, labels=[1, 2])
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
    from uroseg.tools.largest_component import largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    # Large multi-label blob (connected)
    data[10:20, 10:20, 10:20] = 1
    data[10:20, 10:20, 20:25] = 2   # adjacent to label-1 block, forms one big region
    # Small isolated blob
    data[1:3, 1:3, 1:3] = 1
    result = largest_component(data, binarize=True)
    # Large region should be kept with original label values
    assert result[15, 15, 15] == 1
    assert result[15, 15, 22] == 2
    # Small isolated blob should be removed
    assert result[1, 1, 1] == 0


def test_keep_largest_component_dilate(tmp_path):
    """--dilate connects nearby components then masks back to original voxels."""
    from uroseg.tools.largest_component import largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    # Two small blobs of label 1 separated by a 2-voxel gap
    data[10:13, 10:13, 10:13] = 1   # left blob  (~27 voxels)
    data[10:13, 10:13, 15:18] = 1   # right blob (~27 voxels), gap of 2 voxels
    # Tiny isolated blob far away
    data[1:2, 1:2, 1:2] = 1         # 1 voxel
    # Without dilation the two main blobs are separate CCs; the larger one wins.
    # With dilate=3 the two main blobs merge and together dominate the tiny one.
    result = largest_component(data, dilate=3)
    # Both main blobs should survive (they merge under dilation)
    assert result[11, 11, 11] == 1
    assert result[11, 11, 16] == 1
    # Tiny isolated blob should be removed
    assert result[1, 1, 1] == 0
    # Background voxels in the gap must remain zero (dilation is undone)
    assert result[10, 10, 13] == 0


def test_dilate_uses_6conn_not_26conn():
    """Dilation uses 6-connectivity (not 26-connectivity), ensuring it cannot be accidentally changed.

    Two equal-size blobs separated by a 1-voxel diagonal gap:
    - Under 6-conn dilation (correct): gap not bridged → 2 CCs, one is removed.
    - Under 26-conn dilation (would change behavior): corner-adjacent dilation would bridge gap → 1 CC.

    This test documents that dilation employs explicit 6-connectivity structure (self-documenting;
    matches scipy default) and ensures the behavior remains stable.
    """
    from uroseg.tools.largest_component import largest_component
    data = np.zeros((25, 25, 25), dtype=np.int16)
    data[3:8, 3:8, 3:8] = 1    # blob A (5x5x5 = 125 voxels)
    data[9:14, 9:14, 9:14] = 1  # blob B (5x5x5 = 125 voxels), 1-voxel diagonal gap
    # Blob A corner: (7,7,7). Blob B corner: (9,9,9). Gap voxel: (8,8,8).
    # 6-conn: dilation from A reaches (8,7,7),(7,8,7),(7,7,8) but NOT (8,8,8).
    # 26-conn: dilation from A would reach (8,8,8) — bridging the diagonal gap.
    result = largest_component(data, dilate=1)
    # With 6-conn: two disconnected CCs of equal size → tie-break keeps blob A (first labeled)
    assert result[5, 5, 5] == 1    # blob A center kept
    assert result[11, 11, 11] == 0  # blob B removed (not merged)


def test_keep_largest_component_26connectivity():
    """Diagonal neighbours (face-corner touching) should be connected with 26-connectivity."""
    from uroseg.tools.largest_component import largest_component
    data = np.zeros((10, 10, 10), dtype=np.int16)
    # Two voxels touching only at a corner (diagonal in all three axes)
    data[3, 3, 3] = 1
    data[4, 4, 4] = 1   # 26-connected to (3,3,3) but NOT 6-connected
    result = largest_component(data)
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

def test_crop_to_seg_reduces_image(img_file, seg_file, tmp_path):
    """crop_to_seg returns cropped image smaller than input."""
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped = img.crop_to_seg(seg)
    assert cropped.data.shape[0] <= img.data.shape[0]


def test_crop_margin_expands_bbox(img_file, seg_file, tmp_path):
    """margin N expands the bounding box by N voxels on each side."""
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    no_margin = img.crop_to_seg(seg, margin=0)
    with_margin = img.crop_to_seg(seg, margin=2)
    for i in range(3):
        assert with_margin.data.shape[i] >= no_margin.data.shape[i]


def test_crop_margin_clamped_to_image(img_file, seg_file, tmp_path):
    """Very large margin is clamped to image boundary."""
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped = img.crop_to_seg(seg, margin=100)
    for i in range(3):
        assert cropped.data.shape[i] <= img.data.shape[i]


def test_crop_cli_image_only(img_file, seg_file, tmp_path):
    """CLI writes only the image output; no seg file is created."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'crop',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out', str(out), '--out-suffix', '_crop'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_crop.nii.gz').exists()
    # Seg must NOT be saved
    assert not (out / 'seg_crop.nii.gz').exists()


def test_crop_margin_cli(img_file, seg_file, tmp_path):
    """CLI --margin flag is accepted and produces valid output."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'crop',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out', str(out), '--out-suffix', '_crop', '--margin', '2'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_crop.nii.gz').exists()


# ── preview_jpg ───────────────────────────────────────────────────────────────

def test_make_preview_no_seg(img_file):
    from uroseg.tools.preview import preview
    img = Image.load(img_file)
    result = preview(img.data, seg_data=None)
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.dtype == np.uint8


def test_make_preview_with_seg(img_file, seg_file):
    from uroseg.tools.preview import preview
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    result = preview(img.data, seg_data=seg.data)
    assert result.ndim == 3
    assert result.dtype == np.uint8


def test_make_preview_orient_ax(img_file):
    """--orient ax extracts a slice from axis 2 (shape: [dim0, dim1, 3])."""
    from uroseg.tools.preview import preview
    img = Image.load(img_file)
    result = preview(img.data, orient='ax', sliceloc=0.5)
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.shape[0] == img.data.shape[0]
    assert result.shape[1] == img.data.shape[1]
    assert result.dtype == np.uint8


def test_make_preview_sliceloc_25(img_file):
    """sliceloc=0.25 produces a slice at 25% of the sagittal axis."""
    from uroseg.tools.preview import preview
    img = Image.load(img_file)
    result = preview(img.data, orient='sag', sliceloc=0.25)
    assert result.ndim == 3
    assert result.dtype == np.uint8


def test_parse_label_text_pairs():
    """_parse_label_text parses label:text pairs."""
    from uroseg.tools.preview import _parse_label_text
    result = _parse_label_text(['1:L1', '2:L2'])
    assert result == {1: 'L1', 2: 'L2'}


def test_parse_label_text_json(tmp_path):
    """_parse_label_text parses a JSON file."""
    from uroseg.tools.preview import _parse_label_text
    labels_json = tmp_path / 'labels.json'
    labels_json.write_text(json.dumps({'1': 'L1', '2': 'L2'}))
    result = _parse_label_text([str(labels_json)])
    assert result == {1: 'L1', 2: 'L2'}


def test_parse_label_text_empty():
    from uroseg.tools.preview import _parse_label_text
    assert _parse_label_text([]) == {}


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
    assert (out / 'img_preview_sag_0.5.jpg').exists()


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
    assert (out / 'img_preview_sag_0.5.jpg').exists()


def test_preview_cli_orient_cor_sliceloc(img_file, tmp_path):
    """CLI --orient cor --sliceloc 0.3 produces correctly named output."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'preview',
         '--img', str(img_file), '--out', str(out),
         '--out-suffix', '_preview', '--orient', 'cor', '--sliceloc', '0.3'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_preview_cor_0.3.jpg').exists()


def test_preview_cli_label_text_right(img_file, seg_file, tmp_path):
    """CLI --label-text-right 1:L1 is accepted and produces output."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'preview',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out', str(out), '--out-suffix', '_preview',
         '--label-text-right', '1:L1', '2:L2'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_preview_sag_0.5.jpg').exists()


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
    from uroseg.tools.transform_seg2image import transform_seg2image
    # create a differently-shaped seg
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2:8, 2:8, 2:8] = 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    seg_small = Image(data, affine, None)
    ref = Image.load(img_file)
    result = transform_seg2image(seg_small, ref, interpolation='nearest')
    assert result.data.shape == ref.data.shape


def test_transform_seg2image_nearest_regression(img_file, seg_file, tmp_path):
    """Regression: --interpolation nearest produces integer segmentation output."""
    from uroseg.tools.transform_seg2image import transform_seg2image
    seg = Image.load(seg_file)
    ref = Image.load(img_file)
    result = transform_seg2image(seg, ref, interpolation='nearest')
    assert result.data.shape == ref.data.shape
    assert np.issubdtype(result.data.dtype, np.integer) or result.data.dtype == np.uint8


def test_transform_seg2image_linear_produces_float(img_file, seg_file, tmp_path):
    """--interpolation linear produces float32 output."""
    from uroseg.tools.transform_seg2image import transform_seg2image
    seg = Image.load(seg_file)
    ref = Image.load(img_file)
    result = transform_seg2image(seg, ref, interpolation='linear')
    assert result.data.shape == ref.data.shape
    assert result.data.dtype == np.float32


def test_transform_seg2image_label_single_voxel(tmp_path):
    """--interpolation label places single-voxel landmarks at center of mass."""
    from uroseg.tools.transform_seg2image import transform_seg2image
    import nibabel as nib
    # Seg: 20^3 at 1mm, two single-voxel labels
    seg_data = np.zeros((20, 20, 20), dtype=np.uint8)
    seg_data[5, 5, 5] = 1
    seg_data[15, 15, 15] = 2
    seg = Image(seg_data, np.eye(4), nib.Nifti1Header())
    # Ref: same space (should round-trip)
    ref_data = np.zeros((20, 20, 20), dtype=np.int16)
    ref = Image(ref_data, np.eye(4), nib.Nifti1Header())
    result = transform_seg2image(seg, ref, interpolation='label')
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


# ── public API functions ──────────────────────────────────────────────────────

def test_map_labels_public_api(seg_file, tmp_path):
    from uroseg.tools.map_labels import map_labels_file
    out = tmp_path / 'out.nii.gz'
    result = map_labels_file(seg_file, out, map={1: 10, 2: 20})
    assert result == out
    assert out.exists()
    img = Image.load(out)
    assert img.data[3, 3, 3] == 10


def test_resample_public_api(img_file, tmp_path):
    from uroseg.tools.resample import resample_file
    out = tmp_path / 'out.nii.gz'
    result = resample_file(img_file, out, mm=2.0)
    assert result == out
    assert out.exists()


def test_reorient_public_api(img_file, tmp_path):
    from uroseg.tools.reorient import reorient_file
    out = tmp_path / 'out.nii.gz'
    result = reorient_file(img_file, out)
    assert result == out
    assert out.exists()


def test_largest_component_public_api(seg_file, tmp_path):
    from uroseg.tools.largest_component import largest_component_file
    out = tmp_path / 'out.nii.gz'
    result = largest_component_file(seg_file, out)
    assert result == out
    assert out.exists()


def test_map_labels_dir_public_api(tmp_path):
    from uroseg.tools.map_labels import map_labels_dir
    import nibabel as nib
    in_dir = tmp_path / 'in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    data = np.zeros((5, 5, 5), dtype=np.int16)
    data[1, 1, 1] = 1
    for i in range(2):
        nib.save(nib.Nifti1Image(data, np.eye(4)), in_dir / f'seg{i}.nii.gz')
    map_labels_dir(in_dir, out_dir, map={1: 99}, out_suffix='_mapped', n_jobs=1)
    assert len(list(out_dir.glob('*.nii.gz'))) == 2
