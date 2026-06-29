from __future__ import annotations
import csv
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path


def _make_seg(tmp_path: Path, data: np.ndarray, zooms=(2.0, 2.0, 2.0), name='seg.nii.gz') -> Path:
    affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(zooms)
    path = tmp_path / name
    nib.save(img, str(path))
    return path


def test_volume_single_label(tmp_path):
    from uroseg.tools.volume import volume
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0:2, 0:2, 0:2] = 1  # 8 voxels, each 2×2×2 = 8 mm³ → total 64 mm³
    seg = _make_seg(tmp_path, data)
    result = volume(seg, {'bladder': 1})
    assert 'bladder' in result
    assert abs(result['bladder'] - 64.0) < 1e-6


def test_volume_skips_background(tmp_path):
    from uroseg.tools.volume import volume
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1, 1, 1] = 1
    seg = _make_seg(tmp_path, data)
    result = volume(seg, {'background': 0, 'bladder': 1})
    assert 'background' not in result
    assert 'bladder' in result


def test_volume_multi_label(tmp_path):
    from uroseg.tools.volume import volume
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0, 0, 0] = 1  # 1 voxel
    data[1, 1, 1] = 2  # 1 voxel
    data[2, 2, 2] = 3  # 1 voxel
    seg = _make_seg(tmp_path, data, zooms=(1.0, 1.0, 1.0))
    result = volume(seg, {'prostate': [1, 2, 3], 'prostate_tz': 2})
    # prostate includes labels 1,2,3 → 3 voxels × 1mm³ = 3 mm³
    assert abs(result['prostate'] - 3.0) < 1e-6
    # prostate_tz is label 2 → 1 voxel × 1mm³ = 1 mm³
    assert abs(result['prostate_tz'] - 1.0) < 1e-6


def test_volume_1mm_isotropic(tmp_path):
    from uroseg.tools.volume import volume
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 1  # 27 voxels @ 1mm³
    seg = _make_seg(tmp_path, data, zooms=(1.0, 1.0, 1.0))
    result = volume(seg, {'region': 1})
    assert abs(result['region'] - 27.0) < 1e-6


def test_volume_dir_creates_csv(tmp_path):
    from uroseg.tools.volume import volume_dir
    seg_dir = tmp_path / 'segs'
    seg_dir.mkdir()
    for i in range(3):
        data = np.zeros((4, 4, 4), dtype=np.uint8)
        data[0, 0, 0] = 1
        _make_seg(seg_dir, data, zooms=(1.0, 1.0, 1.0), name=f'case{i:02d}.nii.gz')

    out_csv = tmp_path / 'volumes.csv'
    volume_dir(seg_dir, out_csv, {'bladder': 1}, quiet=True)

    assert out_csv.exists()
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        assert 'filename' in row
        assert 'bladder' in row
        assert abs(float(row['bladder']) - 1.0) < 1e-6


def test_volume_dir_skips_existing(tmp_path):
    from uroseg.tools.volume import volume_dir
    seg_dir = tmp_path / 'segs'
    seg_dir.mkdir()
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    _make_seg(seg_dir, data, name='a.nii.gz')

    out_csv = tmp_path / 'out.csv'
    out_csv.write_text('sentinel')
    volume_dir(seg_dir, out_csv, {'bladder': 1}, overwrite=False, quiet=True)
    assert out_csv.read_text() == 'sentinel'


def test_volume_dir_overwrites(tmp_path):
    from uroseg.tools.volume import volume_dir
    seg_dir = tmp_path / 'segs'
    seg_dir.mkdir()
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0, 0, 0] = 1
    _make_seg(seg_dir, data, zooms=(1.0, 1.0, 1.0), name='a.nii.gz')

    out_csv = tmp_path / 'out.csv'
    out_csv.write_text('old content')
    volume_dir(seg_dir, out_csv, {'bladder': 1}, overwrite=True, quiet=True)
    assert out_csv.read_text() != 'old content'


def test_volume_cli_single_file_stdout(tmp_path, capsys):
    from uroseg.tools.volume import main
    import sys
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1, 1, 1] = 1
    seg = _make_seg(tmp_path, data, zooms=(1.0, 1.0, 1.0))
    sys.argv = ['volume', '--seg', str(seg), '--labels', '{"bladder": 1}']
    main()
    out = capsys.readouterr().out
    assert 'bladder' in out
    assert 'mm³' in out


def test_volume_cli_single_file_to_csv(tmp_path):
    from uroseg.tools.volume import main
    import sys
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1, 1, 1] = 1
    seg = _make_seg(tmp_path, data, zooms=(1.0, 1.0, 1.0))
    out_csv = tmp_path / 'out.csv'
    sys.argv = ['volume', '--seg', str(seg), '--out', str(out_csv), '--labels', '{"bladder": 1}']
    main()
    assert out_csv.exists()
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 1
    assert rows[0]['filename'] == seg.stem.removesuffix('.nii')


def test_volume_public_api():
    import uroseg
    assert hasattr(uroseg, 'volume')
    assert hasattr(uroseg, 'volume_dir')
    assert callable(uroseg.volume)
    assert callable(uroseg.volume_dir)
