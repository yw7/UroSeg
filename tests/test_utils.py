import os
import pytest
from pathlib import Path
from uroseg.utils.utils import (
    collect_niftis,
    build_output_path,
    build_pairs,
    resolve_data_path,
    get_model,
    get_all_models,
)


def test_collect_niftis_single_file(nifti_file):
    result = collect_niftis(nifti_file)
    assert result == [Path(nifti_file)]


def test_collect_niftis_folder(nifti_folder):
    result = collect_niftis(nifti_folder)
    assert len(result) == 3
    assert all(p.suffix == '.gz' for p in result)
    assert result == sorted(result)


def test_collect_niftis_empty_folder(tmp_path):
    result = collect_niftis(tmp_path)
    assert result == []


def test_build_output_path_nii_gz(nifti_file, tmp_path):
    out = build_output_path(Path(nifti_file), tmp_path, prefix='', suffix='_seg')
    assert out == tmp_path / 'test_seg.nii.gz'


def test_build_output_path_nii(tmp_path):
    inp = tmp_path / 'case001.nii'
    out = build_output_path(inp, tmp_path, prefix='pred_', suffix='')
    assert out == tmp_path / 'pred_case001.nii.gz'


def test_build_pairs_basic(nifti_folder, tmp_path):
    pairs = build_pairs(nifti_folder, tmp_path, '_seg', '', overwrite=True)
    assert len(pairs) == 3
    for inp, out in pairs:
        assert out.parent == tmp_path
        assert out.name.endswith('_seg.nii.gz')


def test_build_pairs_skip_existing(nifti_folder, tmp_path):
    out_path = build_output_path(
        list(collect_niftis(nifti_folder))[0], tmp_path, '', '_seg'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch()
    pairs = build_pairs(nifti_folder, tmp_path, '_seg', '', overwrite=False)
    assert len(pairs) == 2


def test_build_pairs_overwrite(nifti_folder, tmp_path):
    for inp in collect_niftis(nifti_folder):
        build_output_path(inp, tmp_path, '', '_seg').parent.mkdir(parents=True, exist_ok=True)
        build_output_path(inp, tmp_path, '', '_seg').touch()
    pairs = build_pairs(nifti_folder, tmp_path, '_seg', '', overwrite=True)
    assert len(pairs) == 3


def test_resolve_data_path_default(monkeypatch):
    monkeypatch.delenv('UROSEG_DATA', raising=False)
    path = resolve_data_path()
    assert path == Path.home() / 'uroseg'


def test_resolve_data_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv('UROSEG_DATA', str(tmp_path))
    path = resolve_data_path()
    assert path == tmp_path


def test_resolve_data_path_arg(tmp_path):
    path = resolve_data_path(str(tmp_path))
    assert path == tmp_path


def test_get_model_prostate():
    model = get_model('prostate')
    assert model['name'] == 'prostate'
    assert 'labels' in model
    assert 'nnunet_task' in model
    assert 'channel_names' in model
    assert 'regions_class_order' in model


def test_get_model_bladder():
    model = get_model('bladder')
    assert model['name'] == 'bladder'
    assert '1' in model['labels']
    assert 'regions_class_order' not in model


def test_get_model_unknown():
    with pytest.raises(Exception):
        get_model('nonexistent_organ')


def test_get_all_models():
    models = get_all_models()
    assert 'prostate' in models
    assert 'bladder' in models
    assert all('labels' in m for m in models.values())
