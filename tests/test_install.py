import pytest
from pathlib import Path
from uroseg.models import ModelDef
from uroseg.commands.install import (
    extract_release_id,
    get_install_dir,
    is_installed,
    download_and_extract,
)


def test_extract_release_id():
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    assert extract_release_id(url) == 'r20260101'


def test_get_install_dir(tmp_path):
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    d = get_install_dir('Dataset001_Prostate', url, tmp_path)
    assert d == tmp_path / 'nnUNet' / 'results' / 'r20260101' / 'Dataset001_Prostate'


def test_is_installed_false(tmp_path):
    url = 'https://example.com/releases/download/r20260101/X.zip'
    assert not is_installed('Dataset001_Prostate', url, tmp_path)


def test_is_installed_true(tmp_path):
    url = 'https://example.com/releases/download/r20260101/X.zip'
    install_dir = get_install_dir('Dataset001_Prostate', url, tmp_path)
    install_dir.mkdir(parents=True)
    assert is_installed('Dataset001_Prostate', url, tmp_path)


def test_download_and_extract_no_url(tmp_path, capsys):
    model = ModelDef(name='test', description='', weights_url='', labels={})
    download_and_extract(model, 'Dataset999_Test', tmp_path)
    captured = capsys.readouterr()
    assert 'skip' in captured.out.lower() or 'no weights' in captured.out.lower()
