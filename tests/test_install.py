# tests/test_install.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_extract_release_id():
    from uroseg.models.base import _extract_release_id
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    assert _extract_release_id(url) == 'r20260101'


def test_find_model_dir_returns_newest(tmp_path):
    from uroseg.models.base import _find_model_dir
    (tmp_path / 'prostate' / 'r20260101').mkdir(parents=True)
    (tmp_path / 'prostate' / 'r20260601').mkdir(parents=True)
    result = _find_model_dir('prostate', tmp_path)
    assert result.name == 'r20260601'


def test_find_model_dir_not_found(tmp_path):
    from uroseg.models.base import _find_model_dir
    with pytest.raises(FileNotFoundError):
        _find_model_dir('prostate', tmp_path)


def test_install_skips_no_url(tmp_path, capsys):
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'test'; description = ''; weights_url = ''; labels = {}
    M().install(tmp_path)
    out = capsys.readouterr().out
    assert 'skip' in out.lower() or 'no weights' in out.lower()


def test_install_skips_if_already_installed(tmp_path, capsys):
    from uroseg.models.prostate import Prostate
    release_id = 'r20260101'
    (tmp_path / 'prostate' / release_id).mkdir(parents=True)
    Prostate().install(tmp_path)
    assert 'already installed' in capsys.readouterr().out.lower()


def test_install_downloads_to_temp_and_extracts(tmp_path):
    from uroseg.models.prostate import Prostate
    with patch('uroseg.models.base._download_zip') as mock_dl, \
         patch('uroseg.models.base._extract_zip') as mock_ex:
        mock_dl.return_value = tmp_path / 'x.zip'
        Prostate().install(tmp_path)
        assert mock_dl.called
        assert mock_ex.called
        # dest must be data_dir/prostate/<release_id>/
        dest = mock_ex.call_args[0][1]
        assert dest.parent.name == 'prostate'
