from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_extract_release_id():
    from uroseg.models.base import _extract_release_id
    url = 'https://github.com/x/releases/download/r20260101/X.zip'
    assert _extract_release_id(url) == 'r20260101'


def test_find_model_dir_newest_first(tmp_path):
    from uroseg.models.base import _find_model_dir
    (tmp_path / 'prostate' / 'r20260101').mkdir(parents=True)
    (tmp_path / 'prostate' / 'r20260201').mkdir(parents=True)
    result = _find_model_dir('prostate', tmp_path)
    assert result.name == 'r20260201'


def test_find_model_dir_not_found(tmp_path):
    from uroseg.models.base import _find_model_dir
    with pytest.raises(FileNotFoundError):
        _find_model_dir('nonexistent', tmp_path)


def test_segmodel_install_skips_when_no_url(tmp_path, capsys):
    from uroseg.models.base import SegModel
    class EmptyModel(SegModel):
        name = 'test'; description = 'd'; weights_url = ''; labels = {}
    EmptyModel().install(tmp_path)
    assert 'skip' in capsys.readouterr().out.lower()


def test_segmodel_install_skips_when_already_installed(tmp_path, capsys):
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'x'; description = ''; weights_url = 'https://h/releases/download/r1/x.zip'; labels = {}
    (tmp_path / 'x' / 'r1').mkdir(parents=True)
    M().install(tmp_path)
    assert 'already installed' in capsys.readouterr().out.lower()


def test_nnunet_segmodel_is_segmodel():
    from uroseg.models.base import SegModel, NNUNetSegModel
    assert issubclass(NNUNetSegModel, SegModel)


def test_segmodel_predict_raises():
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().predict(Path('x.nii.gz'), Path('/tmp'))


def test_segmodel_predict_dir_calls_predict(tmp_path):
    from uroseg.models.base import SegModel
    import nibabel as nib, numpy as np
    f = tmp_path / 'a.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((5,5,5), dtype=np.int16), np.eye(4)), f)

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        calls = []
        def predict(self, input, output_dir, **kwargs):
            M.calls.append(input)

    M().predict_dir(tmp_path, tmp_path / 'out', n_jobs=1)
    assert len(M.calls) == 1
