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


def test_segmodel_init_predictor_raises_not_implemented():
    """init_predictor() raises NotImplementedError on the base class."""
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().init_predictor(Path('/tmp'), fold=0, device='cpu')


def test_segmodel_predict_image_raises_not_implemented():
    """predict_image() raises NotImplementedError on the base class."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    data = np.zeros((5, 5, 5), dtype=np.float32)
    img = Image(data=data, affine=np.eye(4), header=nib.Nifti1Image(data, np.eye(4)).header)
    with pytest.raises(NotImplementedError):
        M().predict_image(None, img)


def test_segmodel_predict_raises_via_init_predictor(tmp_path):
    """predict() propagates NotImplementedError from init_predictor()."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch
    from uroseg.models.base import SegModel
    inp = tmp_path / 'a.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), str(inp))
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('uroseg.models.base._resolve_data_dir', return_value=tmp_path):
        with pytest.raises(NotImplementedError):
            M().predict(inp, tmp_path / 'out')


def test_segmodel_predict_applies_largest_component(tmp_path):
    """predict() calls keep_largest_component when post_largest_component=True."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock, patch
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    inp = tmp_path / 'img.nii.gz'
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(inp))

    seg_data = np.zeros((8, 8, 8), dtype=np.uint8)
    seg_data[3:5, 3:5, 3:5] = 1

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        post_largest_component = True
        def init_predictor(self, model_dir, fold=0, device='cuda'):
            return MagicMock()
        def predict_image(self, predictor, img):
            return Image(data=seg_data.copy(), affine=np.eye(4),
                         header=nib.Nifti1Image(seg_data, np.eye(4)).header)

    with patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('uroseg.models.base._resolve_data_dir', return_value=tmp_path):
        out = M().predict(inp, tmp_path / 'out')
    assert out.exists()


def test_segmodel_predict_iso_false_resamples_back(tmp_path):
    """predict() with iso=False resamples seg back to original affine."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock, patch
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'img.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def init_predictor(self, model_dir, fold=0, device='cuda'):
            return MagicMock()
        def predict_image(self, predictor, img):
            # img is already 1mm — return seg in that space
            seg_data = np.zeros_like(img.data, dtype=np.uint8)
            return Image(data=seg_data, affine=img.affine, header=img.header)

    with patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('uroseg.models.base._resolve_data_dir', return_value=tmp_path):
        out = M().predict(inp, tmp_path / 'out', iso=False)
    result = Image.load(out)
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 2.0, atol=0.2)


def test_segmodel_predict_iso_true_keeps_model_space(tmp_path):
    """predict() with iso=True leaves seg in 1mm space."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock, patch
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'img.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def init_predictor(self, model_dir, fold=0, device='cuda'):
            return MagicMock()
        def predict_image(self, predictor, img):
            seg_data = np.zeros_like(img.data, dtype=np.uint8)
            return Image(data=seg_data, affine=img.affine, header=img.header)

    with patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('uroseg.models.base._resolve_data_dir', return_value=tmp_path):
        out = M().predict(inp, tmp_path / 'out', iso=True)
    result = Image.load(out)
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 1.0, atol=0.2)


def test_nnunet_predict_image_calls_run_inference(tmp_path):
    """NNUNetSegModel.predict_image calls _run_inference and returns Image."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch, MagicMock
    from uroseg.models.base import NNUNetSegModel
    from uroseg.utils.image import Image

    class TestModel(NNUNetSegModel):
        name = 'test'; description = ''; weights_url = ''
        labels = {}; nnunet_task = 'Dataset999_Test'

    data = np.ones((8, 8, 8), dtype=np.float32)
    img = Image(data=data, affine=np.eye(4), header=nib.Nifti1Image(data, np.eye(4)).header)
    fake_seg = np.zeros((8, 8, 8), dtype=np.uint8)

    with patch('uroseg.nnunet.helpers._run_inference', return_value=fake_seg) as mock_infer:
        result = TestModel().predict_image(MagicMock(), img)

    assert isinstance(result, Image)
    mock_infer.assert_called_once()
    np.testing.assert_array_equal(result.data, fake_seg)


def test_segmodel_predict_dir_writes_outputs(tmp_path):
    import nibabel as nib
    import numpy as np
    from unittest.mock import MagicMock, patch
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    for name in ('a.nii.gz', 'b.nii.gz'):
        nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)),
                 tmp_path / name)

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def init_predictor(self, model_dir, fold=0, device='cuda'):
            return MagicMock()
        def predict_image(self, predictor, img):
            return Image(data=np.zeros_like(img.data, dtype=np.uint8),
                         affine=img.affine, header=img.header)

    with patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('uroseg.models.base._resolve_data_dir', return_value=tmp_path):
        M().predict_dir(tmp_path, tmp_path / 'out')

    assert len(list((tmp_path / 'out').glob('*.nii.gz'))) == 2


def test_prostate_class_attrs():
    from uroseg.models.prostate import Prostate
    p = Prostate()
    assert p.name == 'prostate'
    assert p.nnunet_task == 'Dataset101_Prostate'
    assert p.labels['background'] == 0
    assert isinstance(p.labels['prostate'], list)


def test_bladder_class_attrs():
    from uroseg.models.bladder import Bladder
    b = Bladder()
    assert b.name == 'bladder'
    assert b.labels['bladder'] == 1


def test_get_model_returns_prostate():
    from uroseg.models import get_model
    from uroseg.models.prostate import Prostate
    m = get_model('prostate')
    assert isinstance(m, Prostate)
    assert m.name == 'prostate'


def test_list_models():
    from uroseg.models import list_models
    models = list_models()
    assert 'prostate' in models
    assert 'bladder' in models


def test_get_model_unknown_raises():
    from uroseg.models import get_model
    with pytest.raises(ValueError, match='Unknown model'):
        get_model('nonexistent_xyz')


def test_prostate_main_is_callable():
    from uroseg.models.prostate import main
    assert callable(main)


def test_compat_model_attr():
    from uroseg.models.prostate import MODEL, NNUNET_TASK
    assert MODEL.name == 'prostate'
    assert NNUNET_TASK == 'Dataset101_Prostate'


def test_run_predict_array_returns_image(tmp_path):
    """run_predict_array calls predict_single_npy_array and returns Image in same space."""
    import sys
    import numpy as np
    import nibabel as nib
    from uroseg.utils.image import Image
    from uroseg.nnunet.helpers import run_predict_array

    # Build a fake model_dir with fold_0 subdir so glob finds it
    fold_dir = tmp_path / 'trainer' / 'fold_0'
    fold_dir.mkdir(parents=True)

    data = np.ones((10, 10, 10), dtype=np.float32)
    affine = np.eye(4)
    img = Image(data=data, affine=affine, header=nib.Nifti1Image(data, affine).header)

    fake_seg = np.zeros((10, 10, 10), dtype=np.uint8)
    fake_seg[3:7, 3:7, 3:7] = 1

    mock_predictor = MagicMock()
    mock_predictor.predict_single_npy_array.return_value = fake_seg

    # Create mock for nnunetv2 module hierarchy
    mock_nnunetv2 = MagicMock()
    mock_nnunetv2.inference.predict_from_raw_data.nnUNetPredictor = MagicMock(return_value=mock_predictor)

    with patch.dict(sys.modules, {
        'nnunetv2': mock_nnunetv2,
        'nnunetv2.inference': mock_nnunetv2.inference,
        'nnunetv2.inference.predict_from_raw_data': mock_nnunetv2.inference.predict_from_raw_data,
    }):
        result = run_predict_array(tmp_path, img, fold=0, device='cpu')

    assert isinstance(result, Image)
    assert result.data.shape == (10, 10, 10)
    np.testing.assert_array_equal(result.data, fake_seg)
    np.testing.assert_array_equal(result.affine, img.affine)

    # Verify predict_single_npy_array was called with correct shape and spacing
    call_args = mock_predictor.predict_single_npy_array.call_args
    assert call_args[0][0].shape == (1, 10, 10, 10)   # (1, x, y, z)
    assert call_args[0][1]['spacing'] == [1.0, 1.0, 1.0]


def test_suppress_nnunet_silences_print(capsys):
    """_suppress_nnunet redirects stdout so prints are hidden."""
    from uroseg.nnunet.helpers import _suppress_nnunet
    with _suppress_nnunet():
        print("should be hidden")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_suppress_nnunet_sets_env_vars():
    """_suppress_nnunet sets dummy env vars for nnunet keys."""
    import os
    from uroseg.nnunet.helpers import _suppress_nnunet
    keys = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
    for k in keys:
        os.environ.pop(k, None)
    with _suppress_nnunet():
        for k in keys:
            assert os.environ.get(k) == ''


def test_suppress_nnunet_restores_env(monkeypatch):
    """_suppress_nnunet restores original env var values on exit."""
    import os
    from uroseg.nnunet.helpers import _suppress_nnunet
    monkeypatch.setenv('nnUNet_raw', 'original_value')
    with _suppress_nnunet():
        pass
    assert os.environ['nnUNet_raw'] == 'original_value'


def test_init_predictor_initializes_from_folder(tmp_path):
    """_init_predictor calls initialize_from_trained_model_folder on the predictor."""
    import sys
    from unittest.mock import MagicMock, patch
    from uroseg.nnunet.helpers import _init_predictor

    fold_dir = tmp_path / 'trainer' / 'fold_0'
    fold_dir.mkdir(parents=True)
    (fold_dir / 'checkpoint_final.pth').touch()

    mock_predictor = MagicMock()
    mock_nnunet = MagicMock()
    mock_nnunet.nnUNetPredictor.return_value = mock_predictor

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.device.return_value = MagicMock()

    with patch.dict(sys.modules, {
        'torch': mock_torch,
        'nnunetv2': MagicMock(),
        'nnunetv2.inference': MagicMock(),
        'nnunetv2.inference.predict_from_raw_data': mock_nnunet,
    }):
        result = _init_predictor(tmp_path, fold=0, device='cpu')

    assert result is mock_predictor
    mock_predictor.initialize_from_trained_model_folder.assert_called_once()


def test_run_inference_calls_predict_single(tmp_path):
    """_run_inference passes (1,x,y,z) array and spacing; returns seg array."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock
    from uroseg.utils.image import Image
    from uroseg.nnunet.helpers import _run_inference

    data = np.ones((8, 8, 8), dtype=np.float32)
    img = Image(data=data, affine=np.eye(4), header=nib.Nifti1Image(data, np.eye(4)).header)
    expected = np.zeros((8, 8, 8), dtype=np.uint8)

    mock_predictor = MagicMock()
    mock_predictor.predict_single_npy_array.return_value = expected

    result = _run_inference(mock_predictor, img)

    assert result is expected
    call_args = mock_predictor.predict_single_npy_array.call_args
    assert call_args[0][0].shape == (1, 8, 8, 8)   # (1, x, y, z)
    assert call_args[0][1]['spacing'] == [1.0, 1.0, 1.0]
