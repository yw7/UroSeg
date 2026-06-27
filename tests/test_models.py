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
        M().predict_image(img)


def test_segmodel_predict_raises_via_predict_image(tmp_path):
    """predict() delegates to predict_image(); NotImplementedError propagates."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    inp = tmp_path / 'a.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), str(inp))
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().predict(inp, tmp_path / 'out')


def test_segmodel_predict_io_wrapper_iso_false(tmp_path):
    """predict() with iso=False transforms seg back to original space."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    # 2 mm isotropic input image, shape (8, 8, 8)
    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'input.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def predict_image(self, img, **kwargs):
            # Simulate model that returns seg in 1 mm iso space
            resampled = img.resample((1.0, 1.0, 1.0))
            seg_data = np.zeros_like(resampled.data, dtype=np.uint8)
            seg_data[5:11, 5:11, 5:11] = 1
            return Image(seg_data, resampled.affine, resampled.header)

    out_path = M().predict(inp, tmp_path / 'out', iso=False)
    assert out_path.exists()
    assert out_path.name == 'input.nii.gz'
    result = Image.load(out_path)
    # iso=False: affine diagonal should be ~2 mm (original spacing)
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 2.0, atol=0.2)


def test_segmodel_predict_io_wrapper_iso_true(tmp_path):
    """predict() with iso=True leaves seg in the model's working space."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'input.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def predict_image(self, img, **kwargs):
            resampled = img.resample((1.0, 1.0, 1.0))
            seg_data = np.zeros_like(resampled.data, dtype=np.uint8)
            return Image(seg_data, resampled.affine, resampled.header)

    out_path = M().predict(inp, tmp_path / 'out', iso=True)
    assert out_path.exists()
    result = Image.load(out_path)
    # iso=True: stays in 1 mm space — diagonal should be ~1 mm
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 1.0, atol=0.2)


def test_nnunet_predict_image_runs_pipeline(tmp_path):
    """NNUNetSegModel.predict_image reorients, resamples, calls run_predict_array, keeps largest CC."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch
    from uroseg.models.base import NNUNetSegModel
    from uroseg.utils.image import Image

    class TestModel(NNUNetSegModel):
        name = 'test'; description = ''; weights_url = ''
        labels = {}; nnunet_task = 'Dataset999_Test'

    # 2 mm isotropic input
    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = Image(data=data, affine=affine, header=nib.Nifti1Image(data, affine).header)

    def fake_run_predict_array(model_dir, img_1mm, fold, device):
        # Return a seg with same shape as 1 mm resampled input
        seg = np.zeros(img_1mm.data.shape, dtype=np.uint8)
        # Two disconnected blobs — keep_largest_component will reduce to one
        mid = np.array(seg.shape) // 2
        seg[mid[0]-2:mid[0]+2, mid[1]-2:mid[1]+2, mid[2]-2:mid[2]+2] = 1
        seg[0:2, 0:2, 0:2] = 1  # small blob that should be removed
        return Image(data=seg, affine=img_1mm.affine, header=img_1mm.header)

    with patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('uroseg.models.base._resolve_data_dir', return_value=tmp_path), \
         patch('uroseg.nnunet.helpers.run_predict_array', side_effect=fake_run_predict_array):
        result = TestModel().predict_image(img, fold=0, device='cpu')

    assert isinstance(result, Image)
    assert result.data.ndim == 3
    # After keep_largest_component, only one blob remains — corner blob gone
    assert result.data[0, 0, 0] == 0


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
