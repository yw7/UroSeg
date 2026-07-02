from __future__ import annotations
import argparse
import pytest
from unittest.mock import MagicMock, patch
import uroseg.models.prostate as prostate_mod
import uroseg.models.bladder as bladder_mod


def test_add_inference_args_positional():
    from uroseg.models.base import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz', 'out/'])
    assert args.img == 'img.nii.gz'
    assert args.out == 'out/'


def test_add_inference_args_out_defaults_to_none():
    from uroseg.models.base import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz'])
    assert args.img == 'img.nii.gz'
    assert args.out is None


def test_add_inference_args_defaults():
    from uroseg.models.base import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz', 'out/'])
    assert args.fold == 0
    assert args.device == 'cuda'
    assert args.out_suffix == ''
    assert args.out_prefix == ''
    assert args.overwrite is False
    assert args.quiet is False
    assert args.iso is False


def test_prostate_module_has_main():
    assert callable(prostate_mod.main)


def test_bladder_module_has_main():
    assert callable(bladder_mod.main)


def test_prostate_main_parser_prog():
    with patch('sys.argv', ['uroseg', '-h']):
        from uroseg.models.base import add_inference_args
        parser = argparse.ArgumentParser(prog='uroseg prostate')
        add_inference_args(parser)
        assert parser.prog == 'uroseg prostate'


def test_prostate_parser_has_description():
    from uroseg.models.prostate import Prostate
    from uroseg.models.base import add_inference_args
    parser = argparse.ArgumentParser(prog='uroseg prostate', description=Prostate.description)
    add_inference_args(parser)
    assert Prostate.description in parser.description


def test_add_inference_args_iso_flag():
    """--iso flag is parsed correctly."""
    from uroseg.models.base import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz', '--iso'])
    assert args.iso is True


def test_predict_cli_no_tempdir(tmp_path):
    """predict_cli does not create a temporary directory."""
    import numpy as np
    import nibabel as nib
    import tempfile as tempfile_mod
    from uroseg.models.prostate import Prostate
    from uroseg.utils.image import Image

    inp = tmp_path / 'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(inp))

    args = argparse.Namespace(
        img=str(inp), out=str(tmp_path / 'out'),
        fold=0, device='cpu', out_suffix='_seg', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )

    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    model = Prostate()

    tempdir_calls = []
    original_tempdir = tempfile_mod.TemporaryDirectory
    def tracking_tempdir(*a, **kw):
        tempdir_calls.append(True)
        return original_tempdir(*a, **kw)

    with patch.object(model, 'init_predictor', return_value=MagicMock()), \
         patch.object(model, 'predict_image',
                      return_value=Image(fake_seg, np.eye(4),
                                         nib.Nifti1Image(fake_seg, np.eye(4)).header)), \
         patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('tempfile.TemporaryDirectory', side_effect=tracking_tempdir):
        model.predict_cli(args)

    assert len(tempdir_calls) == 0, "predict_cli must not use TemporaryDirectory"


def test_predict_cli_init_predictor_called_once(tmp_path):
    """init_predictor is called once regardless of how many input files."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.prostate import Prostate
    from uroseg.utils.image import Image

    for name in ('a.nii.gz', 'b.nii.gz'):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
                 str(tmp_path / name))

    args = argparse.Namespace(
        img=str(tmp_path), out=str(tmp_path / 'out'),
        fold=0, device='cpu', out_suffix='_seg', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )
    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    model = Prostate()
    init_calls = []

    def fake_init(model_dir, fold=0, device='cuda'):
        init_calls.append(1)
        return MagicMock()

    with patch.object(model, 'init_predictor', side_effect=fake_init), \
         patch.object(model, 'predict_image',
                      return_value=Image(fake_seg, np.eye(4),
                                         nib.Nifti1Image(fake_seg, np.eye(4)).header)), \
         patch('uroseg.models.base._find_model_dir', return_value=tmp_path):
        model.predict_cli(args)

    assert len(init_calls) == 1, "init_predictor must be called exactly once"


def test_predict_cli_auto_installs_if_missing(tmp_path):
    """predict_cli auto-installs when model not found, then proceeds."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.prostate import Prostate
    from uroseg.utils.image import Image

    inp = tmp_path / 'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(inp))

    args = argparse.Namespace(
        img=str(inp), out=str(tmp_path / 'out'),
        fold=0, device='cpu', out_suffix='_seg', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )
    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    model = Prostate()
    installed = []

    def fake_install(data_dir):
        installed.append(data_dir)
        (data_dir / 'prostate' / 'r0').mkdir(parents=True)

    fake_model_dir = MagicMock()

    with patch.object(model, 'install', side_effect=fake_install), \
         patch('uroseg.models.base._find_model_dir',
               side_effect=[FileNotFoundError('not found'), fake_model_dir, fake_model_dir]), \
         patch.object(model, 'init_predictor', return_value=MagicMock()), \
         patch.object(model, 'predict_image',
                      return_value=Image(fake_seg, np.eye(4),
                                         nib.Nifti1Image(fake_seg, np.eye(4)).header)):
        model.predict_cli(args)

    assert len(installed) == 1


def test_predict_cli_single_file_no_out_auto_names(tmp_path):
    """Single file + no out arg → <stem>_<model>.nii.gz next to input."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.prostate import Prostate
    from uroseg.utils.image import Image

    inp = tmp_path / 'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(inp))

    args = argparse.Namespace(
        img=str(inp), out=None,
        fold=0, device='cpu', out_suffix='', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )
    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    model = Prostate()

    with patch.object(model, 'init_predictor', return_value=MagicMock()), \
         patch.object(model, 'predict_image',
                      return_value=Image(fake_seg, np.eye(4),
                                         nib.Nifti1Image(fake_seg, np.eye(4)).header)), \
         patch('uroseg.models.base._find_model_dir', return_value=tmp_path):
        model.predict_cli(args)

    assert (tmp_path / 'scan_prostate.nii.gz').exists()


def test_predict_cli_single_file_out_adds_extension(tmp_path):
    """Single file + out without .nii.gz → extension appended."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.prostate import Prostate
    from uroseg.utils.image import Image

    inp = tmp_path / 'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(inp))

    args = argparse.Namespace(
        img=str(inp), out=str(tmp_path / 'result'),
        fold=0, device='cpu', out_suffix='', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )
    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    model = Prostate()

    with patch.object(model, 'init_predictor', return_value=MagicMock()), \
         patch.object(model, 'predict_image',
                      return_value=Image(fake_seg, np.eye(4),
                                         nib.Nifti1Image(fake_seg, np.eye(4)).header)), \
         patch('uroseg.models.base._find_model_dir', return_value=tmp_path):
        model.predict_cli(args)

    assert (tmp_path / 'result.nii.gz').exists()


def test_load_model_module_called_for_valid_organ():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    assert mod.MODEL.name == 'prostate'
    assert mod.NNUNET_TASK == 'Dataset101_Prostate'


def test_load_model_module_invalid_organ_raises():
    from uroseg.utils.utils import load_model_module
    with pytest.raises(ValueError):
        load_model_module('nonexistent_organ_xyz')
