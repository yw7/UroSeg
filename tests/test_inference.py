from __future__ import annotations
import argparse
import pytest
from unittest.mock import MagicMock, patch
import uroseg.models.prostate as prostate_mod
import uroseg.models.bladder as bladder_mod


def test_add_inference_args_positional():
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz', 'out/'])
    assert args.img == 'img.nii.gz'
    assert args.out == 'out/'


def test_add_inference_args_out_defaults_to_dot():
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz'])
    assert args.img == 'img.nii.gz'
    assert args.out == '.'


def test_add_inference_args_defaults():
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz', 'out/'])
    assert args.fold == 0
    assert args.device == 'cuda'
    assert args.out_suffix == '_seg'
    assert args.out_prefix == ''
    assert args.overwrite is False
    assert args.quiet is False


def test_prostate_module_has_main():
    assert callable(prostate_mod.main)


def test_bladder_module_has_main():
    assert callable(bladder_mod.main)


def test_prostate_main_parser_prog():
    with patch('sys.argv', ['uroseg', '-h']):
        parser = argparse.ArgumentParser(prog='uroseg prostate')
        from uroseg.nnunet.predict import add_inference_args
        add_inference_args(parser)
        assert parser.prog == 'uroseg prostate'


def test_prostate_parser_has_description():
    import argparse
    from uroseg.models.prostate import Prostate
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser(prog='uroseg prostate', description=Prostate.description)
    add_inference_args(parser)
    assert Prostate.description in parser.description


def test_run_predict_cli_auto_installs_if_missing(tmp_path):
    """run_predict_cli auto-installs when model not found, then proceeds."""
    import nibabel as nib
    import numpy as np
    from unittest.mock import patch, MagicMock
    from uroseg.nnunet.predict import run_predict_cli
    from uroseg.models.prostate import Prostate

    inp = tmp_path / 'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(inp))

    args = argparse.Namespace(
        img=str(inp), out=str(tmp_path / 'out'),
        fold=0, device='cpu', out_suffix='_seg', out_prefix='',
        data_dir=str(tmp_path), overwrite=False, quiet=True,
        max_workers=1,
    )

    model = Prostate()
    installed = []

    def fake_install(data_dir):
        installed.append(data_dir)
        (data_dir / 'prostate' / 'r0').mkdir(parents=True)

    fake_model_dir = MagicMock()

    with patch.object(model, 'install', side_effect=fake_install), \
         patch('uroseg.models.base._find_model_dir',
               side_effect=[FileNotFoundError('not found'), fake_model_dir]), \
         patch('uroseg.nnunet.helpers.run_predict'):
        run_predict_cli(model, args)

    assert len(installed) == 1, "install() should have been called once"


def test_load_model_module_called_for_valid_organ():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    assert mod.MODEL.name == 'prostate'
    assert mod.NNUNET_TASK == 'Dataset101_Prostate'


def test_load_model_module_invalid_organ_raises():
    from uroseg.utils.utils import load_model_module
    with pytest.raises(ValueError):
        load_model_module('nonexistent_organ_xyz')
