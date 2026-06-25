import argparse
import pytest
from unittest.mock import MagicMock, patch
from uroseg.utils.inference_utils import add_common_inference_args
import uroseg.resources.models.prostate as prostate_mod
import uroseg.resources.models.bladder as bladder_mod


def test_add_common_inference_args_required():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser)
    args = parser.parse_args(['-i', 'img.nii.gz', '-o', 'out/'])
    assert args.img == 'img.nii.gz'
    assert args.out == 'out/'


def test_add_common_inference_args_defaults():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser)
    args = parser.parse_args(['-i', 'img.nii.gz', '-o', 'out/'])
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
        add_common_inference_args(parser)
        assert parser.prog == 'uroseg prostate'


def test_load_model_module_called_for_valid_organ():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    assert mod.MODEL.name == 'prostate'
    assert mod.NNUNET_TASK == 'Dataset101_Prostate'


def test_load_model_module_invalid_organ_raises():
    from uroseg.utils.utils import load_model_module
    with pytest.raises(ValueError):
        load_model_module('nonexistent_organ_xyz')
