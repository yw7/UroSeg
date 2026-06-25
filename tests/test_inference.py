import pytest
from unittest.mock import MagicMock, patch
from uroseg.commands.inference import build_inference_parser


def test_build_inference_parser_has_required_args():
    parser = build_inference_parser()
    args = parser.parse_args(['prostate', '--img', 'x.nii.gz', '--out', 'out/'])
    assert args.organ == 'prostate'
    assert args.img == 'x.nii.gz'
    assert args.out == 'out/'


def test_build_inference_parser_defaults():
    parser = build_inference_parser()
    args = parser.parse_args(['bladder', '--img', 'x.nii.gz', '--out', 'out/'])
    assert args.fold == 0
    assert args.device == 'cuda'
    assert args.out_suffix == '_seg'
    assert args.out_prefix == ''


def test_load_model_module_called_for_valid_organ():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    assert mod.MODEL.name == 'prostate'
    assert mod.NNUNET_TASK == 'Dataset101_Prostate'


def test_load_model_module_invalid_organ_raises():
    from uroseg.utils.utils import load_model_module
    with pytest.raises(ValueError):
        load_model_module('nonexistent_organ_xyz')


def test_inference_hook_called():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    sentinel_img = object()
    sentinel_result = object()
    predict_fn = MagicMock(return_value=sentinel_result)
    result = mod.inference(sentinel_img, predict_fn)
    predict_fn.assert_called_once_with(sentinel_img)
    assert result is sentinel_result
