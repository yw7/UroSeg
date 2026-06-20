import sys
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from uroseg.commands.inference import build_inference_parser, resolve_organ


def test_build_inference_parser_has_required_args():
    parser = build_inference_parser()
    args = parser.parse_args(['prostate', '--img', 'in/', '--out', 'out/'])
    assert args.organ == 'prostate'
    assert args.img == 'in/'
    assert args.out == 'out/'


def test_build_inference_parser_defaults():
    parser = build_inference_parser()
    args = parser.parse_args(['bladder', '--img', 'in/', '--out', 'out/'])
    assert args.fold == 0
    assert args.out_suffix == '_seg'
    assert args.out_prefix == ''


def test_resolve_organ_valid():
    model = resolve_organ('prostate')
    assert model['name'] == 'prostate'


def test_resolve_organ_invalid():
    with pytest.raises(SystemExit):
        resolve_organ('nonexistent_organ_xyz')
