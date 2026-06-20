import subprocess
import sys
import pytest


def run_uroseg(*args):
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', *args],
        capture_output=True, text=True
    )
    return result


def test_list_shows_models():
    result = run_uroseg('list')
    assert result.returncode == 0
    assert 'prostate' in result.stdout
    assert 'bladder' in result.stdout


def test_no_args_exits_nonzero():
    result = run_uroseg()
    assert result.returncode != 0


def test_unknown_flag_exits_nonzero():
    result = run_uroseg('--unknown-flag')
    assert result.returncode != 0


def test_unknown_organ_exits_nonzero():
    result = run_uroseg('nonexistent_organ', '--img', 'x', '--out', 'y')
    assert result.returncode != 0
