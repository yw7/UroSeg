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


def test_list_prints_model_names(capsys):
    from uroseg.commands.list_models import main
    main()
    out = capsys.readouterr().out
    assert 'prostate' in out
    assert 'bladder' in out


def test_no_args_shows_help_exits_zero():
    result = run_uroseg()
    assert result.returncode == 0
    assert 'prostate' in result.stdout
    assert 'train nnunet' in result.stdout


def test_help_flag_exits_zero():
    for flag in ('-h', '--help'):
        result = run_uroseg(flag)
        assert result.returncode == 0
        assert 'prostate' in result.stdout
        assert 'resample' in result.stdout


def test_help_contains_commands():
    result = run_uroseg('--help')
    assert 'resample' in result.stdout
    assert 'train nnunet' in result.stdout
    assert 'install' in result.stdout
    assert 'crop' in result.stdout


def test_list_redirects_to_help():
    help_result = run_uroseg('--help')
    list_result = run_uroseg('list')
    assert list_result.returncode == 0
    assert list_result.stdout == help_result.stdout


def test_unknown_flag_exits_nonzero():
    result = run_uroseg('--unknown-flag')
    assert result.returncode != 0


def test_unknown_model_exits_nonzero():
    result = run_uroseg('nonexistent_model', '--img', 'x', '--out', 'y')
    assert result.returncode != 0
    assert 'Unknown model' in result.stderr


def test_list_cli():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'list'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert 'prostate' in result.stdout
    assert 'bladder' in result.stdout


def test_train_no_engine_exits_nonzero():
    result = run_uroseg('train')
    assert result.returncode != 0
    assert 'Engines:' in result.stderr  # new train.py prints _HELP to stderr


def test_train_unknown_engine_exits_nonzero():
    result = run_uroseg('train', 'foo', 'kidney')
    assert result.returncode != 0
    assert 'Unknown engine' in result.stderr


def test_train_help_exits_zero():
    result = run_uroseg('train', '-h')
    assert result.returncode == 0
    assert 'Engines:' in result.stdout  # new train.py prints _HELP; old argparse doesn't print 'Engines:'
