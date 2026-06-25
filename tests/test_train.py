from __future__ import annotations
import json
import os
from pathlib import Path
from unittest.mock import patch, call, MagicMock
import numpy as np
import nibabel as nib
import pytest


@pytest.fixture(autouse=True)
def _stub_nnunetv2(monkeypatch):
    """nnunetv2 is not installed in the dev environment; stub it so
    auglab.add_trainer can be imported without ModuleNotFoundError."""
    import sys
    monkeypatch.setitem(sys.modules, "nnunetv2", MagicMock())


@pytest.fixture
def prostate_model():
    from uroseg.models import ModelDef
    return ModelDef(
        name="prostate",
        description="Prostate MRI-T2",
        weights_url="https://example.com/prostate.zip",
        labels={
            "background": 0,
            "prostate": [1, 2, 3, 4],
            "prostate_pz": 2,
            "prostate_cz": 3,
            "prostate_afs": 4,
        },
    )


@pytest.fixture
def bladder_model():
    from uroseg.models import ModelDef
    return ModelDef(
        name="bladder",
        description="Urinary bladder (CT)",
        weights_url="https://example.com/bladder.zip",
        labels={"background": 0, "bladder": 1},
    )


@pytest.fixture
def images_tr_dir(tmp_path):
    d = tmp_path / "imagesTr"
    d.mkdir()
    for i in range(3):
        data = np.zeros((10, 10, 10), dtype=np.int16)
        nib.save(nib.Nifti1Image(data, np.eye(4)), d / f"case_{i:03d}_0000.nii.gz")
    return d


# ── normalize_labels ─────────────────────────────────────────────────────────

def test_normalize_labels_canonical_passthrough():
    from uroseg.utils.utils import normalize_labels
    inp = {"background": 0, "disc": [1, 2, 3], "disc_C2_C3": 2}
    assert normalize_labels(inp) == inp


def test_normalize_labels_old_int_key_format():
    from uroseg.utils.utils import normalize_labels
    result = normalize_labels({"0": "background", "1": "prostate", "2": "pz"})
    assert result == {"background": 0, "prostate": 1, "pz": 2}


def test_normalize_labels_comma_key():
    from uroseg.utils.utils import normalize_labels
    result = normalize_labels({"1,2,3": "prostate", "2": "pz"})
    assert result["prostate"] == [1, 2, 3]
    assert result["pz"] == 2


def test_normalize_labels_range_key():
    from uroseg.utils.utils import normalize_labels
    result = normalize_labels({"1-4": "disc", "2": "C2_C3"})
    assert result["disc"] == [1, 2, 3, 4]
    assert result["C2_C3"] == 2


def test_normalize_labels_mixed_formats():
    from uroseg.utils.utils import normalize_labels
    result = normalize_labels({
        "background": 0,
        "prostate": [1, 2, 3, 4],
        "prostate_pz": 2,
    })
    assert result["background"] == 0
    assert result["prostate"] == [1, 2, 3, 4]
    assert result["prostate_pz"] == 2


# ── extract_dataset_id ────────────────────────────────────────────────────────

def test_extract_dataset_id_parses_correctly():
    from uroseg.commands.train_nnunet import extract_dataset_id
    assert extract_dataset_id("Dataset001_Prostate") == 1
    assert extract_dataset_id("Dataset010_Bladder") == 10
    assert extract_dataset_id("Dataset123_Kidney") == 123


def test_extract_dataset_id_raises_on_bad_format():
    from uroseg.commands.train_nnunet import extract_dataset_id
    with pytest.raises(ValueError, match="Invalid nnunet_task"):
        extract_dataset_id("ProstateBadName")


# ── generate_dataset_json ─────────────────────────────────────────────────────

def test_generate_dataset_json_simple(bladder_model, images_tr_dir):
    from uroseg.commands.train_nnunet import generate_dataset_json
    result = generate_dataset_json(bladder_model, images_tr_dir)
    assert result["channel_names"] == {"0": "MRI"}
    assert result["labels"] == {"background": 0, "bladder": 1}
    assert result["numTraining"] == 3
    assert result["file_ending"] == ".nii.gz"
    assert "regions_class_order" not in result


def test_generate_dataset_json_with_regions(prostate_model, images_tr_dir):
    from uroseg.commands.train_nnunet import generate_dataset_json
    result = generate_dataset_json(prostate_model, images_tr_dir)
    assert result["regions_class_order"] == [1, 2, 3, 4]
    assert result["channel_names"] == {"0": "MRI"}


# ── setup_nnunet_env ──────────────────────────────────────────────────────────

def test_setup_nnunet_env_sets_vars(tmp_path):
    from uroseg.commands.train_nnunet import setup_nnunet_env
    setup_nnunet_env(tmp_path)
    assert os.environ["nnUNet_raw"] == str(tmp_path / "nnUNet" / "raw")
    assert os.environ["nnUNet_preprocessed"] == str(tmp_path / "nnUNet" / "preprocessed")
    assert os.environ["nnUNet_results"] == str(tmp_path / "nnUNet" / "results")


# ── main() integration — subprocess mocked ───────────────────────────────────

def test_train_generates_dataset_json(tmp_path):
    from uroseg.commands.train_nnunet import main
    from uroseg.models import ModelDef

    # Scaffold the raw data directory
    raw_dir = tmp_path / "nnUNet" / "raw" / "Dataset001_Prostate"
    images_tr = raw_dir / "imagesTr"
    images_tr.mkdir(parents=True)
    for i in range(2):
        data = np.zeros((10, 10, 10), dtype=np.int16)
        nib.save(nib.Nifti1Image(data, np.eye(4)), images_tr / f"case_{i:03d}_0000.nii.gz")

    # Also create preprocessed dir so plan_and_preprocess is skipped
    preprocessed_dir = tmp_path / "nnUNet" / "preprocessed" / "Dataset001_Prostate"
    preprocessed_dir.mkdir(parents=True)

    # Create a mock module
    mock_module = MagicMock()
    mock_module.NNUNET_TASK = "Dataset001_Prostate"
    mock_module.MODEL = ModelDef(
        name="prostate",
        description="Prostate MRI-T2",
        weights_url="https://example.com/prostate.zip",
        labels={"background": 0, "prostate": [1], "prostate_sub": 1},
    )

    with patch("auglab.add_trainer.add_trainer"), \
         patch("uroseg.commands.train_nnunet.subprocess.run") as mock_run, \
         patch("uroseg.utils.utils.load_model_module") as mock_load_module:

        mock_load_module.return_value = mock_module
        mock_run.return_value = None

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "prostate", "--fold", "0",
            "--data-dir", str(tmp_path),
        ]):
            main()

    dataset_json_path = raw_dir / "dataset.json"
    assert dataset_json_path.exists()
    with open(dataset_json_path) as f:
        data = json.load(f)
    assert data["numTraining"] == 2
    assert data["regions_class_order"] == [1]  # prostate: [1] triggers regions
    assert data["file_ending"] == ".nii.gz"


def test_train_calls_nnunet_train(tmp_path):
    from uroseg.commands.train_nnunet import main
    from uroseg.models import ModelDef

    raw_dir = tmp_path / "nnUNet" / "raw" / "Dataset010_Bladder"
    images_tr = raw_dir / "imagesTr"
    images_tr.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4)),
        images_tr / "case_000_0000.nii.gz",
    )
    preprocessed_dir = tmp_path / "nnUNet" / "preprocessed" / "Dataset010_Bladder"
    preprocessed_dir.mkdir(parents=True)

    # Create a mock module
    mock_module = MagicMock()
    mock_module.NNUNET_TASK = "Dataset010_Bladder"
    mock_module.MODEL = ModelDef(
        name="bladder",
        description="Urinary bladder (CT)",
        weights_url="https://example.com/bladder.zip",
        labels={"background": 0, "bladder": 1},
    )

    with patch("auglab.add_trainer.add_trainer"), \
         patch("uroseg.commands.train_nnunet.subprocess.run") as mock_run, \
         patch("uroseg.utils.utils.load_model_module") as mock_load_module:

        mock_load_module.return_value = mock_module
        mock_run.return_value = None

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "bladder", "--fold", "0",
            "--data-dir", str(tmp_path),
        ]):
            main()

    # nnUNetv2_train must have been called
    called_cmds = [call_args[0][0] for call_args in mock_run.call_args_list]
    train_calls = [cmd for cmd in called_cmds if cmd[0] == "nnUNetv2_train"]
    assert len(train_calls) == 1
    train_cmd = train_calls[0]
    assert "10" in train_cmd  # dataset_id
    assert "3d_fullres" in train_cmd
    assert "0" in train_cmd   # fold
    assert "nnUNetTrainerDAExt" in train_cmd


def test_train_skips_preprocess_if_done(tmp_path):
    from uroseg.commands.train_nnunet import main
    from uroseg.models import ModelDef

    raw_dir = tmp_path / "nnUNet" / "raw" / "Dataset010_Bladder"
    images_tr = raw_dir / "imagesTr"
    images_tr.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4)),
        images_tr / "case_000_0000.nii.gz",
    )
    # Already preprocessed
    (tmp_path / "nnUNet" / "preprocessed" / "Dataset010_Bladder").mkdir(parents=True)

    # Create a mock module
    mock_module = MagicMock()
    mock_module.NNUNET_TASK = "Dataset010_Bladder"
    mock_module.MODEL = ModelDef(
        name="bladder",
        description="Urinary bladder (CT)",
        weights_url="https://example.com/bladder.zip",
        labels={"background": 0, "bladder": 1},
    )

    with patch("auglab.add_trainer.add_trainer"), \
         patch("uroseg.commands.train_nnunet.subprocess.run") as mock_run, \
         patch("uroseg.utils.utils.load_model_module") as mock_load_module:

        mock_load_module.return_value = mock_module
        mock_run.return_value = None

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "bladder", "--fold", "0",
            "--data-dir", str(tmp_path),
        ]):
            main()

    called_cmds = [c[0][0] for c in mock_run.call_args_list]
    preprocess_calls = [c for c in called_cmds if c[0] == "nnUNetv2_plan_and_preprocess"]
    assert len(preprocess_calls) == 0


def test_train_sets_auglab_config_env(tmp_path):
    from uroseg.commands.train_nnunet import main
    from uroseg.models import ModelDef

    raw_dir = tmp_path / "nnUNet" / "raw" / "Dataset010_Bladder"
    images_tr = raw_dir / "imagesTr"
    images_tr.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4)),
        images_tr / "case_000_0000.nii.gz",
    )
    (tmp_path / "nnUNet" / "preprocessed" / "Dataset010_Bladder").mkdir(parents=True)

    auglab_cfg = tmp_path / "auglab.json"
    auglab_cfg.write_text('{"augmentations": []}')

    captured_env = {}

    def capture_run(cmd, **kwargs):
        if cmd[0] == "nnUNetv2_train":
            captured_env["AUGLAB_CONFIG"] = os.environ.get("AUGLAB_CONFIG")

    # Create a mock module
    mock_module = MagicMock()
    mock_module.NNUNET_TASK = "Dataset010_Bladder"
    mock_module.MODEL = ModelDef(
        name="bladder",
        description="Urinary bladder (CT)",
        weights_url="https://example.com/bladder.zip",
        labels={"background": 0, "bladder": 1},
    )

    with patch("auglab.add_trainer.add_trainer"), \
         patch("uroseg.commands.train_nnunet.subprocess.run", side_effect=capture_run), \
         patch("uroseg.utils.utils.load_model_module") as mock_load_module:

        mock_load_module.return_value = mock_module

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "bladder", "--fold", "0",
            "--data-dir", str(tmp_path),
            "--auglab-config", str(auglab_cfg),
        ]):
            main()

    assert captured_env.get("AUGLAB_CONFIG") == str(auglab_cfg)


def test_train_fails_when_no_images_tr(tmp_path):
    from uroseg.commands.train_nnunet import main
    from uroseg.models import ModelDef

    # Create a mock module
    mock_module = MagicMock()
    mock_module.NNUNET_TASK = "Dataset010_Bladder"
    mock_module.MODEL = ModelDef(
        name="bladder",
        description="Urinary bladder (CT)",
        weights_url="https://example.com/bladder.zip",
        labels={"background": 0, "bladder": 1},
    )

    with patch("uroseg.utils.utils.load_model_module") as mock_load_module:
        mock_load_module.return_value = mock_module

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "bladder", "--fold", "0",
            "--data-dir", str(tmp_path),
        ]):
            with pytest.raises((FileNotFoundError, SystemExit)):
                main()


def test_train_cli_help():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'train', 'nnunet', '--help'],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert 'organ' in result.stdout
    assert '--fold' in result.stdout
    assert '--auglab-config' in result.stdout
