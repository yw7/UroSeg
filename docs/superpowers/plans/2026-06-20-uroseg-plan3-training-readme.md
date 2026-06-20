# UroSeg Plan 3 — Training + README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `uroseg list`, `uroseg train` (nnU-Net + AugLab), and the project README.

**Architecture:** `uroseg train` is a thin Python wrapper that (1) reads the model JSON, (2) resolves the data path, (3) auto-generates `dataset.json` for nnU-Net, (4) shells out to `nnUNetv2_plan_and_preprocess` + `nnUNetv2_train`. The AugLab trainer is selected via nnU-Net's `--trainer` flag, not monkey-patching. The README is a single `README.md` at the repo root.

**Tech Stack:** Python 3.10+, nnunetv2, auglab, subprocess, pytest (mocking subprocess), Markdown

## Global Constraints

- Python ≥ 3.10
- Data path resolution order: `--data-dir` → `UROSEG_DATA` env var → `~/uroseg/`
- nnU-Net env vars (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) are set by `uroseg train`, never by the user
- AugLab trainer: `nnUNetTrainerDAExt` (via `--trainer` flag, not patching)
- AugLab config: set via `AUGLAB_CONFIG` env var when `--auglab-config` is provided
- light-the-torch: README-only — NOT imported or called in any Python source
- Model JSON fields: `name`, `description`, `nnunet_task`, `channel_names`, `labels`, optional `regions_class_order`, optional `weights_url`
- Generated `dataset.json` must include `file_ending: ".nii.gz"`

## Prerequisites

Plans 1 and 2 must be complete. Verify before starting:

```bash
python -c "
from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs, collect_niftis, build_output_path, get_model, get_all_models, resolve_data_path
from uroseg.commands.install import main as install_main
print('Plan 1 OK')
"
pytest tests/ -v --tb=short -q
```

Expected: all tests pass, `Plan 1 OK` printed.

---

## File Map

| File | Role |
|------|------|
| `tests/conftest.py` | Add `nifti_folder` fixture (used by Plan 2 cpdir tests) |
| `uroseg/commands/list_models.py` | `uroseg list` — print table of all registered organ models |
| `uroseg/commands/train.py` | `uroseg train` — generate dataset.json + shell out to nnU-Net |
| `tests/test_train.py` | Unit tests for train logic (mocking subprocess) |
| `README.md` | Project README — installation, usage, training, contributing |

---

### Task 1: Add `nifti_folder` fixture to conftest.py

**Files:**
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces: `nifti_folder` pytest fixture — a `tmp_path`-backed folder containing 3 `.nii.gz` files, available to all tests in the `tests/` directory.

This fixture is needed by Plan 2's `test_cpdir_copies_files` and `test_cpdir_skips_existing_without_overwrite` tests. If those tests already pass, this task verifies the fixture exists — no harm running it.

- [ ] **Step 1: Read current conftest.py**

```bash
cat tests/conftest.py
```

Check whether a `nifti_folder` fixture already exists. If it does, skip Step 2 and go straight to Step 3.

- [ ] **Step 2: Add nifti_folder fixture**

Append to `tests/conftest.py` (after the last existing fixture):

```python
@pytest.fixture
def nifti_folder(tmp_path):
    """A temp folder with 3 synthetic NIfTI files for batch-command tests."""
    import numpy as np
    import nibabel as nib
    folder = tmp_path / "niftis"
    folder.mkdir()
    for i in range(3):
        data = np.zeros((10, 10, 10), dtype=np.int16)
        data[2:8, 2:8, 2:8] = i + 1
        nib.save(nib.Nifti1Image(data, np.eye(4)), folder / f"case_{i:03d}.nii.gz")
    return folder
```

- [ ] **Step 3: Run cpdir tests to confirm they pass**

```bash
pytest tests/test_commands.py::test_cpdir_copies_files tests/test_commands.py::test_cpdir_skips_existing_without_overwrite -v
```

Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add nifti_folder fixture to conftest"
```

---

### Task 2: uroseg list

**Files:**
- Create: `uroseg/commands/list_models.py`
- Modify: `uroseg/cli.py` (register `list` subcommand)
- Modify: `tests/test_cli.py` (add list tests)

**Interfaces:**
- Consumes: `get_all_models() -> dict[str, dict]` from `uroseg.utils.utils`
- Produces: `uroseg list` prints a formatted table of all organ models with name, nnunet_task, and description

- [ ] **Step 1: Write failing tests**

In `tests/test_cli.py`, add:

```python
def test_list_prints_model_names(capsys):
    from uroseg.commands.list_models import main
    main()
    out = capsys.readouterr().out
    assert 'prostate' in out
    assert 'bladder' in out


def test_list_cli(tmp_path):
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'list'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert 'prostate' in result.stdout
    assert 'bladder' in result.stdout
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_cli.py::test_list_prints_model_names tests/test_cli.py::test_list_cli -v
```

Expected: `ImportError` or `SystemExit`.

- [ ] **Step 3: Implement uroseg/commands/list_models.py**

```python
from __future__ import annotations
from uroseg.utils.utils import get_all_models


def main() -> None:
    models = get_all_models()
    if not models:
        print("No models found in resources/models/")
        return
    name_w, task_w = 20, 30
    header = f"{'Model':<{name_w}} {'Task':<{task_w}} Description"
    print(header)
    print('-' * (name_w + task_w + 40))
    for name, model in sorted(models.items()):
        task = model.get('nnunet_task', 'N/A')
        desc = model.get('description', '')
        print(f"{name:<{name_w}} {task:<{task_w}} {desc}")
```

- [ ] **Step 4: Register in cli.py**

In `uroseg/cli.py`, in the `SUBCOMMANDS` dict (or equivalent dispatcher), add:

```python
'list': 'uroseg.commands.list_models',
```

If the CLI uses a subparser approach, add the `list` subparser:

```python
sub = subparsers.add_parser('list', help='List available organ models')
sub.set_defaults(func=lambda _args: __import__('uroseg.commands.list_models', fromlist=['main']).main())
```

Read `uroseg/cli.py` first to see the exact pattern used, then match it exactly.

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_cli.py::test_list_prints_model_names tests/test_cli.py::test_list_cli -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add uroseg/commands/list_models.py uroseg/cli.py tests/test_cli.py
git commit -m "feat: uroseg list — print table of available organ models"
```

---

### Task 3: uroseg train

**Files:**
- Create: `uroseg/commands/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes:
  - `get_model(name: str) -> dict` from `uroseg.utils.utils`
  - `resolve_data_path(data_dir: str | None) -> Path` from `uroseg.utils.utils`
- Produces:
  - `extract_dataset_id(nnunet_task: str) -> int` — parses `"Dataset001_Prostate"` → `1`
  - `generate_dataset_json(model: dict, images_tr_dir: Path) -> dict` — builds nnU-Net `dataset.json` dict from model JSON
  - `setup_nnunet_env(data_path: Path) -> None` — sets `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` env vars
  - `main()` — `uroseg train --organ ... --fold ...`

- [ ] **Step 1: Write failing tests in tests/test_train.py**

```python
from __future__ import annotations
import json
import os
from pathlib import Path
from unittest.mock import patch, call
import numpy as np
import nibabel as nib
import pytest


@pytest.fixture
def prostate_model():
    return {
        "name": "prostate",
        "description": "Prostate MRI-T2",
        "nnunet_task": "Dataset001_Prostate",
        "channel_names": {"0": "MRI-T2"},
        "labels": {"0": "background", "1": "prostate", "2": "prostate_pz",
                   "3": "prostate_cz", "4": "prostate_afs"},
        "regions_class_order": [1, 2, 3, 4],
    }


@pytest.fixture
def bladder_model():
    return {
        "name": "bladder",
        "description": "Urinary bladder (CT)",
        "nnunet_task": "Dataset010_Bladder",
        "channel_names": {"0": "CT"},
        "labels": {"0": "background", "1": "bladder"},
    }


@pytest.fixture
def images_tr_dir(tmp_path):
    d = tmp_path / "imagesTr"
    d.mkdir()
    for i in range(3):
        data = np.zeros((10, 10, 10), dtype=np.int16)
        nib.save(nib.Nifti1Image(data, np.eye(4)), d / f"case_{i:03d}_0000.nii.gz")
    return d


# ── extract_dataset_id ────────────────────────────────────────────────────────

def test_extract_dataset_id_parses_correctly():
    from uroseg.commands.train import extract_dataset_id
    assert extract_dataset_id("Dataset001_Prostate") == 1
    assert extract_dataset_id("Dataset010_Bladder") == 10
    assert extract_dataset_id("Dataset123_Kidney") == 123


def test_extract_dataset_id_raises_on_bad_format():
    from uroseg.commands.train import extract_dataset_id
    with pytest.raises(ValueError, match="Invalid nnunet_task"):
        extract_dataset_id("ProstateBadName")


# ── generate_dataset_json ─────────────────────────────────────────────────────

def test_generate_dataset_json_simple(bladder_model, images_tr_dir):
    from uroseg.commands.train import generate_dataset_json
    result = generate_dataset_json(bladder_model, images_tr_dir)
    assert result["channel_names"] == {"0": "CT"}
    assert result["labels"] == {"0": "background", "1": "bladder"}
    assert result["numTraining"] == 3
    assert result["file_ending"] == ".nii.gz"
    assert "regions_class_order" not in result


def test_generate_dataset_json_with_regions(prostate_model, images_tr_dir):
    from uroseg.commands.train import generate_dataset_json
    result = generate_dataset_json(prostate_model, images_tr_dir)
    assert result["regions_class_order"] == [1, 2, 3, 4]
    assert result["channel_names"] == {"0": "MRI-T2"}


# ── setup_nnunet_env ──────────────────────────────────────────────────────────

def test_setup_nnunet_env_sets_vars(tmp_path):
    from uroseg.commands.train import setup_nnunet_env
    setup_nnunet_env(tmp_path)
    assert os.environ["nnUNet_raw"] == str(tmp_path / "nnUNet" / "raw")
    assert os.environ["nnUNet_preprocessed"] == str(tmp_path / "nnUNet" / "preprocessed")
    assert os.environ["nnUNet_results"] == str(tmp_path / "nnUNet" / "results")


# ── main() integration — subprocess mocked ───────────────────────────────────

def test_train_generates_dataset_json(tmp_path):
    from uroseg.commands.train import main

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

    with patch("uroseg.commands.train.subprocess.run") as mock_run, \
         patch("uroseg.utils.utils.get_model") as mock_get_model:

        mock_get_model.return_value = {
            "name": "prostate",
            "nnunet_task": "Dataset001_Prostate",
            "channel_names": {"0": "MRI-T2"},
            "labels": {"0": "background", "1": "prostate"},
            "regions_class_order": [1],
        }
        mock_run.return_value = None

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "train", "--organ", "prostate", "--fold", "0",
            "--data-dir", str(tmp_path),
        ]):
            main()

    dataset_json_path = raw_dir / "dataset.json"
    assert dataset_json_path.exists()
    with open(dataset_json_path) as f:
        data = json.load(f)
    assert data["numTraining"] == 2
    assert data["regions_class_order"] == [1]
    assert data["file_ending"] == ".nii.gz"


def test_train_calls_nnunet_train(tmp_path):
    from uroseg.commands.train import main

    raw_dir = tmp_path / "nnUNet" / "raw" / "Dataset010_Bladder"
    images_tr = raw_dir / "imagesTr"
    images_tr.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4)),
        images_tr / "case_000_0000.nii.gz",
    )
    preprocessed_dir = tmp_path / "nnUNet" / "preprocessed" / "Dataset010_Bladder"
    preprocessed_dir.mkdir(parents=True)

    with patch("uroseg.commands.train.subprocess.run") as mock_run, \
         patch("uroseg.utils.utils.get_model") as mock_get_model:

        mock_get_model.return_value = {
            "name": "bladder",
            "nnunet_task": "Dataset010_Bladder",
            "channel_names": {"0": "CT"},
            "labels": {"0": "background", "1": "bladder"},
        }
        mock_run.return_value = None

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "train", "--organ", "bladder", "--fold", "0",
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
    from uroseg.commands.train import main

    raw_dir = tmp_path / "nnUNet" / "raw" / "Dataset010_Bladder"
    images_tr = raw_dir / "imagesTr"
    images_tr.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4)),
        images_tr / "case_000_0000.nii.gz",
    )
    # Already preprocessed
    (tmp_path / "nnUNet" / "preprocessed" / "Dataset010_Bladder").mkdir(parents=True)

    with patch("uroseg.commands.train.subprocess.run") as mock_run, \
         patch("uroseg.utils.utils.get_model") as mock_get_model:

        mock_get_model.return_value = {
            "name": "bladder",
            "nnunet_task": "Dataset010_Bladder",
            "channel_names": {"0": "CT"},
            "labels": {"0": "background", "1": "bladder"},
        }
        mock_run.return_value = None

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "train", "--organ", "bladder", "--fold", "0",
            "--data-dir", str(tmp_path),
        ]):
            main()

    called_cmds = [c[0][0] for c in mock_run.call_args_list]
    preprocess_calls = [c for c in called_cmds if c[0] == "nnUNetv2_plan_and_preprocess"]
    assert len(preprocess_calls) == 0


def test_train_sets_auglab_config_env(tmp_path):
    from uroseg.commands.train import main

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

    with patch("uroseg.commands.train.subprocess.run", side_effect=capture_run), \
         patch("uroseg.utils.utils.get_model") as mock_get_model:

        mock_get_model.return_value = {
            "name": "bladder",
            "nnunet_task": "Dataset010_Bladder",
            "channel_names": {"0": "CT"},
            "labels": {"0": "background", "1": "bladder"},
        }

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "train", "--organ", "bladder", "--fold", "0",
            "--data-dir", str(tmp_path),
            "--auglab-config", str(auglab_cfg),
        ]):
            main()

    assert captured_env.get("AUGLAB_CONFIG") == str(auglab_cfg)


def test_train_fails_when_no_images_tr(tmp_path):
    from uroseg.commands.train import main

    with patch("uroseg.utils.utils.get_model") as mock_get_model:
        mock_get_model.return_value = {
            "name": "bladder",
            "nnunet_task": "Dataset010_Bladder",
            "channel_names": {"0": "CT"},
            "labels": {"0": "background", "1": "bladder"},
        }

        import sys
        with patch.object(sys, "argv", [
            "uroseg", "train", "--organ", "bladder", "--fold", "0",
            "--data-dir", str(tmp_path),
        ]):
            with pytest.raises((FileNotFoundError, SystemExit)):
                main()


def test_train_cli_help():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'train', '--help'],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert '--organ' in result.stdout
    assert '--fold' in result.stdout
    assert '--auglab-config' in result.stdout
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_train.py -v
```

Expected: `ImportError` — `uroseg.commands.train` not found.

- [ ] **Step 3: Implement uroseg/commands/train.py**

```python
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from uroseg.utils.utils import get_model, resolve_data_path


def extract_dataset_id(nnunet_task: str) -> int:
    match = re.match(r'Dataset(\d+)_', nnunet_task)
    if not match:
        raise ValueError(f"Invalid nnunet_task format: {nnunet_task!r} — expected 'DatasetNNN_Name'")
    return int(match.group(1))


def generate_dataset_json(model: dict, images_tr_dir: Path) -> dict:
    dataset: dict = {
        "channel_names": model["channel_names"],
        "labels": model["labels"],
        "numTraining": len(list(images_tr_dir.glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    if "regions_class_order" in model:
        dataset["regions_class_order"] = model["regions_class_order"]
    return dataset


def setup_nnunet_env(data_path: Path) -> None:
    nnunet_dir = data_path / "nnUNet"
    os.environ["nnUNet_raw"] = str(nnunet_dir / "raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_dir / "preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_dir / "results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an nnU-Net model for a UroSeg organ using AugLab augmentation."
    )
    parser.add_argument("--organ", required=True,
                        help="Organ name matching resources/models/<organ>.json")
    parser.add_argument("--fold", type=int, required=True,
                        help="nnU-Net fold number (0–4)")
    parser.add_argument("--auglab-config", default=None,
                        help="Path to AugLab augmentation config JSON (optional)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--data-dir", default=None,
                        help="Override UROSEG_DATA / ~/uroseg/ with this path")
    args = parser.parse_args()

    model = get_model(args.organ)
    data_path = resolve_data_path(args.data_dir)
    setup_nnunet_env(data_path)

    nnunet_task = model["nnunet_task"]
    dataset_id = extract_dataset_id(nnunet_task)

    raw_dir = data_path / "nnUNet" / "raw" / nnunet_task
    images_tr = raw_dir / "imagesTr"

    if not images_tr.exists():
        print(
            f"Error: training images directory not found: {images_tr}\n"
            f"Place training images in {images_tr}/ (filename pattern: <case>_0000.nii.gz)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-generate dataset.json
    dataset_json = generate_dataset_json(model, images_tr)
    dataset_json_path = raw_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
    print(f"Generated {dataset_json_path}")

    # Plan and preprocess if not already done
    preprocessed_dir = data_path / "nnUNet" / "preprocessed" / nnunet_task
    if not preprocessed_dir.exists():
        print("Running nnU-Net planning and preprocessing...")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"],
            check=True,
        )

    # Set AugLab config env var if provided
    if args.auglab_config:
        os.environ["AUGLAB_CONFIG"] = str(args.auglab_config)

    # Run training
    results_dir = data_path / "nnUNet" / "results"
    print(f"Starting training (dataset {dataset_id}, fold {args.fold})...")
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            str(args.fold),
            "--trainer", "nnUNetTrainerDAExt",
        ],
        check=True,
    )

    trainer_dir = (
        results_dir
        / nnunet_task
        / "nnUNetTrainerDAExt__nnUNetPlans__3d_fullres"
        / f"fold_{args.fold}"
    )
    print(f"\nTraining complete. Model saved to:\n  {trainer_dir}")
```

- [ ] **Step 4: Register train in cli.py**

Read `uroseg/cli.py` and add `'train': 'uroseg.commands.train'` to the subcommand dispatcher, following the same pattern as `install` and other commands.

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_train.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add uroseg/commands/train.py uroseg/cli.py tests/test_train.py
git commit -m "feat: uroseg train — auto-generate dataset.json and run nnU-Net with AugLab trainer"
```

---

### Task 4: README.md

**Files:**
- Create: `README.md`

**Interfaces:**
- Produces: project README for GitHub and PyPI; no tests (visual review only)

- [ ] **Step 1: Create README.md**

```bash
cat > README.md << 'EOF'
# UroSeg

Automated segmentation of urological anatomy from medical images. Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) with [AugLab](https://github.com/neuropoly/AugLab) online augmentation.

---

## Available Models

| Command | Anatomy | Modality | Labels |
|---------|---------|----------|--------|
| `uroseg prostate` | Prostate | MRI T2 | whole prostate, peripheral zone, central zone, anterior fibromuscular stroma |
| `uroseg bladder` | Bladder | CT | bladder |

---

## Installation

### 1. Install PyTorch

Use [light-the-torch](https://github.com/pmeier/light-the-torch) to automatically detect your CUDA version and install the correct PyTorch build:

```bash
pip install light-the-torch
ltt install torch
```

### 2. Install UroSeg

```bash
pip install uroseg
```

Development install:

```bash
git clone https://github.com/neuropoly/uroseg
cd uroseg
pip install -e .
```

### 3. Download model weights

```bash
uroseg install --model prostate
uroseg install --model bladder
uroseg install --all          # download all available models
```

By default, weights are stored in `~/uroseg/nnUNet/results/`. Override with `--data-dir PATH` or by setting `UROSEG_DATA=/path/to/data`.

---

## Usage

### Inference

```bash
# Single image
uroseg prostate --img subject01_T2.nii.gz --out subject01_prostate.nii.gz
uroseg bladder  --img subject01_CT.nii.gz  --out subject01_bladder.nii.gz

# Batch (folder input → folder output)
uroseg prostate --img /data/mri/ --out /data/segs/ --max-workers 4
```

### Utilities

```bash
# Remap label IDs using a JSON map {"src_id": dst_id}
uroseg map --seg seg.nii.gz --out remapped.nii.gz --map labels.json

# Resample to 1×1×1 mm isotropic
uroseg resample --img img.nii.gz --out img_1mm.nii.gz --spacing 1 1 1

# Reorient to RAS canonical
uroseg reorient --img img.nii.gz --out img_ras.nii.gz

# Keep only the largest connected component per label
uroseg largest_component --seg seg.nii.gz --out seg_lc.nii.gz

# Crop image and seg to segmentation bounding box
uroseg crop --img img.nii.gz --seg seg.nii.gz \
            --out-img img_crop.nii.gz --out-seg seg_crop.nii.gz

# Generate JPG preview (3 orthogonal slices, optional seg overlay)
uroseg preview --img img.nii.gz --seg seg.nii.gz --out preview.jpg

# Resample segmentation to match reference image space (nearest-neighbour)
uroseg transform_seg2image --seg seg.nii.gz --img ref.nii.gz --out-seg seg_transformed.nii.gz

# Copy NIfTI files with optional renaming
uroseg cpdir --img /data/mri/ --out /data/mri_copy/ --out-suffix _copy

# List available organ models
uroseg list
```

### CLI reference

All commands support `--overwrite`, `--max-workers N`, and `--quiet`.

```
uroseg <organ>              --img PATH --out PATH [--fold N] [--out-suffix SUFFIX]
uroseg map                  --seg PATH --out PATH --map JSON [--out-suffix SUFFIX]
uroseg resample             --img PATH --out PATH --spacing X Y Z [--out-suffix SUFFIX]
uroseg reorient             --img PATH --out PATH [--orientation RAS] [--out-suffix SUFFIX]
uroseg largest_component    --seg PATH --out PATH [--labels 1 2 3] [--out-suffix SUFFIX]
uroseg preview              --img PATH [--seg PATH] --out PATH [--out-suffix SUFFIX]
uroseg crop                 --img PATH --seg PATH --out-img PATH --out-seg PATH
uroseg transform_seg2image  --seg PATH --img PATH --out-seg PATH [--seg-suffix SUFFIX]
uroseg cpdir                --img PATH --out PATH [--out-suffix SUFFIX] [--out-prefix PREFIX]
uroseg install              --model NAME [NAME ...] | --all [--data-dir PATH]
uroseg train                --organ NAME --fold N [--auglab-config JSON] [--gpus N] [--data-dir PATH]
uroseg list
```

---

## Training

### 1. Create the model JSON

Add `uroseg/resources/models/<organ>.json`. Minimal example (CT, single label):

```json
{
  "name": "kidney",
  "description": "Kidney (CT)",
  "nnunet_task": "Dataset020_Kidney",
  "channel_names": {"0": "CT"},
  "labels": {
    "0": "background",
    "1": "kidney"
  }
}
```

Region-based model with hierarchical sub-labels (single model, sigmoid per region):

```json
{
  "name": "prostate",
  "description": "Prostate MRI-T2: whole prostate (1), peripheral zone (2), central zone (3), anterior fibromuscular stroma (4)",
  "nnunet_task": "Dataset001_Prostate",
  "channel_names": {"0": "MRI-T2"},
  "labels": {
    "0": "background",
    "1": "prostate",
    "2": "prostate_pz",
    "3": "prostate_cz",
    "4": "prostate_afs"
  },
  "regions_class_order": [1, 2, 3, 4]
}
```

`regions_class_order` enables nnU-Net's region-based training: label 1 in ground truth is the union of labels 2, 3, and 4 (the prostate zones together make up the whole prostate). `channel_names` maps channel index → modality name; use `"CT"` for CT-specific normalization; any other string uses per-case z-score normalization.

### 2. Place training data

```
~/uroseg/nnUNet/raw/Dataset020_Kidney/
├── dataset.json          ← auto-generated by uroseg train
├── imagesTr/
│   ├── kidney_001_0000.nii.gz   ← image channel 0
│   └── ...
└── labelsTr/
    ├── kidney_001.nii.gz        ← ground truth segmentation
    └── ...
```

nnU-Net filename convention: images end in `_0000.nii.gz` (channel 0), labels have no channel suffix.

### 3. Run training

```bash
# Basic
uroseg train --organ kidney --fold 0

# With AugLab augmentation config
uroseg train --organ kidney --fold 0 --auglab-config auglab.json

# Custom data directory
uroseg train --organ kidney --fold 0 --data-dir /scratch/uroseg_data
```

`uroseg train` automatically:
- Generates `dataset.json` from the model JSON
- Sets `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` environment variables
- Runs `nnUNetv2_plan_and_preprocess` (skipped if already done)
- Runs `nnUNetv2_train` with `nnUNetTrainerDAExt` (AugLab's trainer subclass)

Trained model location:

```
~/uroseg/nnUNet/results/<nnunet_task>/nnUNetTrainerDAExt__nnUNetPlans__3d_fullres/fold_<N>/
├── checkpoint_best.pth
└── checkpoint_final.pth
```

---

## Data Storage

All UroSeg data lives under a single configurable root:

| Priority | Method | Value |
|----------|--------|-------|
| 1 (highest) | CLI flag | `--data-dir /path/to/data` |
| 2 | Environment variable | `export UROSEG_DATA=/path/to/data` |
| 3 (default) | Default | `~/uroseg/` |

```
~/uroseg/
└── nnUNet/
    ├── raw/                        # user training images (imagesTr/, labelsTr/)
    ├── preprocessed/               # nnU-Net preprocessing cache (auto-created)
    ├── results/
    │   ├── r20260101/              # versioned by release date tag
    │   │   └── Dataset001_Prostate/
    │   └── r20261001/              # newer release coexists
    │       └── Dataset001_Prostate/
    └── exports/                    # downloaded zip archives (removed after extraction)
```

---

## Contributing — Adding a New Organ Model

1. Create `uroseg/resources/models/<organ>.json` with the fields above
2. Place training data in `~/uroseg/nnUNet/raw/<nnunet_task>/imagesTr/` and `labelsTr/`
3. Train: `uroseg train --organ <organ> --fold 0`
4. Archive the trained model: `Dataset###_<Name>_r<YYYYMMDD>.zip`
5. Upload as a GitHub Release asset and set `weights_url` in the model JSON
6. Open a pull request
EOF
```

- [ ] **Step 2: Verify README renders correctly**

```bash
# Check no broken lines, placeholder text, or shell syntax errors
grep -n "TBD\|TODO\|FIXME\|<placeholder>" README.md
```

Expected: no output (no placeholders).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with installation, usage, training, and contributing guide"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `uroseg list` → Task 2
- [x] `uroseg train --organ --fold --auglab-config --gpus --data-dir` → Task 3
- [x] Data path resolution: `--data-dir` → `UROSEG_DATA` → `~/uroseg/` → Task 3 (`resolve_data_path`)
- [x] Auto-set `nnUNet_raw/preprocessed/results` env vars → Task 3 (`setup_nnunet_env`)
- [x] Auto-generate `dataset.json` from model JSON → Task 3 (`generate_dataset_json`)
- [x] `regions_class_order` included in `dataset.json` when present → Task 3 (test: `test_generate_dataset_json_with_regions`)
- [x] Skip `plan_and_preprocess` if already done → Task 3 (test: `test_train_skips_preprocess_if_done`)
- [x] AugLab via `--trainer nnUNetTrainerDAExt` → Task 3
- [x] `AUGLAB_CONFIG` env var when `--auglab-config` provided → Task 3 (test: `test_train_sets_auglab_config_env`)
- [x] Print results path on completion → Task 3
- [x] light-the-torch in README only, not in code → Task 4 (README only)
- [x] README: Available Models table → Task 4
- [x] README: Installation steps → Task 4
- [x] README: Inference usage → Task 4
- [x] README: All utility commands → Task 4
- [x] README: Full CLI reference → Task 4
- [x] README: Training workflow → Task 4
- [x] README: Data storage layout → Task 4
- [x] README: Contributing guide → Task 4
- [x] `nifti_folder` fixture for cpdir tests → Task 1

**Placeholder scan:** None found.

**Type consistency:**
- `extract_dataset_id(nnunet_task: str) -> int` ✓
- `generate_dataset_json(model: dict, images_tr_dir: Path) -> dict` ✓
- `setup_nnunet_env(data_path: Path) -> None` ✓
- `resolve_data_path` consumed from `uroseg.utils.utils` (defined in Plan 1) ✓
- `get_model(name: str) -> dict` consumed from `uroseg.utils.utils` (defined in Plan 1) ✓

---

*All three plans complete. Plans 1, 2, and 3 together constitute the full UroSeg implementation.*
