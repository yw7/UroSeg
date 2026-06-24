# UroSeg CLI UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add canonical `-h` help output, replace `uroseg list` with a rich model+command listing, and introduce a generic `uroseg train <engine>` dispatcher backed by `train_nnunet.py`.

**Architecture:** `list_models.py` owns the `show_help()` function; `cli.py` routes no-args/`-h`/`--help`/`list` to it. A new generic `train.py` dispatcher sub-dispatches to `train_nnunet.py` (current `train.py` content, renamed). No other command files are touched.

**Tech Stack:** Python 3.10+, stdlib only (argparse, sys).

## Global Constraints

- `uroseg`, `uroseg -h`, `uroseg --help`, `uroseg list` all exit 0 and print identical help output
- `uroseg train` (no engine) exits 1; `uroseg train -h` exits 0
- Inference invocation `uroseg prostate -i … -o …` is unchanged
- No new dependencies; no argparse subparsers refactor; no changes to any command file except `list_models.py`, `train.py`, `train_nnunet.py`
- Error message for unknown model: `"Unknown model or subcommand: 'X'\nRun 'uroseg --help' to see available models and commands."`

---

### Task 1: `show_help()` in `list_models.py` + `cli.py` wiring

**Files:**
- Modify: `uroseg/commands/list_models.py`
- Modify: `uroseg/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `show_help() -> None` in `uroseg.commands.list_models` — called by `cli.py` for no-args, `-h`, `--help`, `list`

---

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (keep the existing `run_uroseg` helper and all existing tests; only modify `test_no_args_exits_nonzero` and add new ones):

```python
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


def test_unknown_model_exits_nonzero():
    result = run_uroseg('nonexistent_model', '--img', 'x', '--out', 'y')
    assert result.returncode != 0
    assert 'Unknown model' in result.stderr
```

Delete (or rename) `test_no_args_exits_nonzero` — its assertion `returncode != 0` is now wrong.

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd /workspaces/UroSeg && python -m pytest tests/test_cli.py -v -k "help or no_args or list_redirects or unknown_model" 2>&1 | tail -20
```

Expected: FAIL — `test_no_args_shows_help_exits_zero` fails because no-args currently exits 1; `test_help_flag_exits_zero` fails because `-h` is treated as unknown option.

- [ ] **Step 3: Rewrite `list_models.py`**

Replace the entire file with:

```python
from __future__ import annotations
from uroseg.utils.utils import get_all_models

_COMMANDS = {
    'map':                 'Remap label IDs',
    'resample':            'Resample image to target voxel size',
    'reorient':            'Reorient image to RAS canonical',
    'largest_component':   'Keep largest connected component per label',
    'crop':                'Crop image to segmentation bounding box',
    'preview':             'Generate JPG slice preview',
    'transform_seg2image': 'Resample segmentation to reference image space',
    'cpdir':               'Copy NIfTI files with optional renaming',
    'install':             'Download model weights',
    'train nnunet':        'Train with nnU-Net + AugLab',
}


def show_help() -> None:
    models = get_all_models()
    lines = ['uroseg — urological anatomy segmentation', '']
    lines.append('Models:')
    if models:
        name_w = max(len(n) for n in models) + 2
        for name, info in sorted(models.items()):
            lines.append(f'  {name:<{name_w}}{info.get("description", "")}')
    else:
        lines.append('  (no models installed)')
    lines.append('')
    lines.append('Commands:')
    cmd_w = max(len(c) for c in _COMMANDS) + 2
    for cmd, desc in _COMMANDS.items():
        lines.append(f'  {cmd:<{cmd_w}}{desc}')
    lines.append('')
    lines.append("Run 'uroseg <model|command> --help' for per-command usage.")
    print('\n'.join(lines))


def main() -> None:
    show_help()
```

- [ ] **Step 4: Rewrite `cli.py`**

Replace the entire file with:

```python
from __future__ import annotations
import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        from uroseg.commands.list_models import show_help
        show_help()
        return

    cmd = sys.argv[1]

    if cmd == 'list':
        from uroseg.commands.list_models import show_help
        show_help()
    elif cmd == 'install':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.install import main as run
        run()
    elif cmd == 'train':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.train import main as run
        run()
    elif cmd == 'map':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.map_labels import main as run
        run()
    elif cmd == 'resample':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.resample import main as run
        run()
    elif cmd == 'preview':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.preview_jpg import main as run
        run()
    elif cmd == 'crop':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.crop_image2seg import main as run
        run()
    elif cmd == 'largest_component':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.largest_component import main as run
        run()
    elif cmd == 'reorient':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.reorient_canonical import main as run
        run()
    elif cmd == 'cpdir':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.cpdir import main as run
        run()
    elif cmd == 'transform_seg2image':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.transform_seg2image import main as run
        run()
    elif cmd == 'predict_nnunet':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.predict_nnunet import main as run
        run()
    elif cmd.startswith('-'):
        print(f"Unknown option: {cmd}. Run 'uroseg --help' for help.", file=sys.stderr)
        sys.exit(1)
    else:
        _dispatch_model(cmd)


def _dispatch_model(model: str) -> None:
    from uroseg.utils.utils import get_all_models
    available = get_all_models()
    if model not in available:
        print(
            f"Unknown model or subcommand: '{model}'\n"
            f"Run 'uroseg --help' to see available models and commands.",
            file=sys.stderr,
        )
        sys.exit(1)
    from uroseg.commands.inference import main as run
    run()


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run full test suite**

```bash
cd /workspaces/UroSeg && python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: all existing tests pass + new tests pass. `test_no_args_exits_nonzero` is gone (was deleted/renamed in Step 1).

- [ ] **Step 6: Commit**

```bash
git add uroseg/commands/list_models.py uroseg/cli.py tests/test_cli.py
git commit -m "feat: uroseg -h / --help / list show canonical help; organ->model in messages"
```

---

### Task 2: `train_nnunet.py` + generic `train.py` + README

**Files:**
- Create: `uroseg/commands/train_nnunet.py`
- Modify: `uroseg/commands/train.py` (rewrite as generic dispatcher)
- Modify: `README.md`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `sys.argv` after `cli.py` has stripped `'train'` — so `sys.argv[1]` is the engine name when `train.main()` runs
- Produces: `train_nnunet.main()` in `uroseg.commands.train_nnunet` — called by `train.py` after stripping the engine token

---

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd /workspaces/UroSeg && python -m pytest tests/test_cli.py -v -k "train" 2>&1 | tail -10
```

Expected: `test_train_no_engine_exits_nonzero` FAILS (`'Engines:' in stderr` is false — current `train.py`'s argparse error message does not contain 'Engines:'); `test_train_unknown_engine_exits_nonzero` FAILS (`'Unknown engine' in stderr` is false — current code passes 'foo' as `organ` and argparse says "unrecognized arguments" or tries `get_model('foo')`); `test_train_help_exits_zero` FAILS (`'Engines:' in stdout` is false — current train.py's argparse help does not contain 'Engines:').

- [ ] **Step 3: Create `train_nnunet.py`**

Create `/workspaces/UroSeg/uroseg/commands/train_nnunet.py` with the content below. This is the current `train.py` content with two changes: (1) `prog='uroseg train nnunet'` added to the argparse constructor, (2) the `argv` normalization block (lines 68–71 of the original) removed and replaced with `args = parser.parse_args()`.

```python
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import uroseg.utils.utils as _utils
from uroseg.utils.utils import normalize_labels


def extract_dataset_id(nnunet_task: str) -> int:
    match = re.match(r'Dataset(\d+)_', nnunet_task)
    if not match:
        raise ValueError(f"Invalid nnunet_task format: {nnunet_task!r} — expected 'DatasetNNN_Name'")
    return int(match.group(1))


def generate_dataset_json(model: dict, images_tr_dir: Path) -> dict:
    labels = normalize_labels(model["labels"])

    all_values: set[int] = set()
    has_regions = False
    for v in labels.values():
        if isinstance(v, list):
            all_values.update(int(x) for x in v)
            has_regions = True
        elif int(v) != 0:
            all_values.add(int(v))

    dataset: dict = {
        "channel_names": {"0": "MRI"},
        "labels": labels,
        "numTraining": len(list(images_tr_dir.glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    if has_regions:
        dataset["regions_class_order"] = sorted(all_values)
    return dataset


def setup_nnunet_env(data_path: Path) -> None:
    nnunet_dir = data_path / "nnUNet"
    os.environ["nnUNet_raw"] = str(nnunet_dir / "raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_dir / "preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_dir / "results")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uroseg train nnunet',
        description="Train a UroSeg model with nnU-Net and AugLab augmentation.",
    )
    parser.add_argument("organ",
                        help="Organ name matching resources/models/<organ>.json")
    parser.add_argument("--fold", "-f", type=int, default=0,
                        help="nnU-Net fold number (0–4, default: 0)")
    parser.add_argument("--auglab-config", default=None,
                        help="Path to AugLab augmentation config JSON (optional)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--data-dir", default=None,
                        help="Override UROSEG_DATA / ~/uroseg/ with this path")
    args = parser.parse_args()

    model = _utils.get_model(args.organ)
    data_path = _utils.resolve_data_path(args.data_dir)
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

    dataset_json = generate_dataset_json(model, images_tr)
    dataset_json_path = raw_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
    print(f"Generated {dataset_json_path}")

    preprocessed_dir = data_path / "nnUNet" / "preprocessed" / nnunet_task
    if not preprocessed_dir.exists():
        print("Running nnU-Net planning and preprocessing...")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"],
            check=True,
        )

    from auglab.add_trainer import add_trainer as _add_trainer
    _add_trainer("nnUNetTrainerDAExt")

    if args.auglab_config:
        os.environ["AUGLAB_CONFIG"] = str(args.auglab_config)

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

- [ ] **Step 4: Rewrite `train.py` as generic dispatcher**

Replace the entire content of `uroseg/commands/train.py` with:

```python
from __future__ import annotations
import sys

_ENGINES = {
    'nnunet': 'Train with nnU-Net + AugLab',
}

_HELP = (
    'Usage: uroseg train <engine> ORGAN [options]\n'
    'Engines:\n'
    '  nnunet    Train with nnU-Net + AugLab'
)


def main() -> None:
    argv = sys.argv[1:]

    if argv and argv[0] in ('-h', '--help'):
        print(_HELP)
        sys.exit(0)

    if not argv:
        print(_HELP, file=sys.stderr)
        sys.exit(1)

    engine = argv[0]
    if engine not in _ENGINES:
        print(f"Unknown engine: '{engine}'\n{_HELP}", file=sys.stderr)
        sys.exit(1)

    sys.argv = sys.argv[:1] + sys.argv[2:]

    if engine == 'nnunet':
        from uroseg.commands.train_nnunet import main as run
        run()
```

- [ ] **Step 5: Update `README.md`**

Apply these exact substitutions:

| Old text | New text |
|----------|----------|
| `uroseg train ORGAN` (line 113) | `uroseg train nnunet ORGAN` |
| `` `uroseg train` automatically sets `` (line 155) | `` `uroseg train nnunet` automatically sets `` |
| `` auto-generated by uroseg train`` (line 161) | `` auto-generated by uroseg train nnunet`` |
| `uroseg train kidney` (line 176) | `uroseg train nnunet kidney` |
| `uroseg train kidney --auglab-config auglab.json` (line 179) | `uroseg train nnunet kidney --auglab-config auglab.json` |
| `uroseg train kidney --data-dir /scratch/uroseg_data` (line 182) | `uroseg train nnunet kidney --data-dir /scratch/uroseg_data` |
| `` `uroseg train` automatically:`` (line 185) | `` `uroseg train nnunet` automatically:`` |
| `uroseg train <organ>` (line 231) | `uroseg train nnunet <organ>` |

- [ ] **Step 6: Run full test suite**

```bash
cd /workspaces/UroSeg && python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: all tests pass (110+ passing).

- [ ] **Step 7: Commit**

```bash
git add uroseg/commands/train_nnunet.py uroseg/commands/train.py README.md tests/test_cli.py
git commit -m "feat: uroseg train nnunet — generic train dispatcher, rename train.py -> train_nnunet.py"
```
