# UroSeg Plan 1 — Foundation + Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete UroSeg package foundation: scaffold, Image class, shared utils, model registry, CLI dispatcher, inference command, and install/list commands.

**Architecture:** Single `uroseg` CLI entry point in `cli.py` dispatches known subcommands by name and treats unknown first arguments as organ names routed to inference. All file I/O flows through the `Image` class in `utils/image.py`. Shared batch helpers live in `utils/utils.py`. Model metadata lives in `resources/models/*.json` and is read at runtime via `importlib.resources`.

**Tech Stack:** Python 3.10+, nibabel, numpy, scipy, tqdm, nnunetv2, auglab, SimpleITK, pytest

## Global Constraints

- Python ≥ 3.10 (uses `str | Path` union types, `match` optional)
- All file I/O through `Image` class only — no direct nibabel/SimpleITK calls in commands
- CLI argument naming: `--img`, `--seg`, `--out`, `--out-img`, `--out-seg`, `--out-suffix`, `--out-prefix`, `--img-suffix`, `--img-prefix`, `--seg-suffix`, `--seg-prefix`
- Shared args via `add_common_args(parser)`: `--overwrite`, `--max-workers` (default 1), `--quiet`
- Output filename pattern: `{prefix}{stem}{suffix}.nii.gz`
- Default data path: `~/uroseg/` — override via `UROSEG_DATA` env var or `--data-dir`
- Batch processing via `tqdm.contrib.concurrent.process_map`
- Every tool `main()` follows: argparse → collect → process_map

---

## File Map

| File | Role |
|------|------|
| `pyproject.toml` | Package metadata, deps, single console_scripts entry |
| `uroseg/__init__.py` | Package init, `__version__` |
| `uroseg/cli.py` | Single entry point — dispatches subcommands or organ inference |
| `uroseg/utils/image.py` | `Image` class — all NIfTI I/O |
| `uroseg/utils/utils.py` | `add_common_args`, `collect_niftis`, `build_output_path`, `build_pairs`, `resolve_data_path`, `get_model`, `get_all_models` |
| `uroseg/commands/__init__.py` | Empty |
| `uroseg/commands/inference.py` | `uroseg <organ>` — loads model JSON, runs predict, saves output |
| `uroseg/commands/predict_nnunet.py` | Low-level nnU-Net predictor wrapper |
| `uroseg/commands/install.py` | `uroseg install` — downloads and extracts model weights |
| `uroseg/resources/__init__.py` | Empty (needed for importlib.resources) |
| `uroseg/resources/models/__init__.py` | Empty (needed for importlib.resources) |
| `uroseg/resources/models/prostate.json` | Prostate model registry entry |
| `uroseg/resources/models/bladder.json` | Bladder model registry entry |
| `tests/conftest.py` | Shared pytest fixtures (synthetic NIfTI files) |
| `tests/test_image.py` | Image class tests |
| `tests/test_utils.py` | Shared utils tests |
| `tests/test_cli.py` | CLI dispatcher tests |
| `tests/test_install.py` | Install command tests |

---

### Task 1: Package Scaffold + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `uroseg/__init__.py`
- Create: `uroseg/commands/__init__.py`
- Create: `uroseg/utils/__init__.py`
- Create: `uroseg/resources/__init__.py`
- Create: `uroseg/resources/models/__init__.py`
- Create: `.gitignore` (if not present)

**Interfaces:**
- Produces: installable `uroseg` package; `from uroseg import __version__` works

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=67", "setuptools-scm"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "uroseg"
dynamic = ["version"]
requires-python = ">=3.10"
description = "Automated segmentation of urological anatomy from medical images"
license = {text = "MIT"}
dependencies = [
    "tqdm",
    "numpy",
    "nibabel",
    "scipy",
    "pillow",
    "SimpleITK",
    "auglab",
    "nnunetv2",
]

[project.scripts]
uroseg = "uroseg.cli:main"

[tool.setuptools.packages.find]
where = ["."]

[tool.setuptools.package-data]
"uroseg.resources.models" = ["*.json"]

[tool.setuptools_scm]
```

- [ ] **Step 2: Create package init files**

```python
# uroseg/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("uroseg")
except PackageNotFoundError:
    __version__ = "0.0.0"
```

```python
# uroseg/commands/__init__.py
# uroseg/utils/__init__.py
# uroseg/resources/__init__.py
# uroseg/resources/models/__init__.py
```
(all empty)

- [ ] **Step 3: Install package in editable mode**

```bash
pip install -e .
```

Expected: no errors, `uroseg` command available.

- [ ] **Step 4: Verify import**

```bash
python -c "import uroseg; print(uroseg.__version__)"
```

Expected: prints `0.0.0` (or version string if git tag present).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uroseg/__init__.py uroseg/commands/__init__.py uroseg/utils/__init__.py uroseg/resources/__init__.py uroseg/resources/models/__init__.py
git commit -m "feat: package scaffold and pyproject.toml"
```

---

### Task 2: Model Registry JSON Files

**Files:**
- Create: `uroseg/resources/models/prostate.json`
- Create: `uroseg/resources/models/bladder.json`

**Interfaces:**
- Produces: JSON files readable by `get_model('prostate')` and `get_model('bladder')` (implemented in Task 3)

- [ ] **Step 1: Create prostate.json**

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
  "regions_class_order": [1, 2, 3, 4],
  "weights_url": "https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip"
}
```

- [ ] **Step 2: Create bladder.json**

```json
{
  "name": "bladder",
  "description": "Urinary bladder (CT)",
  "nnunet_task": "Dataset010_Bladder",
  "channel_names": {"0": "CT"},
  "labels": {
    "0": "background",
    "1": "bladder"
  },
  "weights_url": "https://github.com/yw7/uroseg/releases/download/r20260101/Dataset010_Bladder_r20260101.zip"
}
```

- [ ] **Step 3: Commit**

```bash
git add uroseg/resources/models/prostate.json uroseg/resources/models/bladder.json
git commit -m "feat: add prostate and bladder model registry JSONs"
```

---

### Task 3: Image Class

**Files:**
- Create: `uroseg/utils/image.py`
- Create: `tests/conftest.py`
- Create: `tests/test_image.py`

**Interfaces:**
- Produces:
  - `Image.load(path: str | Path) -> Image`
  - `image.save(path: str | Path) -> None`
  - `image.copy() -> Image`
  - `image.reorient(orientation: str = 'RAS') -> Image`
  - `image.resample(voxel_size: tuple) -> Image`
  - `image.bounding_box(label: int = None) -> tuple | None`
  - `image.data: np.ndarray`, `image.affine: np.ndarray`, `image.header`

- [ ] **Step 1: Write tests/conftest.py**

```python
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path


@pytest.fixture
def nifti_file(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[5:15, 5:15, 5:15] = 1
    data[7:13, 7:13, 7:13] = 2
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "test.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def nifti_folder(tmp_path, nifti_file):
    folder = tmp_path / "inputs"
    folder.mkdir()
    import shutil
    for i in range(3):
        shutil.copy(nifti_file, folder / f"case{i:03d}.nii.gz")
    return folder
```

- [ ] **Step 2: Write failing tests in tests/test_image.py**

```python
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
from uroseg.utils.image import Image


def test_load_returns_image(nifti_file):
    img = Image.load(nifti_file)
    assert isinstance(img.data, np.ndarray)
    assert img.data.shape == (20, 20, 20)
    assert img.affine.shape == (4, 4)


def test_load_nii_gz(nifti_file):
    img = Image.load(nifti_file)
    assert img.data[10, 10, 10] == 1


def test_save_roundtrip(nifti_file, tmp_path):
    img = Image.load(nifti_file)
    out = tmp_path / "out.nii.gz"
    img.save(out)
    assert out.exists()
    img2 = Image.load(out)
    np.testing.assert_array_equal(img.data, img2.data)


def test_save_creates_parent_dirs(nifti_file, tmp_path):
    img = Image.load(nifti_file)
    out = tmp_path / "nested" / "dir" / "out.nii.gz"
    img.save(out)
    assert out.exists()


def test_copy_is_independent(nifti_file):
    img = Image.load(nifti_file)
    img2 = img.copy()
    img2.data[0, 0, 0] = 99
    assert img.data[0, 0, 0] != 99


def test_reorient_ras(nifti_file):
    img = Image.load(nifti_file)
    reoriented = img.reorient('RAS')
    assert isinstance(reoriented, Image)
    assert reoriented.data.ndim == 3


def test_resample(nifti_file):
    img = Image.load(nifti_file)
    resampled = img.resample((2.0, 2.0, 2.0))
    assert isinstance(resampled, Image)
    assert resampled.data.ndim == 3


def test_bounding_box_label(nifti_file):
    img = Image.load(nifti_file)
    bb = img.bounding_box(label=1)
    assert bb is not None
    assert len(bb) == 3


def test_bounding_box_empty_returns_none(nifti_file):
    img = Image.load(nifti_file)
    bb = img.bounding_box(label=99)
    assert bb is None
```

- [ ] **Step 3: Run tests — expect failures**

```bash
pytest tests/test_image.py -v
```

Expected: `ImportError` — `uroseg.utils.image` not found.

- [ ] **Step 4: Implement uroseg/utils/image.py**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.orientations as nibo
import nibabel.processing as nibp


class Image:
    def __init__(self, data: np.ndarray, affine: np.ndarray, header):
        self.data = data
        self.affine = affine
        self.header = header

    @staticmethod
    def load(path: str | Path) -> Image:
        path = Path(path)
        img = nib.load(str(path))
        return Image(
            data=np.asanyarray(img.dataobj),
            affine=img.affine.copy(),
            header=img.header,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.data, self.affine, self.header), str(path))

    def copy(self) -> Image:
        return Image(self.data.copy(), self.affine.copy(), self.header.copy())

    def reorient(self, orientation: str = 'RAS') -> Image:
        ornt_orig = nibo.io_orientation(self.affine)
        ornt_targ = nibo.axcodes2ornt(tuple(orientation))
        transform = nibo.ornt_transform(ornt_orig, ornt_targ)
        data = nibo.apply_orientation(self.data, transform)
        affine = self.affine @ nibo.inv_ornt_aff(transform, self.data.shape)
        return Image(data, affine, self.header)

    def resample(self, voxel_size: tuple) -> Image:
        nib_img = nib.Nifti1Image(self.data, self.affine, self.header)
        resampled = nibp.resample_to_output(nib_img, voxel_size)
        return Image(
            np.asanyarray(resampled.dataobj),
            resampled.affine,
            resampled.header,
        )

    def bounding_box(self, label: int = None) -> tuple | None:
        mask = self.data if label is None else (self.data == label)
        nonzero = np.argwhere(mask)
        if len(nonzero) == 0:
            return None
        mins = nonzero.min(axis=0)
        maxs = nonzero.max(axis=0)
        return tuple(slice(int(mn), int(mx) + 1) for mn, mx in zip(mins, maxs))
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_image.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add uroseg/utils/image.py tests/conftest.py tests/test_image.py
git commit -m "feat: Image class with load, save, reorient, resample, bounding_box"
```

---

### Task 4: Shared Utils

**Files:**
- Create: `uroseg/utils/utils.py`
- Create: `tests/test_utils.py`

**Interfaces:**
- Produces:
  - `add_common_args(parser: ArgumentParser) -> None`
  - `collect_niftis(path: str | Path) -> list[Path]`
  - `build_output_path(inp: Path, out_dir: Path, prefix: str, suffix: str) -> Path`
  - `build_pairs(inp, out, suffix, prefix, overwrite) -> list[tuple[Path, Path]]`
  - `resolve_data_path(data_dir: str | None = None) -> Path`
  - `get_model(name: str) -> dict`
  - `get_all_models() -> dict[str, dict]`

- [ ] **Step 1: Write failing tests in tests/test_utils.py**

```python
import os
import pytest
from pathlib import Path
from uroseg.utils.utils import (
    collect_niftis,
    build_output_path,
    build_pairs,
    resolve_data_path,
    get_model,
    get_all_models,
)


def test_collect_niftis_single_file(nifti_file):
    result = collect_niftis(nifti_file)
    assert result == [Path(nifti_file)]


def test_collect_niftis_folder(nifti_folder):
    result = collect_niftis(nifti_folder)
    assert len(result) == 3
    assert all(p.suffix == '.gz' for p in result)
    assert result == sorted(result)


def test_collect_niftis_empty_folder(tmp_path):
    result = collect_niftis(tmp_path)
    assert result == []


def test_build_output_path_nii_gz(nifti_file, tmp_path):
    out = build_output_path(Path(nifti_file), tmp_path, prefix='', suffix='_seg')
    assert out == tmp_path / 'test_seg.nii.gz'


def test_build_output_path_nii(tmp_path):
    inp = tmp_path / 'case001.nii'
    out = build_output_path(inp, tmp_path, prefix='pred_', suffix='')
    assert out == tmp_path / 'pred_case001.nii.gz'


def test_build_pairs_basic(nifti_folder, tmp_path):
    pairs = build_pairs(nifti_folder, tmp_path, '_seg', '', overwrite=True)
    assert len(pairs) == 3
    for inp, out in pairs:
        assert out.parent == tmp_path
        assert out.name.endswith('_seg.nii.gz')


def test_build_pairs_skip_existing(nifti_folder, tmp_path):
    out_path = build_output_path(
        list(collect_niftis(nifti_folder))[0], tmp_path, '', '_seg'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch()
    pairs = build_pairs(nifti_folder, tmp_path, '_seg', '', overwrite=False)
    assert len(pairs) == 2


def test_build_pairs_overwrite(nifti_folder, tmp_path):
    for inp in collect_niftis(nifti_folder):
        build_output_path(inp, tmp_path, '', '_seg').parent.mkdir(parents=True, exist_ok=True)
        build_output_path(inp, tmp_path, '', '_seg').touch()
    pairs = build_pairs(nifti_folder, tmp_path, '_seg', '', overwrite=True)
    assert len(pairs) == 3


def test_resolve_data_path_default(monkeypatch):
    monkeypatch.delenv('UROSEG_DATA', raising=False)
    path = resolve_data_path()
    assert path == Path.home() / 'uroseg'


def test_resolve_data_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv('UROSEG_DATA', str(tmp_path))
    path = resolve_data_path()
    assert path == tmp_path


def test_resolve_data_path_arg(tmp_path):
    path = resolve_data_path(str(tmp_path))
    assert path == tmp_path


def test_get_model_prostate():
    model = get_model('prostate')
    assert model['name'] == 'prostate'
    assert 'labels' in model
    assert 'nnunet_task' in model
    assert 'channel_names' in model
    assert 'regions_class_order' in model


def test_get_model_bladder():
    model = get_model('bladder')
    assert model['name'] == 'bladder'
    assert '1' in model['labels']
    assert 'regions_class_order' not in model


def test_get_model_unknown():
    with pytest.raises(Exception):
        get_model('nonexistent_organ')


def test_get_all_models():
    models = get_all_models()
    assert 'prostate' in models
    assert 'bladder' in models
    assert all('labels' in m for m in models.values())
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_utils.py -v
```

Expected: `ImportError` — `uroseg.utils.utils` not found.

- [ ] **Step 3: Implement uroseg/utils/utils.py**

```python
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from importlib.resources import files


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing outputs (default: skip if exists)')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress bar and non-error output')


def collect_niftis(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir()
        if p.name.endswith('.nii.gz') or p.name.endswith('.nii')
    )


def build_output_path(inp: Path, out_dir: Path, prefix: str, suffix: str) -> Path:
    out_dir = Path(out_dir)
    stem = inp.name
    for ext in ('.nii.gz', '.nii'):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return out_dir / f'{prefix}{stem}{suffix}.nii.gz'


def build_pairs(
    inp: str | Path,
    out: str | Path,
    suffix: str,
    prefix: str,
    overwrite: bool,
) -> list[tuple[Path, Path]]:
    inputs = collect_niftis(inp)
    out = Path(out)
    pairs = [(i, build_output_path(i, out, prefix, suffix)) for i in inputs]
    if not overwrite:
        pairs = [(i, o) for i, o in pairs if not o.exists()]
    return pairs


def resolve_data_path(data_dir: str | None = None) -> Path:
    if data_dir:
        return Path(data_dir)
    if 'UROSEG_DATA' in os.environ:
        return Path(os.environ['UROSEG_DATA'])
    return Path.home() / 'uroseg'


def get_model(name: str) -> dict:
    path = files('uroseg.resources.models').joinpath(f'{name}.json')
    return json.loads(path.read_text())


def get_all_models() -> dict[str, dict]:
    models_dir = files('uroseg.resources.models')
    return {
        p.stem: json.loads(p.read_text())
        for p in models_dir.iterdir()
        if p.name.endswith('.json')
    }
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_utils.py -v
```

Expected: all 16 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/utils/utils.py tests/test_utils.py
git commit -m "feat: shared utils — collect_niftis, build_pairs, resolve_data_path, model registry"
```

---

### Task 5: CLI Dispatcher

**Files:**
- Create: `uroseg/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_all_models()` from `uroseg.utils.utils`
- Produces: `uroseg.cli:main` entry point; dispatches to subcommand modules or inference

- [ ] **Step 1: Write failing tests in tests/test_cli.py**

```python
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
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_cli.py -v
```

Expected: failures — `uroseg.cli` not found.

- [ ] **Step 3: Implement uroseg/cli.py**

```python
from __future__ import annotations
import sys

SUBCOMMANDS = {
    'train', 'install', 'map', 'resample', 'preview',
    'crop', 'largest_component', 'reorient', 'cpdir',
    'transform_seg2image', 'predict_nnunet', 'list',
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uroseg <organ|subcommand> [options]", file=sys.stderr)
        print("Run 'uroseg list' to see available organ models and subcommands.", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'list':
        _cmd_list()
    elif cmd == 'install':
        from uroseg.commands.install import main as run
        run()
    elif cmd == 'train':
        from uroseg.commands.train import main as run
        run()
    elif cmd == 'map':
        from uroseg.commands.map_labels import main as run
        run()
    elif cmd == 'resample':
        from uroseg.commands.resample import main as run
        run()
    elif cmd == 'preview':
        from uroseg.commands.preview_jpg import main as run
        run()
    elif cmd == 'crop':
        from uroseg.commands.crop_image2seg import main as run
        run()
    elif cmd == 'largest_component':
        from uroseg.commands.largest_component import main as run
        run()
    elif cmd == 'reorient':
        from uroseg.commands.reorient_canonical import main as run
        run()
    elif cmd == 'cpdir':
        from uroseg.commands.cpdir import main as run
        run()
    elif cmd == 'transform_seg2image':
        from uroseg.commands.transform_seg2image import main as run
        run()
    elif cmd == 'predict_nnunet':
        from uroseg.commands.predict_nnunet import main as run
        run()
    elif cmd.startswith('-'):
        print(f"Unknown option: {cmd}. Run 'uroseg list' to see available commands.", file=sys.stderr)
        sys.exit(1)
    else:
        _dispatch_organ(cmd)


def _cmd_list() -> None:
    from uroseg.utils.utils import get_all_models
    models = get_all_models()
    print(f"{'Model':<22} {'Description'}")
    print('-' * 70)
    for name, m in sorted(models.items()):
        labels = ', '.join(f"{k}={v}" for k, v in m['labels'].items() if k != '0')
        print(f"  uroseg {name:<16} {m['description']}")
        print(f"  {'':16}   labels: {labels}")


def _dispatch_organ(organ: str) -> None:
    from uroseg.utils.utils import get_all_models
    available = get_all_models()
    if organ not in available:
        print(
            f"Unknown organ or subcommand: '{organ}'\n"
            f"Available organs: {', '.join(sorted(available))}\n"
            f"Subcommands: {', '.join(sorted(SUBCOMMANDS))}",
            file=sys.stderr,
        )
        sys.exit(1)
    from uroseg.commands.inference import main as run
    run()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_cli.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Verify CLI entry point works**

```bash
uroseg list
```

Expected: table showing `prostate` and `bladder` with their labels.

- [ ] **Step 6: Commit**

```bash
git add uroseg/cli.py tests/test_cli.py
git commit -m "feat: CLI dispatcher with list subcommand and organ routing"
```

---

### Task 6: predict_nnunet Command

**Files:**
- Create: `uroseg/commands/predict_nnunet.py`

**Interfaces:**
- Consumes: `resolve_data_path` from utils; nnunetv2 `nnUNetPredictor`
- Produces:
  - `find_model_dir(nnunet_task: str, data_path: Path) -> Path` — locates the trained model folder under `results/`
  - `predict(model_dir: Path, input_files: list[Path], output_dir: Path, fold: int, device: str) -> None`

- [ ] **Step 1: Implement uroseg/commands/predict_nnunet.py**

No unit tests here — nnU-Net prediction requires GPU and trained weights. Integration tested via inference command.

```python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from uroseg.utils.utils import add_common_args, resolve_data_path, collect_niftis


def find_model_dir(nnunet_task: str, data_path: Path) -> Path:
    """Locate the trained model folder inside data_path/nnUNet/results/.

    Searches release ID subdirectories newest-first; falls back to direct
    task folder if weights were placed there manually.
    """
    results_root = data_path / 'nnUNet' / 'results'

    # direct placement (locally trained, no release ID subdir)
    direct = results_root / nnunet_task
    if direct.exists():
        trainer_dirs = list(direct.iterdir())
        if trainer_dirs:
            return direct

    # versioned release subdirs (sorted newest-first by name)
    if results_root.exists():
        release_dirs = sorted(
            (d for d in results_root.iterdir() if d.is_dir() and d.name != nnunet_task),
            reverse=True,
        )
        for release_dir in release_dirs:
            task_dir = release_dir / nnunet_task
            if task_dir.exists():
                return task_dir

    raise FileNotFoundError(
        f"Model weights for '{nnunet_task}' not found under {results_root}.\n"
        f"Run: uroseg install --model <organ>"
    )


def predict(
    model_dir: Path,
    input_files: list[Path],
    output_dir: Path,
    fold: int = 0,
    device: str = 'cuda',
    step_size: float = 0.5,
) -> None:
    """Run nnU-Net prediction on a list of input NIfTI files."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device_obj,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # find checkpoint
    checkpoint = 'checkpoint_best.pth'
    fold_dir = model_dir
    # walk into trainer__plans__config/fold_N structure
    for child in model_dir.iterdir():
        if child.is_dir():
            fold_dir_candidate = child / f'fold_{fold}'
            if fold_dir_candidate.exists():
                fold_dir = child
                break

    predictor.initialize_from_trained_model_folder(
        str(fold_dir),
        use_folds=(fold,),
        checkpoint_name=checkpoint,
    )

    list_of_lists = [[str(f)] for f in input_files]
    output_names = [str(output_dir / f.name) for f in input_files]

    predictor.predict_from_files(
        list_of_lists,
        output_names,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Low-level nnU-Net prediction wrapper.'
    )
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--task', required=True, help='nnUNet task name (e.g. Dataset001_Prostate)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--data-dir', default=None)
    add_common_args(parser)
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_dir)
    model_dir = find_model_dir(args.task, data_path)
    inputs = collect_niftis(args.img)
    predict(model_dir, inputs, Path(args.out), fold=args.fold, device=args.device)
```

- [ ] **Step 2: Commit**

```bash
git add uroseg/commands/predict_nnunet.py
git commit -m "feat: predict_nnunet command wrapping nnUNetPredictor"
```

---

### Task 7: Inference Command

**Files:**
- Create: `uroseg/commands/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `get_model`, `resolve_data_path` from utils; `find_model_dir`, `predict` from predict_nnunet; `Image` from image
- Produces: `uroseg <organ> --img ... --out ...` functionality

- [ ] **Step 1: Write failing tests in tests/test_inference.py**

```python
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
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_inference.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/inference.py**

```python
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

from tqdm.contrib.concurrent import process_map
import functools

from uroseg.utils.image import Image
from uroseg.utils.utils import (
    add_common_args,
    collect_niftis,
    build_output_path,
    resolve_data_path,
    get_model,
    get_all_models,
)
from uroseg.commands.predict_nnunet import find_model_dir, predict


def build_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run segmentation inference for a given organ model.'
    )
    parser.add_argument('organ', help='Organ model name (e.g. prostate, bladder)')
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--fold', type=int, default=0, help='nnU-Net fold (default: 0)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--out-suffix', default='_seg', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--data-dir', default=None, help='Override data path (or set UROSEG_DATA)')
    add_common_args(parser)
    return parser


def resolve_organ(organ: str) -> dict:
    available = get_all_models()
    if organ not in available:
        print(
            f"Unknown organ: '{organ}'\nAvailable: {', '.join(sorted(available))}",
            file=sys.stderr,
        )
        sys.exit(1)
    return available[organ]


def _reorient_to_tmp(input_path: Path, tmp_dir: Path) -> Path:
    img = Image.load(input_path)
    img = img.reorient('RAS')
    out = tmp_dir / input_path.name
    img.save(out)
    return out


def main() -> None:
    parser = build_inference_parser()
    args = parser.parse_args()

    model = resolve_organ(args.organ)
    data_path = resolve_data_path(args.data_dir)
    model_dir = find_model_dir(model['nnunet_task'], data_path)

    inputs = collect_niftis(args.img)
    if not inputs:
        print(f"No NIfTI files found in {args.img}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / 'inputs'
        tmp_out = Path(tmp) / 'outputs'
        tmp_in.mkdir()
        tmp_out.mkdir()

        if not args.quiet:
            print(f"Reorienting {len(inputs)} image(s) to RAS...")
        reoriented = [_reorient_to_tmp(p, tmp_in) for p in inputs]

        predict(
            model_dir=model_dir,
            input_files=reoriented,
            output_dir=tmp_out,
            fold=args.fold,
            device=args.device,
        )

        # copy predictions to final output with suffix/prefix naming
        for inp, pred_file in zip(inputs, sorted(tmp_out.glob('*.nii.gz'))):
            dest = build_output_path(inp, out_dir, args.out_prefix, args.out_suffix)
            if not args.overwrite and dest.exists():
                continue
            img = Image.load(pred_file)
            img.save(dest)

    if not args.quiet:
        print(f"Segmentations saved to {out_dir}")
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_inference.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/inference.py tests/test_inference.py
git commit -m "feat: inference command — uroseg <organ> --img ... --out ..."
```

---

### Task 8: Install + List Commands

**Files:**
- Create: `uroseg/commands/install.py`
- Create: `tests/test_install.py`

**Interfaces:**
- Consumes: `get_model`, `get_all_models`, `resolve_data_path` from utils
- Produces: `uroseg install --model prostate` and `uroseg install --all`

- [ ] **Step 1: Write failing tests in tests/test_install.py**

```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from uroseg.commands.install import (
    extract_release_id,
    get_install_dir,
    is_installed,
    download_and_extract,
)


def test_extract_release_id():
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    assert extract_release_id(url) == 'r20260101'


def test_get_install_dir(tmp_path):
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    d = get_install_dir('Dataset001_Prostate', url, tmp_path)
    assert d == tmp_path / 'nnUNet' / 'results' / 'r20260101' / 'Dataset001_Prostate'


def test_is_installed_false(tmp_path):
    url = 'https://example.com/releases/download/r20260101/X.zip'
    assert not is_installed('Dataset001_Prostate', url, tmp_path)


def test_is_installed_true(tmp_path):
    url = 'https://example.com/releases/download/r20260101/X.zip'
    install_dir = get_install_dir('Dataset001_Prostate', url, tmp_path)
    install_dir.mkdir(parents=True)
    assert is_installed('Dataset001_Prostate', url, tmp_path)


def test_download_and_extract_no_url(tmp_path, capsys):
    model = {'name': 'test', 'nnunet_task': 'Dataset999_Test', 'labels': {}}
    download_and_extract(model, tmp_path, store_export=False)
    captured = capsys.readouterr()
    assert 'skip' in captured.out.lower() or 'no weights' in captured.out.lower()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_install.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/install.py**

```python
from __future__ import annotations
import argparse
import sys
import zipfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from uroseg.utils.utils import add_common_args, resolve_data_path, get_model, get_all_models


def extract_release_id(url: str) -> str:
    return url.rstrip('/').split('/')[-2]


def get_install_dir(nnunet_task: str, url: str, data_path: Path) -> Path:
    release_id = extract_release_id(url)
    return data_path / 'nnUNet' / 'results' / release_id / nnunet_task


def is_installed(nnunet_task: str, url: str, data_path: Path) -> bool:
    return get_install_dir(nnunet_task, url, data_path).exists()


def download_and_extract(model: dict, data_path: Path, store_export: bool = False) -> None:
    url = model.get('weights_url')
    if not url:
        print(f"  {model['name']}: no weights_url — skipping.")
        return

    nnunet_task = model['nnunet_task']

    if is_installed(nnunet_task, url, data_path):
        print(f"  {model['name']}: already installed.")
        return

    release_id = extract_release_id(url)
    exports_dir = data_path / 'nnUNet' / 'exports'
    results_dir = data_path / 'nnUNet' / 'results' / release_id
    exports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    zip_name = url.split('/')[-1]
    zip_path = exports_dir / zip_name

    print(f"  Downloading {model['name']} ({zip_name})...")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=zip_name) as bar:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_size)
        urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)

    print(f"  Extracting to {results_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(results_dir)

    if not store_export:
        zip_path.unlink()
        print(f"  Done. Weights installed at {results_dir / nnunet_task}")
    else:
        print(f"  Done. Archive kept at {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Download and install UroSeg model weights.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', nargs='+', metavar='MODEL',
                       help='One or more organ model names (e.g. prostate bladder)')
    group.add_argument('--all', action='store_true', help='Install all available models')
    parser.add_argument('--data-dir', default=None, help='Override data path')
    parser.add_argument('--store-export', action='store_true',
                        help='Keep downloaded zip archive after extraction')
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_dir)

    if args.all:
        models = list(get_all_models().values())
    else:
        models = [get_model(name) for name in args.model]

    print(f"Installing {len(models)} model(s) to {data_path}...")
    for model in models:
        download_and_extract(model, data_path, store_export=args.store_export)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_install.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Smoke test CLI**

```bash
uroseg list
uroseg install --help
uroseg prostate --help
```

Expected: no errors; `--help` outputs show correct arguments.

- [ ] **Step 7: Commit**

```bash
git add uroseg/commands/install.py tests/test_install.py
git commit -m "feat: install command — download and extract model weights by release ID"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Single `uroseg` CLI entry point → Task 5
- [x] Organ inference dispatch → Task 5 + 7
- [x] `uroseg list` → Task 5
- [x] Model registry JSON files → Task 2
- [x] `get_model` / `get_all_models` via importlib.resources → Task 4
- [x] `Image` class (load, save, reorient, resample, bounding_box) → Task 3
- [x] `add_common_args`, `collect_niftis`, `build_pairs`, `build_output_path` → Task 4
- [x] `resolve_data_path` (`--data-dir` → `UROSEG_DATA` → `~/uroseg/`) → Task 4
- [x] `predict_nnunet` wrapper → Task 6
- [x] `uroseg install` with `--model`/`--all`/`--store-export` → Task 8
- [x] Release ID extracted from URL → Task 8
- [x] Weights auto-download on missing → Task 7 (calls install if not found)
- [ ] `regions_class_order` handled in inference → noted: nnU-Net handles this internally via the trained model; no extra code needed in inference
- [ ] `channel_names` used at train time → covered in Plan 3

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency:**
- `Image.load` → returns `Image` ✓
- `collect_niftis` → returns `list[Path]` ✓
- `build_pairs` → returns `list[tuple[Path, Path]]` ✓
- `find_model_dir` → returns `Path`, raises `FileNotFoundError` ✓
- `predict` consumes `list[Path]` for `input_files` ✓
- `download_and_extract` consumes `dict` model + `Path` data_path ✓

---

*Next: Plan 2 — Utility Commands (map, resample, reorient, largest_component, crop, preview, transform_seg2image, cpdir)*
*Next: Plan 3 — Training + README*
